import pandas as pd
import numpy as np
from pathlib import Path

INPUT_XLSX = Path(r"C:\Users\felicia\OneDrive\Desktop\.ipynb_checkpoints\data\whole data.xlsx")
OUTPUT_XLSX = Path(r"C:\Users\felicia\OneDrive\Desktop\.ipynb_checkpoints\data\whole data_labeled.xlsx")

TIME_COL = "time"

HOT = ["TI-644B1.PV", "TI-645B1.PV", "TI-646B1.PV", "TI-647B1.PV"]
COLD = ["TI-825B.PV", "TI-829B.PV", "TI-834B.PV", "TI-839B.PV"]
GROUP_NAMES = ["Group 1", "Group 2", "Group 3", "Group 4"]

# ---------------------------
# 可調參數
# ---------------------------
TH_HOT = 195.0          # 熱流門檻
HOURS_CONFIRM = 48     # 連續幾小時才確認切狀態（你資料每小時一筆，所以=24就是24小時）
COLD_JUMP_TH = 50.0      # 冷流出口「突然上升」的差分門檻（每小時上升 >= 2°C）
OSC_STD_TH = 6.0        # 震盪：24h rolling std >= 1°C
OSC_RANGE_TH = 10.0      # 震盪：24h rolling range >= 4°C
RECENT_JUMP_BOOST_HRS = 20  # 最近 N 小時出現冷流 jump 就加權（用來處理多台同時被判 0）

# =========================
# 4) 工具函式
# =========================
def to_num(s: pd.Series) -> pd.Series:
    """把 'No Good Data ...' 等字串轉成 NaN，避免轉 float 失敗"""
    return pd.to_numeric(s, errors="coerce")

def rolling_all_true(cond: pd.Series, window: int) -> pd.Series:
    """
    window 內全部為 True 才回 True（需滿足完整 window 長度）
    回傳 True 的點出現在該 window 的最後一個 index（這就是確認點）
    """
    cond = cond.astype(bool)
    return (
        cond.rolling(window=window, min_periods=window)
        .apply(lambda x: 1.0 if np.all(x) else 0.0, raw=True)
        .astype(bool)
    )

def detect_one_machine(hot_raw: pd.Series, cold_raw: pd.Series) -> pd.DataFrame:
    """
    單台機器：輸出 raw state（1正常/0保養） + score（用來多台同時保養時挑一台）
    規則：
      - cold_jump → 直接開始保養（不回填）
      - (oscillating & below_24h) → 開始保養（不回填）
      - above_24h → 結束保養，且「只在這裡做回填」：把前 24 小時一起填成正常(1)
    """
    hot = to_num(hot_raw)
    cold = to_num(cold_raw)

    # 冷流突然上升（開始保養的強訊號）
    cold_jump = (cold.diff().fillna(0.0) >= COLD_JUMP_TH)

    # 熱流條件
    below = (hot < TH_HOT).fillna(False)
    above = (hot > TH_HOT).fillna(False)

    below_24h = rolling_all_true(below, HOURS_CONFIRM)   # 確認點：已連續 24h < TH_HOT
    above_24h = rolling_all_true(above, HOURS_CONFIRM)   # 確認點：已連續 24h > TH_HOT

    # 熱流震盪（24h rolling std + range）
    roll_std = hot.rolling(window=HOURS_CONFIRM, min_periods=HOURS_CONFIRM).std()
    roll_rng = hot.rolling(window=HOURS_CONFIRM, min_periods=HOURS_CONFIRM).apply(
        lambda x: float(np.nanmax(x) - np.nanmin(x)), raw=True
    )
    osc_24h = ((roll_std >= OSC_STD_TH) & (roll_rng >= OSC_RANGE_TH)).fillna(False)

    # 開始保養：cold_jump OR (震盪且已連續 24h < 190)
    start_maint = cold_jump | (osc_24h & below_24h)

    # 結束保養（恢復正常）：已連續 24h > 190
    end_maint = above_24h

    # 狀態機：初始化都當正常
    n = len(hot)
    state = np.ones(n, dtype=int)
    cur = 1  # 1=正常, 0=保養

    for i in range(n):
        # 先處理保養開始（不回填）
        if start_maint.iat[i]:
            cur = 0

        # 再處理保養結束（要回填前 24 小時為正常）
        if end_maint.iat[i]:
            start_idx = max(0, i - HOURS_CONFIRM + 1)
            state[start_idx:i+1] = 1   # ✅只回填「恢復正常」的 24 小時
            cur = 1

        # 當下狀態寫入
        state[i] = cur

    # score：用來「同一時間多台被判保養」時挑最像保養的那台
    recent_jump = cold_jump.rolling(window=RECENT_JUMP_BOOST_HRS, min_periods=1).max().astype(int)
    hot_fill = hot.fillna(hot.median())

    # 權重：recent_jump(超大) > hot 越低越像保養 > cold_diff 越大越像保養
    score = recent_jump * 1_000_000 + (-hot_fill) * 1000 + cold.diff().fillna(0.0) * 10

    return pd.DataFrame({
        "state_raw": state,                 # 1正常/0保養（尚未套用「一次只允許一台」）
        "score": score.astype(float),
        "cold_jump": cold_jump.astype(int), # 方便debug（可留可刪）
        "osc_24h": osc_24h.astype(int),     # 方便debug（可留可刪）
        "below_24h": below_24h.astype(int), # 方便debug（可留可刪）
        "above_24h": above_24h.astype(int), # 方便debug（可留可刪）
    })

# =========================
# 5) 主程式
# =========================
def main():
    # 讀取第一個分頁（原始數據）
    xl = pd.ExcelFile(INPUT_XLSX)
    raw_sheet = xl.sheet_names[0]
    df = xl.parse(raw_sheet)

    # 時間欄位（若存在就嘗試轉 datetime）
    if TIME_COL in df.columns:
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")

    # 檢查欄位
    missing = [c for c in HOT + COLD if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in Excel: {missing}")

    # 每台先偵測 raw state + score
    det_list = []
    for i in range(4):
        det = detect_one_machine(df[HOT[i]], df[COLD[i]])
        det.columns = [f"m{i+1}_{c}" for c in det.columns]
        det_list.append(det)

    det_all = pd.concat(det_list, axis=1)

    raw_states = np.vstack([det_all[f"m{i+1}_state_raw"].to_numpy() for i in range(4)]).T  # (N,4)
    scores = np.vstack([det_all[f"m{i+1}_score"].to_numpy() for i in range(4)]).T

    # ✅ 套用規則：四台同時最多只會有一台在保養
    final_states = raw_states.copy()
    for t in range(final_states.shape[0]):
        off_idx = np.where(final_states[t] == 0)[0]
        if len(off_idx) <= 100:
            continue

        # 多台都說自己保養 → 用 score 選最可能那台
        best = off_idx[np.argmax(scores[t, off_idx])]
        final_states[t, :] = 1
        final_states[t, best] = 0

    # 把 label 加回原始 df
    for i in range(4):
        df[f"Group_{i+1}_label"] = final_states[:, i]  # 1正常/0保養

    # 輸出 Excel：原始分頁 + Group 1~4
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=raw_sheet[:31], index=False)

        for i in range(4):
            cols = ([TIME_COL] if TIME_COL in df.columns else []) + [HOT[i], COLD[i], f"Group_{i+1}_label"]
            out = df[cols].copy()
            out.rename(columns={f"Group_{i+1}_label": "label(0=保養,1=正常)"}, inplace=True)
            out.to_excel(writer, sheet_name=GROUP_NAMES[i], index=False)

    print(f"Saved output to: {OUTPUT_XLSX}")

if __name__ == "__main__":
    main()