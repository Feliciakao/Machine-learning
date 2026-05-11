import os
import numpy as np
import pandas as pd

# pip install hmmlearn openpyxl
from hmmlearn.hmm import GaussianHMM


# =========================================================
# 1) 檔案路徑（請修改）
# =========================================================
INPUT_XLSX = r"C:\Users\felicia\OneDrive\Desktop\.ipynb_checkpoints\data\whole data.xlsx"
OUTPUT_XLSX = None  # None => 產生 whole data_hmm.xlsx


# =========================================================
# 2) 欄位定義（依你之前的分組）
# =========================================================
TIME_COL_CANDIDATES = ["time", "Time", "timestamp", "Datetime", "datetime", "DateTime"]

GROUPS = {
    "Row1": {
        "hot_in": "TI-644B1.PV",
        "cold_out": "TI-825B.PV",
        "hot_out": "TI-644B2.PV",
        "cold_in": "TI-644B3.PV",
        "tic": "TIC-640B.PV",
    },
    "Row2": {
        "hot_in": "TI-645B1.PV",
        "cold_out": "TI-829B.PV",
        "hot_out": "TI-645B2.PV",
        "cold_in": "TI-645B3.PV",
        "tic": "TIC-641B.PV",
    },
    "Row3": {
        "hot_in": "TI-646B1.PV",
        "cold_out": "TI-834B.PV",
        "hot_out": "TI-646B2.PV",
        "cold_in": "TI-646B3.PV",
        "tic": "TIC-642B.PV",
    },
    "Row4": {
        "hot_in": "TI-647B1.PV",
        "cold_out": "TI-839B.PV",
        "hot_out": "TI-647B2.PV",
        "cold_in": "TI-647B3.PV",
        "tic": "TIC-643B.PV",
    },
}


# =========================================================
# 3) HMM 與後處理參數（可調）
# =========================================================
ROLL_HRS = 48                # rolling window
START_JUMP_Q = 0.995         # cold_out jump 的高分位數 (越高越嚴格)
MIN_MAINT_HRS = 24            # 小於這個小時數的段落會被移除
MERGE_GAP_HRS = 24            # 兩段保養間隔 <= 2 小時會合併

# 3-state：Normal(0) -> Start(1) -> Steady(2) -> Normal(0)
TRANS = np.array(
    [
        [0.997, 0.003, 0.000],  # Normal -> Normal/Start
        [0.020, 0.050, 0.930],  # Start  -> Normal/Start/Steady (Start 要短)
        [0.020, 0.000, 0.980],  # Steady -> Normal/Steady
    ],
    dtype=float,
)

STARTPROB = np.array([0.995, 0.003, 0.002], dtype=float)


# =========================================================
# 4) 工具函式（資料清洗 / 特徵 / HMM / 後處理）
# =========================================================
def pick_time_col(df: pd.DataFrame) -> str:
    for c in TIME_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"找不到時間欄位，請確認是否有：{TIME_COL_CANDIDATES}")


def safe_to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def to_numeric_series(s: pd.Series) -> pd.Series:
    """
    把像 ' No Good Data For This Time -11059' 這種字串轉成 NaN。
    然後用前向/後向補值，避免 HMM 看到 NaN。
    """
    s_str = s.astype(str).str.strip()
    s_num = pd.to_numeric(s_str, errors="coerce")
    s_num = s_num.ffill().bfill()
    return s_num


def rolling_median(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=max(3, w // 4)).median()


def rolling_std(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=max(3, w // 4)).std()


def zscore_by_col(X: np.ndarray) -> np.ndarray:
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return (X - mu) / sd


def contiguous_segments(mask: np.ndarray):
    segs = []
    n = len(mask)
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1]:
            j += 1
        segs.append((i, j))
        i = j + 1
    return segs


def apply_min_len_and_merge(mask: np.ndarray, min_len: int, merge_gap: int) -> np.ndarray:
    mask2 = mask.copy()

    # 1) 移除過短段
    segs = contiguous_segments(mask2)
    for a, b in segs:
        if (b - a + 1) < min_len:
            mask2[a : b + 1] = False

    # 2) 合併小間隔
    segs = contiguous_segments(mask2)
    if len(segs) <= 1:
        return mask2

    cur_a, cur_b = segs[0]
    for a, b in segs[1:]:
        gap = a - cur_b - 1
        if gap <= merge_gap:
            mask2[cur_b + 1 : a] = True  # bridge
            cur_b = b
        else:
            cur_a, cur_b = a, b

    return mask2


def build_features(df: pd.DataFrame, hot_in: str, cold_out: str):
    """
    4 維特徵：
      x1 cold jump (cold_out diff)
      x2 cold baseline diff (cold_out - rolling median)
      x3 hot oscillation (rolling std of hot_in)
      x4 hot diff (hot_in diff)
    """
    hot = to_numeric_series(df[hot_in])
    cold = to_numeric_series(df[cold_out])

    x1 = cold.diff()
    x2 = cold - rolling_median(cold, ROLL_HRS)
    x3 = rolling_std(hot, ROLL_HRS)
    x4 = hot.diff()

    X = np.vstack([x1.values, x2.values, x3.values, x4.values]).T

    # Start 候選：只看正向 jump 的高分位
    x1_pos = x1.copy()
    x1_pos[x1_pos < 0] = np.nan
    thr = np.nanquantile(x1_pos.values, START_JUMP_Q)
    start_jump_mask = (x1.values >= thr) & np.isfinite(x1.values)

    return X, start_jump_mask


def hmm_fit_decode(features: np.ndarray, start_jump_mask: np.ndarray):
    """
    回傳:
      labels (N,) str: Normal/Start/Steady
      maint_score (N,) float: 越大越像保養
    """
    X = features.copy()

    # hmmlearn 不能吃 NaN：用欄均值補
    col_mu = np.nanmean(X, axis=0)
    nan_pos = np.where(np.isnan(X))
    X[nan_pos] = np.take(col_mu, nan_pos[1])

    Xz = zscore_by_col(X)

    N, D = Xz.shape
    idx_start = np.where(start_jump_mask)[0]
    idx_other = np.where(~start_jump_mask)[0]

    # 防呆：Start 太少 -> 用第 1 維最大的一小撮當 Start
    if len(idx_start) < max(10, N // 500):
        k = max(10, N // 200)
        idx_start = np.argsort(Xz[:, 0])[-k:]
        idx_other = np.array([i for i in range(N) if i not in set(idx_start)], dtype=int)

    mean_start = np.mean(Xz[idx_start], axis=0)
    mean_norm = np.mean(Xz[idx_other], axis=0)

    # Steady 初值：hot oscillation(第3維) 大的一小撮
    idx_steady = np.argsort(Xz[:, 2])[-max(20, N // 200) :]
    mean_steady = np.mean(Xz[idx_steady], axis=0)

    means_init = np.vstack([mean_norm, mean_start, mean_steady])

    var = np.var(Xz, axis=0)
    var = np.where(var < 1e-4, 1.0, var)
    covars_init = np.vstack([var, var, var])

    model = GaussianHMM(
        n_components=3,
        covariance_type="diag",
        n_iter=200,
        tol=1e-3,
        init_params="",
        params="mc",  # 只更新 mean/covar，固定 startprob/transmat
        random_state=0,
    )
    model.startprob_ = STARTPROB.copy()
    model.transmat_ = TRANS.copy()
    model.means_ = means_init.copy()
    model.covars_ = covars_init.copy()

    model.fit(Xz)
    states = model.predict(Xz)
    m = model.means_

    # 自動辨識：
    # Start = cold jump(第1維) mean 最大
    start_k = int(np.argmax(m[:, 0]))
    remain = [k for k in range(3) if k != start_k]
    # Steady = hot osc(第3維) mean 最大（排除 Start）
    steady_k = remain[int(np.argmax(m[remain, 2]))]
    normal_k = [k for k in range(3) if k not in (start_k, steady_k)][0]

    labels = np.empty_like(states, dtype=object)
    labels[states == normal_k] = "Normal"
    labels[states == start_k] = "Start"
    labels[states == steady_k] = "Steady"

    # 分數：用距離 proxy（工程上穩）

    def neg_mahalanobis_diag(x, mean, cov):
        cov = np.asarray(cov)
    # cov 可能是 full matrix -> 取對角
        if cov.ndim == 2:
          cov = np.diag(cov)

    # cov 可能是 scalar(spherical) -> 展開成 (D,)
        if cov.ndim == 0:
         cov = np.full((x.shape[1],), float(cov))

        cov = cov.reshape(-1)  # 保證 (D,)
        return -np.sum((x - mean) ** 2 / (cov + 1e-6), axis=1)

    ll_start = neg_mahalanobis_diag(Xz, m[start_k], model.covars_[start_k]) 
    ll_steady = neg_mahalanobis_diag(Xz, m[steady_k], model.covars_[steady_k])
    ll_norm = neg_mahalanobis_diag(Xz, m[normal_k], model.covars_[normal_k])

    maint_score = np.maximum(ll_start, ll_steady) - ll_norm

    return labels, maint_score


# =========================================================
# 5) 主流程
# =========================================================
def main():
    if OUTPUT_XLSX is None:
        base, ext = os.path.splitext(INPUT_XLSX)
        out_path = base + "_hmm" + ext
    else:
        out_path = OUTPUT_XLSX

    xls = pd.ExcelFile(INPUT_XLSX)
    df = pd.read_excel(INPUT_XLSX, sheet_name=xls.sheet_names[0])

    time_col = pick_time_col(df)
    df[time_col] = safe_to_datetime(df[time_col])

    # 欄位檢查
    missing = []
    for g, spec in GROUPS.items():
        for key in ["hot_in", "cold_out"]:
            col = spec[key]
            if col not in df.columns:
                missing.append(f"{g}.{key} missing: {col}")
    if missing:
        raise ValueError("欄位缺失：\n" + "\n".join(missing))

    results = pd.DataFrame({time_col: df[time_col]})
    seg_rows = []

    # 每排跑 HMM
    score_list = []
    pp_masks = {}

    for gname, spec in GROUPS.items():
        X, start_mask = build_features(df, spec["hot_in"], spec["cold_out"])
        labels, score = hmm_fit_decode(X, start_mask)

        maint_raw = (labels != "Normal")
        maint_pp = apply_min_len_and_merge(maint_raw.astype(bool), MIN_MAINT_HRS, MERGE_GAP_HRS)

        results[f"{gname}_state"] = labels
        results[f"{gname}_maint_pp"] = maint_pp.astype(int)
        results[f"{gname}_score"] = score

        score_list.append(score)
        pp_masks[gname] = maint_pp

    # =====================================================
    # 互斥：同時間只允許一排在保養（選 score 最大者）
    # 若全部都不夠像保養 => 全部 0
    # =====================================================
    score_mat = np.vstack(score_list).T  # (N,4)
    gnames = list(GROUPS.keys())

    global_thr = np.nanquantile(score_mat, 0.95)  # 可調：0.90~0.97
    chosen = np.argmax(score_mat, axis=1)

    excl = np.zeros_like(score_mat, dtype=int)
    for t in range(score_mat.shape[0]):
        k = chosen[t]
        if score_mat[t, k] >= global_thr:
            excl[t, k] = 1

    for i, g in enumerate(gnames):
        results[f"{g}_maint_excl"] = (excl[:, i] & results[f"{g}_maint_pp"].values).astype(int)

    # =====================================================
    # 產出段落表
    # =====================================================
    for g in gnames:
        m = results[f"{g}_maint_excl"].values.astype(bool)
        segs = contiguous_segments(m)
        for a, b in segs:
            seg_rows.append(
                {
                    "Row": g,
                    "StartTime": results.loc[a, time_col],
                    "EndTime": results.loc[b, time_col],
                    "Hours": int(b - a + 1),
                }
            )

    seg_df = (
        pd.DataFrame(seg_rows).sort_values(["Row", "StartTime"])
        if seg_rows
        else pd.DataFrame(columns=["Row", "StartTime", "EndTime", "Hours"])
    )

    # =====================================================
    # 寫出 Excel（保留原本所有 sheet，再加兩張結果）
    # =====================================================
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        # 保留所有原 sheet
        for sh in xls.sheet_names:
            tmp = pd.read_excel(INPUT_XLSX, sheet_name=sh)
            tmp.to_excel(writer, sheet_name=sh, index=False)

        results.to_excel(writer, sheet_name="HMM_result", index=False)
        seg_df.to_excel(writer, sheet_name="Maintenance_segments", index=False)

    print(f"Done. Output saved to: {out_path}")


if __name__ == "__main__":
    main()