import pandas as pd

# =========================================================
# 1. 基本設定
# =========================================================
INPUT_FILE = r"C:\Users\felicia\OneDrive\Desktop\.ipynb_checkpoints\data\TH195train\whole data_TH195_K48-1.xlsx"
OUTPUT_FILE = r"C:\Users\felicia\OneDrive\Desktop\.ipynb_checkpoints\data\TH195train\whole data_TH195_after72hr.xlsx"

GROUP_SHEETS = ["Group 1", "Group 2", "Group 3", "Group 4"]

TIME_COL = "time"
CURRENT_LABEL_COL = "label(0=保養,1=正常)"
AFTER_COL = "after72hr"

HORIZON_HOURS = 72


# =========================================================
# 2. 主流程
# =========================================================
with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    for sheet in GROUP_SHEETS:
        print(f"Processing {sheet}...")

        df = pd.read_excel(INPUT_FILE, sheet_name=sheet)

        # 避免欄位名稱前後有空白
        df.columns = df.columns.str.strip()

        if TIME_COL not in df.columns:
            raise ValueError(f"{sheet} 缺少時間欄位: {TIME_COL}")

        if CURRENT_LABEL_COL not in df.columns:
            raise ValueError(f"{sheet} 缺少 label 欄位: {CURRENT_LABEL_COL}")

        # 時間排序
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
        df = df.dropna(subset=[TIME_COL])
        df = df.sort_values(TIME_COL).reset_index(drop=True)

        # label 數值化
        df[CURRENT_LABEL_COL] = pd.to_numeric(
            df[CURRENT_LABEL_COL],
            errors="coerce"
        )

        # 建立 after72hr
        df[AFTER_COL] = df[CURRENT_LABEL_COL].shift(-HORIZON_HOURS)

        # 最後 72 筆沒有未來資料，所以會是 NaN
        # 如果你要保留，這行不要打開
        # 如果你要刪掉，打開下面這行
        # df = df.dropna(subset=[AFTER_COL]).reset_index(drop=True)

        # after72hr 轉成 Int64，可保留 NaN
        df[AFTER_COL] = df[AFTER_COL].astype("Int64")

        print("目前 label 分布:")
        print(df[CURRENT_LABEL_COL].value_counts(dropna=False).sort_index())

        print("after72hr 分布:")
        print(df[AFTER_COL].value_counts(dropna=False).sort_index())
        print()

        df.to_excel(writer, sheet_name=sheet, index=False)

print(f"完成，已輸出：{OUTPUT_FILE}")