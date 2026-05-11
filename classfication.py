import os
import pandas as pd

# ========= 你只要改這裡 =========
INPUT_XLSX = r"C:\Users\felicia\OneDrive\Desktop\.ipynb_checkpoints\data\whole data.xlsx"  # 例如 r"C:\Users\xxx\Desktop\whole data.xlsx"
TIME_COL = "time"               # 如果你的時間欄位不是 time，改成你的欄位名
# ===============================

GROUPS = {
    "Group1": ["TI-644B1.PV", "TI-825B.PV", "TI-644B2.PV", "TI-644B3.PV", "TIC-640B.PV"],
    "Group2": ["TI-644B5.PV", "TI-829B.PV", "TI-645B2.PV", "TI-645B3.PV", "TIC-641B.PV"],
    "Group3": ["TI-646B1.PV", "TI-834B.PV", "TI-646B2.PV", "TI-646B3.PV", "TIC-642B.PV"],
    "Group4": ["TI-647B1.PV", "TI-839B.PV", "TI-647B2.PV", "TI-647B3.PV", "TIC-643B.PV"],
}

def main():
    if not os.path.exists(INPUT_XLSX):
        raise FileNotFoundError(f"找不到檔案：{INPUT_XLSX}")

    # 讀取整本 Excel 的第一個 sheet 當作原始數據來源
    df = pd.read_excel(INPUT_XLSX, sheet_name=0)

    # 開始把內容寫回同一個檔案（覆寫式輸出）
    # 注意：這會重寫整本檔案；第 1 頁會寫成 RawData，其它分頁依序寫入
    with pd.ExcelWriter(INPUT_XLSX, engine="openpyxl", mode="w") as writer:
        # 第 1 頁：原始數據
        df.to_excel(writer, sheet_name="RawData", index=False)

        # 第 2~5 頁：分組欄位
        for sheet_name, cols in GROUPS.items():
            use_cols = []
            if TIME_COL in df.columns:
                use_cols.append(TIME_COL)

            missing = [c for c in cols if c not in df.columns]
            present = [c for c in cols if c in df.columns]
            use_cols.extend(present)

            if missing:
                print(f"[警告] {sheet_name} 缺少欄位：{missing}")

            out_df = df[use_cols].copy()
            out_df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"完成：已在同一個檔案內建立分頁 -> {INPUT_XLSX}")
    print("分頁：RawData, Group1, Group2, Group3, Group4")
if __name__ == "__main__":
    main()



