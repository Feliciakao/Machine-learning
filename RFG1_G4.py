import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# =========================================================
# 1) 基本設定
# =========================================================
FILE_PATH = r"C:\Users\felicia\OneDrive\Desktop\.ipynb_checkpoints\data\TH195train\whole data_TH195_K48.xlsx"

GROUP_SHEETS = ["Group 1", "Group 2", "Group 3", "Group 4"]

TIME_COL = "time"
LABEL_COL = "label(0=保養,1=正常)"

SENSOR_COLS = [
    "heatflow_in",
    "heatflow_out",
    "heatflow_mid",
    "coldflow2",
    "coldflow1"
]

TEST_RATIO = 0.2

# 每小時一筆
ROLL_WINDOWS = [3, 6, 12, 24, 48]
DIFF_LAGS = [1, 3, 6, 12, 24]
SLOPE_WINDOWS = [6, 12, 24, 48]

THRESHOLD = 0.6

RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 12,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "random_state": 42,
    "n_jobs": -1,
    "class_weight": "balanced"
}

OUTPUT_PRED_CSV = "G1_G4_rf_predictions_merged.csv"
OUTPUT_IMPORTANCE_CSV = "G1_G4_rf_feature_importance_merged.csv"


# =========================================================
# 2) 工具函式
# =========================================================
def rolling_slope(arr):
    arr = np.asarray(arr, dtype=float)

    if np.isnan(arr).any():
        return np.nan

    x = np.arange(len(arr), dtype=float)
    x_mean = x.mean()
    y_mean = arr.mean()

    denom = np.sum((x - x_mean) ** 2)
    if denom == 0:
        return 0.0

    slope = np.sum((x - x_mean) * (arr - y_mean)) / denom
    return slope


def create_time_features(df, time_col):
    out = pd.DataFrame(index=df.index)
    out["hour"] = df[time_col].dt.hour
    out["dayofweek"] = df[time_col].dt.dayofweek
    out["month"] = df[time_col].dt.month
    return out


def create_ts_features(df, base_cols, diff_lags, roll_windows, slope_windows):
    feat_dict = {}

    for col in base_cols:
        s = df[col]

        for lag in diff_lags:
            feat_dict[f"{col}_diff_{lag}"] = s.diff(lag)

        for w in roll_windows:
            r = s.rolling(window=w, min_periods=w)
            feat_dict[f"{col}_mean_{w}"] = r.mean()
            feat_dict[f"{col}_std_{w}"] = r.std()
            feat_dict[f"{col}_min_{w}"] = r.min()
            feat_dict[f"{col}_max_{w}"] = r.max()
            feat_dict[f"{col}_range_{w}"] = r.max() - r.min()

        for w in slope_windows:
            feat_dict[f"{col}_slope_{w}"] = (
                s.rolling(window=w, min_periods=w)
                 .apply(rolling_slope, raw=True)
            )

    return pd.DataFrame(feat_dict, index=df.index)


def chronological_split(df, test_ratio=0.2):
    n = len(df)
    split_idx = int(n * (1 - test_ratio))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


def load_one_group(file_path, sheet_name, group_id):
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce")

    for col in SENSOR_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[TIME_COL, LABEL_COL]).copy()
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    base_df = df[[TIME_COL, LABEL_COL] + SENSOR_COLS].copy()

    # 類別型 group 名稱
    base_df["group_name"] = sheet_name
    # 數值型 group id
    base_df["group_id"] = group_id

    time_feat_df = create_time_features(df, TIME_COL)
    ts_feat_df = create_ts_features(
        df=df,
        base_cols=SENSOR_COLS,
        diff_lags=DIFF_LAGS,
        roll_windows=ROLL_WINDOWS,
        slope_windows=SLOPE_WINDOWS
    )

    df_feat = pd.concat([base_df, time_feat_df, ts_feat_df], axis=1).copy()
    return df_feat


# =========================================================
# 3) 主流程
# =========================================================
def main():
    all_train = []
    all_test = []

    print("==== 開始讀取 G1 ~ G4 ====\n")

    for idx, sheet in enumerate(GROUP_SHEETS, start=1):
        print(f"--- 處理 {sheet} ---")

        df_feat = load_one_group(FILE_PATH, sheet, idx)

        print(f"資料筆數: {len(df_feat)}")
        print(f"時間範圍: {df_feat[TIME_COL].min()} ~ {df_feat[TIME_COL].max()}")
        print("Label 分布：")
        print(df_feat[LABEL_COL].value_counts().sort_index())
        print()

        train_df, test_df = chronological_split(df_feat, TEST_RATIO)

        print(f"{sheet} Train 筆數: {len(train_df)}")
        print(f"{sheet} Test  筆數: {len(test_df)}")
        print()

        all_train.append(train_df)
        all_test.append(test_df)

    # -----------------------------------------------------
    # 3.1 合併 train/test
    # -----------------------------------------------------
    train_merged = pd.concat(all_train, axis=0, ignore_index=True)
    test_merged = pd.concat(all_test, axis=0, ignore_index=True)

    print("==== 合併完成 ====")
    print(f"Train 總筆數: {len(train_merged)}")
    print(f"Test  總筆數: {len(test_merged)}")

    print("\nTrain label 分布：")
    print(train_merged[LABEL_COL].value_counts().sort_index())

    print("\nTest label 分布：")
    print(test_merged[LABEL_COL].value_counts().sort_index())

    print("\nTrain group 分布：")
    print(train_merged["group_name"].value_counts().sort_index())

    print("\nTest group 分布：")
    print(test_merged["group_name"].value_counts().sort_index())

    # -----------------------------------------------------
    # 3.2 建立特徵欄
    # 不使用 group_name（字串）
    # 保留 group_id（數值）
    # -----------------------------------------------------
    exclude_cols = [TIME_COL, LABEL_COL, "group_name"]
    feature_cols = [c for c in train_merged.columns if c not in exclude_cols]

    print("\n==== 特徵工程完成 ====")
    print(f"特徵數量: {len(feature_cols)}")
    print("本版本：G1~G4 合併訓練，保留 group_id，不使用差值特徵")

    X_train = train_merged[feature_cols]
    y_train = train_merged[LABEL_COL].astype(int)

    X_test = test_merged[feature_cols]
    y_test = test_merged[LABEL_COL].astype(int)

    # -----------------------------------------------------
    # 3.3 建模
    # -----------------------------------------------------
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(**RF_PARAMS))
    ])

    model.fit(X_train, y_train)

    # -----------------------------------------------------
    # 3.4 預測
    # -----------------------------------------------------
    y_train_prob = model.predict_proba(X_train)[:, 1]
    y_test_prob = model.predict_proba(X_test)[:, 1]

    y_train_pred = (y_train_prob > THRESHOLD).astype(int)
    y_test_pred = (y_test_prob > THRESHOLD).astype(int)

    # -----------------------------------------------------
    # 3.5 評估
    # -----------------------------------------------------
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    train_f1 = f1_score(y_train, y_train_pred, average="weighted")
    test_f1 = f1_score(y_test, y_test_pred, average="weighted")

    print("\n==== 模型結果 ====")
    print(f"Threshold      : {THRESHOLD}")
    print(f"Train Accuracy : {train_acc:.4f}")
    print(f"Test  Accuracy : {test_acc:.4f}")
    print(f"Train F1-score : {train_f1:.4f}")
    print(f"Test  F1-score : {test_f1:.4f}")

    print("\n==== Confusion Matrix (Test) ====")
    print(confusion_matrix(y_test, y_test_pred))

    print("\n==== Classification Report (Test) ====")
    print(classification_report(y_test, y_test_pred, digits=4))

    # -----------------------------------------------------
    # 3.6 各 group 個別成績
    # -----------------------------------------------------
    print("\n==== 各 Group 測試結果 ====")
    for sheet in GROUP_SHEETS:
        sub = test_merged["group_name"] == sheet
        y_sub = y_test[sub]
        y_sub_pred = y_test_pred[sub]

        sub_acc = accuracy_score(y_sub, y_sub_pred)
        sub_f1 = f1_score(y_sub, y_sub_pred, average="weighted")

        print(f"\n--- {sheet} ---")
        print(f"Accuracy : {sub_acc:.4f}")
        print(f"F1-score : {sub_f1:.4f}")
        print("Confusion Matrix:")
        print(confusion_matrix(y_sub, y_sub_pred))

    # -----------------------------------------------------
    # 3.7 特徵重要度
    # -----------------------------------------------------
    rf_model = model.named_steps["rf"]

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": rf_model.feature_importances_
    }).sort_values("importance", ascending=False)

    print("\n==== Top 30 特徵重要度 ====")
    print(importance_df.head(30).to_string(index=False))

    importance_df.to_csv(OUTPUT_IMPORTANCE_CSV, index=False, encoding="utf-8-sig")
    print(f"\n特徵重要度已輸出：{OUTPUT_IMPORTANCE_CSV}")

    # -----------------------------------------------------
    # 3.8 匯出預測結果
    # -----------------------------------------------------
    pred_df = test_merged[
        [TIME_COL, "group_name", "group_id", LABEL_COL] + SENSOR_COLS
    ].copy()

    pred_df["prob_normal_1"] = y_test_prob
    pred_df["y_pred"] = y_test_pred

    pred_df.to_csv(OUTPUT_PRED_CSV, index=False, encoding="utf-8-sig")
    print(f"測試集預測結果已輸出：{OUTPUT_PRED_CSV}")


if __name__ == "__main__":
    main()