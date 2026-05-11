import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# =========================================================
# 1. 基本設定
# =========================================================

FILE_PATH = r"C:\Users\felicia\OneDrive\Desktop\.ipynb_checkpoints\data\TH195train\whole data_TH195_K48.xlsx"

GROUP_SHEETS = {
    "Group 1": "Group 1",
    "Group 2": "Group 2",
    "Group 3": "Group 3",
    "Group 4": "Group 4",
}

TIME_COL = "time"

# 目前狀態 label
# label = 0 代表目前正在保養
# label = 1 代表目前正常
CURRENT_LABEL_COL = "label"

# 未來 72 小時 label
# after72hr = 0 代表未來 72 小時會保養
# after72hr = 1 代表未來 72 小時不會保養
FUTURE_LABEL_COL = "after72hr"

INPUT_COLS = [
    "heatflow_in",
    "coldflow2",
    "heatflow_mid",
    "heatflow_out",
]

TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

ROLL_WINDOWS = [6, 12, 24]
DIFF_STEPS = [1, 6, 24]
SLOPE_WINDOWS = [6, 12, 24]

RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 12,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "random_state": 42,
    "n_jobs": -1,
    "class_weight" : "balanced"
} 

OUTPUT_IMPORTANCE_CSV = "rf_feature_importance.csv"
OUTPUT_ALL_PRED_CSV = "rf_all_predictions.csv"
OUTPUT_EXCEL = "rf_result_summary.xlsx"


# =========================================================
# 2. Feature functions
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


def create_time_features(df):
    """
    建立時間特徵：
    hour, dayofweek, month

    這 3 個 feature 加入後：
    91 features + 3 time features = 94 features
    """
    out = pd.DataFrame(index=df.index)

    out["hour"] = df[TIME_COL].dt.hour
    out["dayofweek"] = df[TIME_COL].dt.dayofweek
    out["month"] = df[TIME_COL].dt.month

    return out


def create_base_diff_features(df):
    """
    建立 3 個溫差 feature。
    """
    out = pd.DataFrame(index=df.index)

    out["diff_in_out"] = df["heatflow_in"] - df["heatflow_out"]
    out["diff_mid_out"] = df["heatflow_mid"] - df["heatflow_out"]
    out["diff_in_mid"] = df["heatflow_in"] - df["heatflow_mid"]

    return out


def create_ts_features(df, base_cols):
    """
    對 7 個欄位建立：
    1. diff lag
    2. rolling mean/std
    3. rolling slope

    若 base_cols 有 7 個欄位：

    diff lag:
        7 * 3 = 21

    rolling mean/std:
        7 * 3 windows * 2 = 42

    slope:
        7 * 3 = 21

    總共：
        21 + 42 + 21 = 84
    """

    feat_dict = {}

    for col in base_cols:
        s = df[col]

        # =================================================
        # diff lag features
        # =================================================
        for step in DIFF_STEPS:
            diff_col = f"{col}_diff{step}"
            feat_dict[diff_col] = s.diff(step).fillna(0)

        # =================================================
        # rolling mean / std
        # =================================================
        for win in ROLL_WINDOWS:
            r = s.rolling(window=win, min_periods=1)

            mean_col = f"{col}_roll{win}_mean"
            std_col = f"{col}_roll{win}_std"

            feat_dict[mean_col] = r.mean()
            feat_dict[std_col] = r.std().fillna(0)

        # =================================================
        # rolling slope features
        # =================================================
        for win in SLOPE_WINDOWS:
            slope_col = f"{col}_slope{win}"

            feat_dict[slope_col] = (
                s.rolling(window=win, min_periods=win)
                .apply(rolling_slope, raw=True)
                .fillna(0)
            )

    return pd.DataFrame(feat_dict, index=df.index)


# =========================================================
# 3. 切 train / val / test
# =========================================================

def chronological_split_60_20_20(df):
    n = len(df)

    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    train_df["dataset"] = "train"
    val_df["dataset"] = "val"
    test_df["dataset"] = "test"

    return train_df, val_df, test_df


# =========================================================
# 4. 讀取單一 group
# =========================================================

def load_one_group(group_name, sheet_name):
    df = pd.read_excel(FILE_PATH, sheet_name=sheet_name)

    df.columns = df.columns.astype(str).str.strip()

    required_cols = [
        TIME_COL
    ] + INPUT_COLS + [
        CURRENT_LABEL_COL,
        FUTURE_LABEL_COL
    ]

    missing_cols = [c for c in required_cols if c not in df.columns]

    if len(missing_cols) > 0:
        raise ValueError(
            f"{sheet_name} 缺少欄位: {missing_cols}\n"
            f"目前欄位: {df.columns.tolist()}"
        )

    df = df[required_cols].copy()

    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df[CURRENT_LABEL_COL] = pd.to_numeric(df[CURRENT_LABEL_COL], errors="coerce")
    df[FUTURE_LABEL_COL] = pd.to_numeric(df[FUTURE_LABEL_COL], errors="coerce")

    for col in INPUT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=[TIME_COL, CURRENT_LABEL_COL, FUTURE_LABEL_COL]
    ).copy()

    df = df.sort_values(TIME_COL).reset_index(drop=True)

    before_count = len(df)

    # =====================================================
    # 移除目前正在保養的資料
    # label = 0 代表目前正在保養
    # 這裡要刪掉，避免模型學到「保養期間」而不是「保養前兆」
    # =====================================================
    df = df[df[CURRENT_LABEL_COL] != 0].copy()

    after_count = len(df)

    print(
        f"{sheet_name}: remove current maintenance rows: "
        f"{before_count - after_count}"
    )

    df = df.reset_index(drop=True)

    df["group"] = group_name

    return df


# =========================================================
# 5. 建立 94 features
# =========================================================

def build_features_for_group(df):
    """
    目標 feature 數量 = 94

    組成：
    原始值 4 個：
        heatflow_in
        coldflow2
        heatflow_mid
        heatflow_out

    溫差 3 個：
        diff_in_out
        diff_mid_out
        diff_in_mid

    時間特徵 3 個：
        hour
        dayofweek
        month

    rolling mean/std 42 個：
        7 個欄位 * 3 個 window * 2 種統計量

    diff lag 21 個：
        7 個欄位 * 3 個 step

    slope 21 個：
        7 個欄位 * 3 個 window

    總共：
        4 + 3 + 3 + 42 + 21 + 21 = 94
    """

    base_diff_df = create_base_diff_features(df)

    rolling_base_df = pd.concat(
        [
            df[INPUT_COLS],
            base_diff_df
        ],
        axis=1
    )

    rolling_base_cols = rolling_base_df.columns.tolist()

    ts_feat_df = create_ts_features(
        df=rolling_base_df,
        base_cols=rolling_base_cols
    )

    time_feat_df = create_time_features(df)

    base_df = df[
        [
            TIME_COL,
            "group",
            CURRENT_LABEL_COL,
            FUTURE_LABEL_COL
        ] + INPUT_COLS
    ].copy()

    df_feat = pd.concat(
        [
            base_df,
            base_diff_df,
            time_feat_df,
            ts_feat_df
        ],
        axis=1
    ).copy()

    df_feat = df_feat.replace([np.inf, -np.inf], np.nan)

    return df_feat


# =========================================================
# 6. Evaluation
# =========================================================

def evaluate_by_threshold(y_true, maintenance_prob, threshold):
    """
    maintenance_prob = P(after72hr = 0)

    如果 P(保養=0) >= threshold：
        預測為 0，也就是未來72hr會保養

    否則：
        預測為 1，也就是未來72hr正常
    """

    y_pred = np.where(maintenance_prob >= threshold, 0, 1)

    acc = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        pos_label=0,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        pos_label=0,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        pos_label=0,
        zero_division=0
    )

    return {
        "threshold": threshold,
        "accuracy": acc,
        "precision_maintenance": precision,
        "recall_maintenance": recall,
        "f1_maintenance": f1,
        "y_pred": y_pred
    }


def get_maintenance_prob(model, X, class_list):
    """
    回傳 P(after72hr = 0)

    sklearn 的 predict_proba 欄位順序由 model.classes_ 決定。
    所以這裡明確找 class 0 的 index。

    class 0 = future_maintenance_0
    """

    if 0 not in class_list:
        raise ValueError(
            "訓練資料中沒有 class 0，無法取得 P(after72hr=0 / 保養)。"
        )

    maintenance_index = class_list.index(0)

    proba_all = model.predict_proba(X)

    maintenance_prob = proba_all[:, maintenance_index]

    return maintenance_prob


# =========================================================
# 7. Main
# =========================================================

def main():
    group_feature_dfs = []

    print("==== 讀取並建立 Group 1~4 特徵 ====")

    for group_name, sheet_name in GROUP_SHEETS.items():
        raw_df = load_one_group(group_name, sheet_name)

        feat_df = build_features_for_group(raw_df)

        train_df, val_df, test_df = chronological_split_60_20_20(feat_df)

        group_feature_dfs.append(train_df)
        group_feature_dfs.append(val_df)
        group_feature_dfs.append(test_df)

        print(f"\n{group_name}")
        print(
            f"total={len(feat_df)}, "
            f"train={len(train_df)}, "
            f"val={len(val_df)}, "
            f"test={len(test_df)}"
        )
        print("after72hr 分布:")
        print(feat_df[FUTURE_LABEL_COL].value_counts(dropna=False).sort_index())

    all_df = pd.concat(group_feature_dfs, axis=0).reset_index(drop=True)

    forbidden_cols = [
        TIME_COL,
        "group",
        CURRENT_LABEL_COL,
        FUTURE_LABEL_COL,
        "dataset"
    ]

    feature_cols = [
        c for c in all_df.columns
        if c not in forbidden_cols
    ]

    print("\n==== 合併完成 ====")
    print(f"總資料筆數: {len(all_df)}")
    print(f"特徵數量: {len(feature_cols)}")

    print("\nFeature columns:")
    for i, col in enumerate(feature_cols, start=1):
        print(f"{i:02d}. {col}")

    if len(feature_cols) != 94:
        print(
            "\nWarning: 目前 feature 數量不是 94，"
            "請檢查 time/slope/diff/rolling feature 是否正確。"
        )

    print("\n總 after72hr label 分布:")
    print(all_df[FUTURE_LABEL_COL].value_counts(dropna=False).sort_index())

    train_all = all_df[all_df["dataset"] == "train"].copy()
    val_all = all_df[all_df["dataset"] == "val"].copy()
    test_all = all_df[all_df["dataset"] == "test"].copy()

    X_train = train_all[feature_cols]
    y_train = train_all[FUTURE_LABEL_COL].astype(int)

    X_val = val_all[feature_cols]
    y_val = val_all[FUTURE_LABEL_COL].astype(int)

    X_test = test_all[feature_cols]
    y_test = test_all[FUTURE_LABEL_COL].astype(int)

    # =====================================================
    # Train model
    # =====================================================

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(**RF_PARAMS))
    ])

    print("\n==== 開始訓練 Random Forest ====")
    model.fit(X_train, y_train)

    rf_model = model.named_steps["rf"]

    print("\nRF class order:")
    print(rf_model.classes_)
    print("說明：classes_[i] 對應 predict_proba[:, i]")

    class_list = list(rf_model.classes_)

    if 0 not in class_list:
        raise ValueError(
            "訓練資料中沒有 class 0，無法取得 P(after72hr=0 / 保養)。"
        )

    print("\n確認：本程式使用 P(after72hr=0) 作為保養機率。")
    print("也就是 probability 越高，越偏向判斷未來72小時會保養。")

    # =====================================================
    # Predict probability: P(after72hr = 0)
    # =====================================================

    train_prob_maintenance = get_maintenance_prob(
        model=model,
        X=X_train,
        class_list=class_list
    )

    val_prob_maintenance = get_maintenance_prob(
        model=model,
        X=X_val,
        class_list=class_list
    )

    test_prob_maintenance = get_maintenance_prob(
        model=model,
        X=X_test,
        class_list=class_list
    )

    # =====================================================
    # Threshold sweep on VAL
    # =====================================================

    threshold_candidates = np.arange(0.05, 0.81, 0.01)

    print("\n==== Threshold Sweep on VAL：使用 P(after72hr=0 / 保養) ====")

    val_threshold_results = []
    best_val_threshold = None
    best_val_f1 = -1.0
    best_val_result = None

    for th in threshold_candidates:
        r = evaluate_by_threshold(
            y_true=y_val,
            maintenance_prob=val_prob_maintenance,
            threshold=th
        )

        val_threshold_results.append({
            "threshold": th,
            "accuracy": r["accuracy"],
            "precision_after72hr_0": r["precision_maintenance"],
            "recall_after72hr_0": r["recall_maintenance"],
            "f1_after72hr_0": r["f1_maintenance"]
        })

        print(
            f"TH={th:.2f} | "
            f"Acc={r['accuracy']:.4f} | "
            f"Precision(after72hr=0)={r['precision_maintenance']:.4f} | "
            f"Recall(after72hr=0)={r['recall_maintenance']:.4f} | "
            f"F1(after72hr=0)={r['f1_maintenance']:.4f}"
        )

        if r["f1_maintenance"] > best_val_f1:
            best_val_f1 = r["f1_maintenance"]
            best_val_threshold = th
            best_val_result = r

    print("\n==== Best Threshold from VAL ====")
    print("Best VAL threshold:", best_val_threshold)
    print("Best VAL F1(after72hr=0):", best_val_f1)
    print("Best VAL Accuracy:", best_val_result["accuracy"])
    print("Best VAL Precision(after72hr=0):", best_val_result["precision_maintenance"])
    print("Best VAL Recall(after72hr=0):", best_val_result["recall_maintenance"])

    # =====================================================
    # Evaluate TEST with best VAL threshold
    # =====================================================

    test_result = evaluate_by_threshold(
        y_true=y_test,
        maintenance_prob=test_prob_maintenance,
        threshold=best_val_threshold
    )

    test_pred = test_result["y_pred"]

    print("\n============================================================")
    print(f"Classification Report on TEST with BEST VAL THRESHOLD = {best_val_threshold}")
    print("============================================================")

    print(
        classification_report(
            y_test,
            test_pred,
            labels=[0, 1],
            target_names=[
                "future_maintenance_0",
                "future_normal_1"
            ],
            zero_division=0
        )
    )

    print("\nConfusion Matrix on TEST:")
    print("Rows = true label, Columns = predicted label")
    print("labels: 0 = future maintenance, 1 = future normal")

    cm = confusion_matrix(
        y_test,
        test_pred,
        labels=[0, 1]
    )

    print(cm)

    print("\nInterpretation:")
    print(f"True 0, Pred 0：{cm[0, 0]}  正確抓到未來會保養")
    print(f"True 0, Pred 1：{cm[0, 1]}  漏掉未來會保養")
    print(f"True 1, Pred 0：{cm[1, 0]}  誤報保養")
    print(f"True 1, Pred 1：{cm[1, 1]}  正確判斷正常")

    print("\n==== Final TEST Metrics ====")
    print(f"Using best VAL threshold: {best_val_threshold}")
    print(f"TEST Accuracy: {test_result['accuracy']:.4f}")
    print(f"TEST Precision(after72hr=0): {test_result['precision_maintenance']:.4f}")
    print(f"TEST Recall(after72hr=0): {test_result['recall_maintenance']:.4f}")
    print(f"TEST F1(after72hr=0): {test_result['f1_maintenance']:.4f}")

    # =====================================================
    # Apply best VAL threshold to train / val / test
    # =====================================================

    train_pred = np.where(train_prob_maintenance >= best_val_threshold, 0, 1)
    val_pred = np.where(val_prob_maintenance >= best_val_threshold, 0, 1)
    test_pred = np.where(test_prob_maintenance >= best_val_threshold, 0, 1)

    train_all["pred_after72hr"] = train_pred
    train_all["maintenance_prob_P_after72hr_0"] = train_prob_maintenance

    val_all["pred_after72hr"] = val_pred
    val_all["maintenance_prob_P_after72hr_0"] = val_prob_maintenance

    test_all["pred_after72hr"] = test_pred
    test_all["maintenance_prob_P_after72hr_0"] = test_prob_maintenance

    result_all = pd.concat(
        [train_all, val_all, test_all],
        axis=0
    ).reset_index(drop=True)

    result_all = result_all.sort_values(
        ["group", TIME_COL, "dataset"]
    ).reset_index(drop=True)

    # =====================================================
    # Feature importance
    # =====================================================

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": rf_model.feature_importances_
    }).sort_values("importance", ascending=False)

    print("\n==== Top 30 Feature Importance ====")
    print(importance_df.head(30).to_string(index=False))

    importance_df.to_csv(
        OUTPUT_IMPORTANCE_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    # =====================================================
    # Export all predictions
    # =====================================================

    export_cols = [
        TIME_COL,
        "group",
        "dataset",
        CURRENT_LABEL_COL,
        FUTURE_LABEL_COL,
        "pred_after72hr",
        "maintenance_prob_P_after72hr_0",
    ] + INPUT_COLS

    result_all[export_cols].to_csv(
        OUTPUT_ALL_PRED_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    # =====================================================
    # RNN/LSTM-like Excel output
    # Sheet: All_Group_Result
    # 欄位格式：
    # time
    # Group 1 after72hr
    # Group 1 model_result
    # Group 2 after72hr
    # Group 2 model_result
    # ...
    # =====================================================

    summary_df = None

    for group_name in GROUP_SHEETS.keys():
        gdf = result_all[result_all["group"] == group_name].copy()

        if len(gdf) == 0:
            print(f"Warning: {group_name} 沒有可輸出的資料")
            continue

        temp = gdf[[
            TIME_COL,
            FUTURE_LABEL_COL,
            "pred_after72hr"
        ]].copy()

        temp = temp.sort_values(TIME_COL).reset_index(drop=True)

        temp = temp.rename(columns={
            FUTURE_LABEL_COL: f"{group_name} after72hr",
            "pred_after72hr": f"{group_name} model_result"
        })

        if summary_df is None:
            summary_df = temp
        else:
            summary_df = pd.merge(
                summary_df,
                temp,
                on=TIME_COL,
                how="outer"
            )

    summary_df = summary_df.sort_values(TIME_COL).reset_index(drop=True)

    ordered_cols = [TIME_COL]

    for group_name in GROUP_SHEETS.keys():
        ordered_cols.append(f"{group_name} after72hr")
        ordered_cols.append(f"{group_name} model_result")

    summary_df = summary_df[ordered_cols]

    # 空格 / NaN 補 0
    for col in summary_df.columns:
        if col != TIME_COL:
            summary_df[col] = summary_df[col].fillna(0)

    # after72hr / model_result 轉成整數
    for col in summary_df.columns:
        if col != TIME_COL:
            summary_df[col] = summary_df[col].astype(int)

    # =====================================================
    # VAL threshold summary
    # =====================================================

    val_threshold_summary_df = pd.DataFrame(val_threshold_results)

    # =====================================================
    # TEST final summary
    # =====================================================

    test_final_summary_df = pd.DataFrame([{
        "best_threshold_source": "VAL",
        "best_val_threshold": best_val_threshold,
        "best_val_f1_after72hr_0": best_val_f1,
        "test_accuracy": test_result["accuracy"],
        "test_precision_after72hr_0": test_result["precision_maintenance"],
        "test_recall_after72hr_0": test_result["recall_maintenance"],
        "test_f1_after72hr_0": test_result["f1_maintenance"],
        "probability_used": "P(after72hr=0)",
        "meaning": "maintenance probability",
        "feature_count": len(feature_cols),
        "feature_note": "raw + diff + time features + rolling mean/std + diff lag + slope"
    }])

    # =====================================================
    # 輸出 Excel
    # =====================================================

    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        summary_df.to_excel(
            writer,
            sheet_name="All_Group_Result",
            index=False
        )

        val_threshold_summary_df.to_excel(
            writer,
            sheet_name="VAL_Threshold_Summary",
            index=False
        )

        test_final_summary_df.to_excel(
            writer,
            sheet_name="TEST_Final_Summary",
            index=False
        )

        importance_df.to_excel(
            writer,
            sheet_name="Feature_Importance",
            index=False
        )

        result_all[export_cols].to_excel(
            writer,
            sheet_name="All_Predictions",
            index=False
        )

    print("\n==== 輸出完成 ====")
    print(f"特徵重要度 CSV: {OUTPUT_IMPORTANCE_CSV}")
    print(f"全部預測結果 CSV: {OUTPUT_ALL_PRED_CSV}")
    print(f"結果 Excel: {OUTPUT_EXCEL}")
    print(f"使用的最佳 VAL threshold：{best_val_threshold}")
    print(f"最佳 VAL F1(after72hr=0)：{best_val_f1}")
    print(f"TEST F1(after72hr=0)：{test_result['f1_maintenance']}")
    print("機率欄位 maintenance_prob_P_after72hr_0 = P(after72hr=0)，也就是保養機率。")


if __name__ == "__main__":
    main()