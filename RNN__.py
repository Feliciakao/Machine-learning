# =========================================================
# RNN for Future 72hr Maintenance Prediction
# Clean Feature 
# =========================================================

import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# =========================================================
# 1. 基本參數設定
# =========================================================

FILE_PATH = r"C:\Users\felicia\OneDrive\Desktop\.ipynb_checkpoints\data\TH195train\whole data_TH195_K48.xlsx"

SHEET_NAMES = [
    "Group 1",
    "Group 2",
    "Group 3",
    "Group 4"
]

BASE_FEATURES = [
    "heatflow_in",
    "coldflow2",
    "heatflow_mid",
    "heatflow_out"
]

CURRENT_LABEL_COL = "label"
FUTURE_LABEL_COL = "after72hr"
TIME_COL = "time"

SEQ_LEN = 72

ROLL_WINDOWS = [6, 12, 24]
DIFF_STEPS = [1, 6, 24]

TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

BATCH_SIZE = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 200
MIN_EPOCHS = 30
PATIENCE = 30

HIDDEN_SIZE = 32
NUM_LAYERS = 1
DROPOUT = 0.3

THRESHOLD = 0.2

MANUAL_CLASS_WEIGHTS = [4.0, 1.0]

RANDOM_SEED = 42

MODEL_SAVE_PATH = "best_rnn_future72hr_feature_model.pth"
RESULT_SAVE_PATH = "rnn_future72hr_feature_result.xlsx"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", DEVICE)


# =========================================================
# 2. 固定 random seed
# =========================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(RANDOM_SEED)


# =========================================================
# 3. 欄位清理與檢查
# =========================================================

def clean_column_names(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def check_required_columns(df, sheet_name):
    required_cols = BASE_FEATURES + [
        CURRENT_LABEL_COL,
        FUTURE_LABEL_COL
    ]

    if TIME_COL in df.columns:
        required_cols = [TIME_COL] + required_cols

    missing_cols = []

    for col in required_cols:
        if col not in df.columns:
            missing_cols.append(col)

    if len(missing_cols) > 0:
        raise ValueError(
            f"\nSheet [{sheet_name}] 缺少必要欄位：{missing_cols}\n"
            f"目前欄位有：{df.columns.tolist()}\n"
            f"請確認 Excel 第一列欄位名稱是否完全一致。"
        )

# =========================================================
# 5. 建立乾淨 feature
# =========================================================

def build_clean_features(df, sheet_name):
    df = clean_column_names(df)

    if TIME_COL in df.columns:
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
        df = df.sort_values(TIME_COL).reset_index(drop=True)

    check_required_columns(df, sheet_name)

    required_base_cols = BASE_FEATURES + [
        CURRENT_LABEL_COL,
        FUTURE_LABEL_COL
    ]

    if TIME_COL in df.columns:
        data = df[[TIME_COL] + required_base_cols].copy()
    else:
        data = df[required_base_cols].copy()
        data[TIME_COL] = pd.NaT

    for col in BASE_FEATURES + [CURRENT_LABEL_COL, FUTURE_LABEL_COL]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # =====================================================
    # 1. 原始感測特徵
    # =====================================================

    feature_cols = BASE_FEATURES.copy()

    # =====================================================
    # 2. 差值特徵
    # =====================================================

    data["diff_in_out"] = data["heatflow_in"] - data["heatflow_out"]
    data["diff_mid_out"] = data["heatflow_mid"] - data["heatflow_out"]
    data["diff_in_mid"] = data["heatflow_in"] - data["heatflow_mid"]

    diff_features = [
        "diff_in_out",
        "diff_mid_out",
        "diff_in_mid"
    ]

    feature_cols += diff_features

    rolling_base_cols = BASE_FEATURES + diff_features

    # =====================================================
    # 3. rolling mean / std
    # =====================================================

    for col in rolling_base_cols:
        for win in ROLL_WINDOWS:
            mean_col = f"{col}_roll{win}_mean"
            std_col = f"{col}_roll{win}_std"

            r = data[col].rolling(
                window=win,
                min_periods=1
            )

            data[mean_col] = r.mean()
            data[std_col] = r.std().fillna(0)

            feature_cols.append(mean_col)
            feature_cols.append(std_col)



        # =====================================================
    # 5. diff lag features
    # =====================================================

    for col in rolling_base_cols:
        for step in DIFF_STEPS:
            diff_col = f"{col}_diff{step}"

            data[diff_col] = data[col].diff(step)
            data[diff_col] = data[diff_col].fillna(0)

            feature_cols.append(diff_col)

    keep_cols = [TIME_COL] + feature_cols + [
        CURRENT_LABEL_COL,
        FUTURE_LABEL_COL
    ]

    data = data[keep_cols].copy()

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(
        subset=feature_cols + [CURRENT_LABEL_COL, FUTURE_LABEL_COL]
    ).reset_index(drop=True)

    forbidden_cols = [
        TIME_COL,
        CURRENT_LABEL_COL,
        FUTURE_LABEL_COL,
        "Unnamed: 8",
        "(0=保養,1=正常)",
        "1=正常)"
    ]

    for bad_col in forbidden_cols:
        if bad_col in feature_cols:
            raise ValueError(
                f"錯誤：{bad_col} 被放進 feature_cols 了，請檢查程式。"
            )

    print(f"\n========== Clean Features: {sheet_name} ==========")
    print("Feature count:", len(feature_cols))

   
    return data, feature_cols


# =========================================================
# 6. 建立 sequence
# =========================================================

def create_sequences_from_sheet(
    df,
    sheet_name,
    feature_cols,
    current_label_col,
    future_label_col,
    seq_len
):
    use_cols = [TIME_COL] + feature_cols + [
        current_label_col,
        future_label_col
    ]

    data = df[use_cols].copy()

    for col in feature_cols + [current_label_col, future_label_col]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(
        subset=feature_cols + [current_label_col, future_label_col]
    ).reset_index(drop=True)

    X_raw = data[feature_cols].values.astype(np.float32)

    current_y = data[current_label_col].values.astype(np.int64)
    future_y = data[future_label_col].values.astype(np.int64)

    time_raw = data[TIME_COL].values

    X_seq = []
    y_seq = []
    end_indices = []
    end_times = []

    skipped_by_current_maintenance = 0
    skipped_by_invalid_label = 0

    for i in range(len(data) - seq_len + 1):
        end_index = i + seq_len - 1

        x_window = X_raw[i : i + seq_len]
        current_window = current_y[i : i + seq_len]

        # input window 裡面只要包含目前保養 label=0，就跳過
        if np.any(current_window == 0):
            skipped_by_current_maintenance += 1
            continue

        y_target = future_y[end_index]

        if y_target not in [0, 1]:
            skipped_by_invalid_label += 1
            continue

        X_seq.append(x_window)
        y_seq.append(y_target)
        end_indices.append(end_index)
        end_times.append(time_raw[end_index])

    X_seq = np.array(X_seq, dtype=np.float32)
    y_seq = np.array(y_seq, dtype=np.int64)
    end_indices = np.array(end_indices, dtype=np.int64)
    end_times = np.array(end_times)

    print(f"\n========== Sequence: {sheet_name} ==========")
    print("valid sequences:", X_seq.shape)
    print("skipped because input contains label=0:", skipped_by_current_maintenance)
    print("skipped because invalid after72hr:", skipped_by_invalid_label)

    if len(y_seq) > 0:
        unique, counts = np.unique(y_seq, return_counts=True)

        print("after72hr distribution:")
        for u, c in zip(unique, counts):
            if u == 0:
                print(f"  after72hr=0 未來72hr會保養: {c}")
            else:
                print(f"  after72hr=1 未來72hr不會保養: {c}")
    else:
        print("Warning: 此 sheet 沒有產生有效 sequence。")

    return X_seq, y_seq, end_indices, end_times


# =========================================================
# 7. 讀取 Excel
# =========================================================

excel_data = pd.read_excel(FILE_PATH, sheet_name=None)

print("\n========== Excel sheets ==========")
print(list(excel_data.keys()))

for sheet_name in SHEET_NAMES:
    if sheet_name not in excel_data:
        raise ValueError(
            f"找不到 sheet：{sheet_name}\n"
            f"目前 Excel 內有：{list(excel_data.keys())}"
        )


# =========================================================
# 8. 建立 train / val / test
# =========================================================

train_X_list = []
train_y_list = []
train_meta_list = []

val_X_list = []
val_y_list = []
val_meta_list = []

test_X_list = []
test_y_list = []
test_meta_list = []

all_feature_cols = None

for sheet_name in SHEET_NAMES:
    print(f"\nLoading sheet: {sheet_name}")

    df_sheet = excel_data[sheet_name].copy()

    df_feature, feature_cols = build_clean_features(
        df=df_sheet,
        sheet_name=sheet_name
    )

    if all_feature_cols is None:
        all_feature_cols = feature_cols
    else:
        if feature_cols != all_feature_cols:
            raise ValueError(
                f"{sheet_name} 的 feature columns 和前面 sheet 不一致。"
            )

    X_g, y_g, end_indices_g, end_times_g = create_sequences_from_sheet(
        df=df_feature,
        sheet_name=sheet_name,
        feature_cols=feature_cols,
        current_label_col=CURRENT_LABEL_COL,
        future_label_col=FUTURE_LABEL_COL,
        seq_len=SEQ_LEN
    )

    if len(X_g) == 0:
        print(f"Warning: {sheet_name} 沒有可用 sequence，略過。")
        continue

    n = len(X_g)

    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    X_train_g = X_g[:train_end]
    y_train_g = y_g[:train_end]
    train_indices_g = end_indices_g[:train_end]
    train_times_g = end_times_g[:train_end]

    X_val_g = X_g[train_end:val_end]
    y_val_g = y_g[train_end:val_end]
    val_indices_g = end_indices_g[train_end:val_end]
    val_times_g = end_times_g[train_end:val_end]

    X_test_g = X_g[val_end:]
    y_test_g = y_g[val_end:]
    test_indices_g = end_indices_g[val_end:]
    test_times_g = end_times_g[val_end:]

    train_X_list.append(X_train_g)
    train_y_list.append(y_train_g)

    val_X_list.append(X_val_g)
    val_y_list.append(y_val_g)

    test_X_list.append(X_test_g)
    test_y_list.append(y_test_g)

    train_meta_list.append(pd.DataFrame({
        "sheet": sheet_name,
        "dataset": "train",
        "source_row_index_after_dropna": train_indices_g,
        "time": train_times_g,
        "true_after72hr": y_train_g
    }))

    val_meta_list.append(pd.DataFrame({
        "sheet": sheet_name,
        "dataset": "val",
        "source_row_index_after_dropna": val_indices_g,
        "time": val_times_g,
        "true_after72hr": y_val_g
    }))

    test_meta_list.append(pd.DataFrame({
        "sheet": sheet_name,
        "dataset": "test",
        "source_row_index_after_dropna": test_indices_g,
        "time": test_times_g,
        "true_after72hr": y_test_g
    }))


if len(train_X_list) == 0:
    raise ValueError(
        "沒有任何可訓練資料。\n"
        "請檢查 sheet 名稱、欄位名稱、label、after72hr 或 SEQ_LEN。"
    )

X_train = np.concatenate(train_X_list, axis=0)
y_train = np.concatenate(train_y_list, axis=0)

X_val = np.concatenate(val_X_list, axis=0)
y_val = np.concatenate(val_y_list, axis=0)

X_test = np.concatenate(test_X_list, axis=0)
y_test = np.concatenate(test_y_list, axis=0)

train_meta = pd.concat(train_meta_list, axis=0).reset_index(drop=True)
val_meta = pd.concat(val_meta_list, axis=0).reset_index(drop=True)
test_meta = pd.concat(test_meta_list, axis=0).reset_index(drop=True)

print("\n========== Final Dataset ==========")
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val  :", X_val.shape, "y_val  :", y_val.shape)
print("X_test :", X_test.shape, "y_test :", y_test.shape)

print("\nFeature count:", len(all_feature_cols))

print("\nTrain label distribution:")
print(np.unique(y_train, return_counts=True))

print("\nVal label distribution:")
print(np.unique(y_val, return_counts=True))

print("\nTest label distribution:")
print(np.unique(y_test, return_counts=True))


# =========================================================
# 9. StandardScaler
# =========================================================

num_features = X_train.shape[2]

scaler = StandardScaler()

X_train_2d = X_train.reshape(-1, num_features)
X_val_2d = X_val.reshape(-1, num_features)
X_test_2d = X_test.reshape(-1, num_features)

scaler.fit(X_train_2d)

X_train_scaled = scaler.transform(X_train_2d).reshape(X_train.shape)
X_val_scaled = scaler.transform(X_val_2d).reshape(X_val.shape)
X_test_scaled = scaler.transform(X_test_2d).reshape(X_test.shape)


# =========================================================
# 10. Dataset / DataLoader
# =========================================================

class MaintenanceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = MaintenanceDataset(X_train_scaled, y_train)
val_dataset = MaintenanceDataset(X_val_scaled, y_val)
test_dataset = MaintenanceDataset(X_test_scaled, y_test)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

train_eval_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# =========================================================
# 11. RNN Model
# =========================================================

class RNNMaintenanceModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(RNNMaintenanceModel, self).__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity="tanh",
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        out, h_n = self.rnn(x)

        # 使用最後一個時間點的 hidden output
        last_out = out[:, -1, :]

        logits = self.fc(last_out)

        return logits


model = RNNMaintenanceModel(
    input_size=num_features,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
).to(DEVICE)

print("\n========== Model ==========")
print(model)


# =========================================================
# 12. Loss / Optimizer
# =========================================================

class_counts = np.bincount(y_train, minlength=2)

print("\nTrain class counts:", class_counts)

class_weights_np = np.array(MANUAL_CLASS_WEIGHTS, dtype=np.float32)
class_weights = torch.tensor(class_weights_np, dtype=torch.float32).to(DEVICE)

print("Manual Class weights:", class_weights)

criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# =========================================================
# 13. Train / Evaluate function
# =========================================================

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)

    avg_loss = total_loss / len(loader.dataset)

    return avg_loss


def get_loss(model, loader, criterion):
    model.eval()

    total_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            total_loss += loss.item() * X_batch.size(0)

    avg_loss = total_loss / len(loader.dataset)

    return avg_loss


def predict_with_prob(model, loader):
    model.eval()

    all_y_true = []
    all_maint_prob = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)

            logits = model(X_batch)
            probs = torch.softmax(logits, dim=1)

            # P(after72hr = 0)
            maintenance_prob = probs[:, 0]

            all_y_true.extend(y_batch.cpu().numpy())
            all_maint_prob.extend(maintenance_prob.cpu().numpy())

    all_y_true = np.array(all_y_true)
    all_maint_prob = np.array(all_maint_prob)

    return all_y_true, all_maint_prob


def evaluate_by_threshold(y_true, maintenance_prob, threshold):
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


def print_time_ranges(meta_dict):
    print("\n========== Time Range Summary ==========")

    for dataset_name, meta_df in meta_dict.items():
        print(f"\n[{dataset_name.upper()}]")
        for sheet_name in SHEET_NAMES:
            temp = meta_df[meta_df["sheet"] == sheet_name].copy()

            if len(temp) == 0:
                print(f"{sheet_name}: no data")
                continue

            start_time = temp["time"].min()
            end_time = temp["time"].max()

            print(f"{sheet_name}: {start_time} ~ {end_time} | count={len(temp)}")


# =========================================================
# 14. Training with Early Stopping
# =========================================================

best_val_loss = float("inf")
best_epoch = 0
patience_counter = 0

print("\n========== Training ==========")

for epoch in range(1, MAX_EPOCHS + 1):
    train_loss = train_one_epoch(
        model=model,
        loader=train_loader,
        criterion=criterion,
        optimizer=optimizer
    )

    val_loss = get_loss(
        model=model,
        loader=val_loader,
        criterion=criterion
    )

    print(
        f"Epoch [{epoch:03d}/{MAX_EPOCHS}] "
        f"Train Loss: {train_loss:.5f} | "
        f"Val Loss: {val_loss:.5f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        patience_counter = 0

        torch.save({
            "model_state_dict": model.state_dict(),
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "seq_len": SEQ_LEN,
            "sheet_names": SHEET_NAMES,
            "base_features": BASE_FEATURES,
            "feature_cols": all_feature_cols,
            "current_label_col": CURRENT_LABEL_COL,
            "future_label_col": FUTURE_LABEL_COL,
            "threshold": THRESHOLD,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "manual_class_weights": MANUAL_CLASS_WEIGHTS,
            "model_type": "RNN"
        }, MODEL_SAVE_PATH)

        print(f"Save best model at epoch {epoch}, val_loss={val_loss:.5f}")

    else:
        patience_counter += 1

        if epoch >= MIN_EPOCHS and patience_counter >= PATIENCE:
            print(
            f"\nEarly stopping at epoch {epoch}. "
            f"Best epoch = {best_epoch}, "
            f"Best val loss = {best_val_loss:.5f}"
            )
            break

        if epoch >= MIN_EPOCHS and patience_counter >= PATIENCE:
            print(
            f"\nEarly stopping at epoch {epoch}. "
            f"Best epoch = {best_epoch}, "
            f"Best val loss = {best_val_loss:.5f}"
        )
        break

# =========================================================
# 15. Load Best Model
# =========================================================

checkpoint = torch.load(
    MODEL_SAVE_PATH,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"])

print("\n========== Best Model Loaded ==========")
print("Best epoch:", best_epoch)
print("Best val loss:", best_val_loss)


# =========================================================
# 16. Predict train / val / test
# =========================================================

train_y_true, train_maint_prob = predict_with_prob(
    model=model,
    loader=train_eval_loader
)

val_y_true, val_maint_prob = predict_with_prob(
    model=model,
    loader=val_loader
)

test_y_true, test_maint_prob = predict_with_prob(
    model=model,
    loader=test_loader
)


# =========================================================
# =========================================================
# 17. Threshold sweep on VAL and find best VAL F1
# =========================================================

threshold_candidates = np.arange(0.05, 0.81, 0.01)

print("\n========== Threshold Sweep on VAL ==========")

val_threshold_results = []
best_val_threshold = None
best_val_f1 = -1.0
best_val_result = None

for th in threshold_candidates:
    r = evaluate_by_threshold(
        y_true=val_y_true,
        maintenance_prob=val_maint_prob,
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

print("\n========== Best Threshold from VAL ==========")
print("Best VAL threshold:", best_val_threshold)
print("Best VAL F1(after72hr=0):", best_val_f1)
print("Best VAL Accuracy:", best_val_result["accuracy"])
print("Best VAL Precision(after72hr=0):", best_val_result["precision_maintenance"])
print("Best VAL Recall(after72hr=0):", best_val_result["recall_maintenance"])


# =========================================================
# 18. Final Evaluation on TEST with best TEST threshold
# =========================================================

# =========================================================
# 18. Final Evaluation on TEST with best VAL threshold
# =========================================================

test_result_with_best_val_th = evaluate_by_threshold(
    y_true=test_y_true,
    maintenance_prob=test_maint_prob,
    threshold=best_val_threshold
)

test_y_pred = test_result_with_best_val_th["y_pred"]

print("\n============================================================")
print(f"Classification Report on TEST with BEST VAL THRESHOLD = {best_val_threshold}")
print("============================================================")

print(
    classification_report(
        test_y_true,
        test_y_pred,
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
    test_y_true,
    test_y_pred,
    labels=[0, 1]
)

print(cm)

print("\nInterpretation:")
print(f"True 0, Pred 0：{cm[0, 0]}  正確抓到未來會保養")
print(f"True 0, Pred 1：{cm[0, 1]}  漏掉未來會保養")
print(f"True 1, Pred 0：{cm[1, 0]}  誤報保養")
print(f"True 1, Pred 1：{cm[1, 1]}  正確判斷正常")

print("\n========== Final TEST Metrics ==========")
print(f"Using best VAL threshold: {best_val_threshold}")
print(f"TEST Accuracy: {test_result_with_best_val_th['accuracy']:.4f}")
print(f"TEST Precision(after72hr=0): {test_result_with_best_val_th['precision_maintenance']:.4f}")
print(f"TEST Recall(after72hr=0): {test_result_with_best_val_th['recall_maintenance']:.4f}")
print(f"TEST F1(after72hr=0): {test_result_with_best_val_th['f1_maintenance']:.4f}")

print("\n============================================================")
print(f"Classification Report on TEST with BEST VAL THRESHOLD = {best_val_threshold}")
print("============================================================")

print(
    classification_report(
        test_y_true,
        test_y_pred,
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
cm = confusion_matrix(test_y_true, test_y_pred)
print(cm)

print("\nInterpretation:")
print(f"True 0, Pred 0：{cm[0, 0]}  正確抓到未來會保養")
print(f"True 0, Pred 1：{cm[0, 1]}  漏掉未來會保養")
print(f"True 1, Pred 0：{cm[1, 0]}  誤報保養")
print(f"True 1, Pred 1：{cm[1, 1]}  正確判斷正常")


# =========================================================
# 19. 使用 best TEST threshold 產生 train / val / test 預測結果
# =========================================================

train_pred_final = np.where(train_maint_prob >= best_val_threshold, 0, 1)
val_pred_final = np.where(val_maint_prob >= best_val_threshold, 0, 1)
test_pred_final = np.where(test_maint_prob >= best_val_threshold, 0, 1)

train_meta["pred_after72hr"] = train_pred_final
train_meta["maintenance_prob_P_after72hr_0"] = train_maint_prob

val_meta["pred_after72hr"] = val_pred_final
val_meta["maintenance_prob_P_after72hr_0"] = val_maint_prob

test_meta["pred_after72hr"] = test_pred_final
test_meta["maintenance_prob_P_after72hr_0"] = test_maint_prob

all_result_df = pd.concat(
    [train_meta, val_meta, test_meta],
    axis=0
).reset_index(drop=True)

all_result_df = all_result_df.sort_values(
    ["sheet", "time", "dataset"]
).reset_index(drop=True)


# =========================================================
# 20. 印出 train / val / test 時間區間
# =========================================================

print_time_ranges({
    "train": train_meta,
    "val": val_meta,
    "test": test_meta
})


# =========================================================
# 21. 輸出 Excel：同一個 Sheet，依照 time 橫向整理 Group 1~4
# =========================================================

summary_df = None

for sheet_name in SHEET_NAMES:
    temp = all_result_df[all_result_df["sheet"] == sheet_name].copy()

    if len(temp) == 0:
        print(f"Warning: {sheet_name} 沒有可輸出的結果")
        continue

    temp = temp.sort_values("time").reset_index(drop=True)

    temp = temp[[
        "time",
        "true_after72hr",
        "pred_after72hr"
    ]].copy()

    temp = temp.rename(columns={
        "true_after72hr": f"{sheet_name} after72hr",
        "pred_after72hr": f"{sheet_name} model_result"
    })

    if summary_df is None:
        summary_df = temp
    else:
        summary_df = pd.merge(
            summary_df,
            temp,
            on="time",
            how="outer"
        )

summary_df = summary_df.sort_values("time").reset_index(drop=True)

ordered_cols = ["time"]

for sheet_name in SHEET_NAMES:
    ordered_cols.append(f"{sheet_name} after72hr")
    ordered_cols.append(f"{sheet_name} model_result")

summary_df = summary_df[ordered_cols]

# 空格 / NaN 補 0
for col in summary_df.columns:
    if col != "time":
        summary_df[col] = summary_df[col].fillna(0)

# after72hr / model_result 轉成整數
for col in summary_df.columns:
    if col != "time":
        summary_df[col] = summary_df[col].astype(int)

threshold_summary_df = pd.DataFrame(val_threshold_results)

with pd.ExcelWriter(RESULT_SAVE_PATH, engine="openpyxl") as writer:
    summary_df.to_excel(
        writer,
        sheet_name="All_Group_Result",
        index=False
    )

    threshold_summary_df.to_excel(
        writer,
        sheet_name="Threshold_Summary",
        index=False
    )

print(f"\n已輸出完整結果：{RESULT_SAVE_PATH}")
print(f"已儲存最佳模型：{MODEL_SAVE_PATH}")
print(f"使用的最佳 VAL threshold：{best_val_threshold}")
print(f"最佳 VAL F1(after72hr=0)：{best_val_f1}")
print(f"TEST F1(after72hr=0)：{test_result_with_best_val_th['f1_maintenance']}")