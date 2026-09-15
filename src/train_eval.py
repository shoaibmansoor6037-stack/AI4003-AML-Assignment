
import pandas as pd
import numpy as np
import sys

# Allow importing regression.py
sys.path.append("src")

from regression import (
    fit_normal,
    fit_batch_gd,
    fit_sgd,
    rmse
)

# Load prepared data
comparison = pd.read_csv("data/plant1_hourly.csv")
weather = pd.read_csv("data/plant1_openmeteo.csv")

comparison["datetime"] = pd.to_datetime(comparison["datetime"])
weather["datetime"] = pd.to_datetime(weather["datetime"])

comparison = comparison.merge(
    weather,
    on="datetime",
    how="inner"
)

comparison["sw_radiation_kw"] = (
    comparison["sw_radiation"] / 1000
)

comparison["hour"] = comparison["datetime"].dt.hour

# Train/test split
train = comparison[
    (comparison["datetime"] >= "2020-05-15") &
    (comparison["datetime"] < "2020-06-11")
].copy()

test = comparison[
    (comparison["datetime"] >= "2020-06-11") &
    (comparison["datetime"] < "2020-06-18")
].copy()

# Remove missing values
train = train.dropna(
    subset=[
        "ac_power",
        "irradiation",
        "module_temp",
        "ambient_temp"
    ]
).copy()

test = test.dropna(
    subset=[
        "ac_power",
        "irradiation",
        "module_temp",
        "ambient_temp"
    ]
).copy()

# Time features
train["sin_hour"] = np.sin(
    2 * np.pi * train["hour"] / 24
)

train["cos_hour"] = np.cos(
    2 * np.pi * train["hour"] / 24
)

test["sin_hour"] = np.sin(
    2 * np.pi * test["hour"] / 24
)

test["cos_hour"] = np.cos(
    2 * np.pi * test["hour"] / 24
)

# ==========================================
# SET A
# ==========================================

features_A = [
    "irradiation",
    "module_temp",
    "ambient_temp"
]

mean_A = train[features_A].mean()
std_A = train[features_A].std()

train_A = (train[features_A] - mean_A) / std_A
test_A = (test[features_A] - mean_A) / std_A

X_train_A = np.column_stack([
    np.ones(len(train)),
    train_A.values,
    train["sin_hour"].values,
    train["cos_hour"].values
])

X_test_A = np.column_stack([
    np.ones(len(test)),
    test_A.values,
    test["sin_hour"].values,
    test["cos_hour"].values
])

# ==========================================
# SET B
# ==========================================

features_B = [
    "sw_radiation_kw",
    "temp_2m",
    "cloud_cover"
]

mean_B = train[features_B].mean()
std_B = train[features_B].std()

train_B = (train[features_B] - mean_B) / std_B
test_B = (test[features_B] - mean_B) / std_B

X_train_B = np.column_stack([
    np.ones(len(train)),
    train_B.values,
    train["sin_hour"].values,
    train["cos_hour"].values
])

X_test_B = np.column_stack([
    np.ones(len(test)),
    test_B.values,
    test["sin_hour"].values,
    test["cos_hour"].values
])

y_train = train["ac_power"].values
y_test = test["ac_power"].values

# ==========================================
# TRAIN ALL SIX MODELS
# ==========================================

# Set A
theta_normal_A = fit_normal(X_train_A, y_train)

theta_batch_A, _ = fit_batch_gd(
    X_train_A,
    y_train,
    0.001,
    500
)

theta_sgd_A, _ = fit_sgd(
    X_train_A,
    y_train,
    0.01,
    50
)

# Set B
theta_normal_B = fit_normal(X_train_B, y_train)

theta_batch_B, _ = fit_batch_gd(
    X_train_B,
    y_train,
    0.001,
    500
)

theta_sgd_B, _ = fit_sgd(
    X_train_B,
    y_train,
    0.001,
    50
)

# ==========================================
# PREDICTIONS
# ==========================================

pred_normal_A = np.maximum(
    X_test_A @ theta_normal_A, 0
)

pred_batch_A = np.maximum(
    X_test_A @ theta_batch_A, 0
)

pred_sgd_A = np.maximum(
    X_test_A @ theta_sgd_A, 0
)

pred_normal_B = np.maximum(
    X_test_B @ theta_normal_B, 0
)

pred_batch_B = np.maximum(
    X_test_B @ theta_batch_B, 0
)

pred_sgd_B = np.maximum(
    X_test_B @ theta_sgd_B, 0
)

# ==========================================
# RMSE
# ==========================================

results = pd.DataFrame({
    "Weather Set": [
        "Set A", "Set A", "Set A",
        "Set B", "Set B", "Set B"
    ],
    "Method": [
        "Normal Equation",
        "Batch GD",
        "SGD",
        "Normal Equation",
        "Batch GD",
        "SGD"
    ],
    "Alpha": [
        "-", "0.001", "0.01",
        "-", "0.001", "0.001"
    ],
    "Iterations / Epochs": [
        "-", "500", "50",
        "-", "500", "50"
    ],
    "Test RMSE (kW)": [
        rmse(y_test, pred_normal_A),
        rmse(y_test, pred_batch_A),
        rmse(y_test, pred_sgd_A),
        rmse(y_test, pred_normal_B),
        rmse(y_test, pred_batch_B),
        rmse(y_test, pred_sgd_B)
    ]
})

# Save results
results.to_csv(
    "results/table2_rmse.csv",
    index=False
)

# Save Set B model
np.savez(
    "results/setB_model.npz",
    theta=theta_normal_B,
    mean=mean_B.values,
    std=std_B.values
)

print("Training and evaluation completed!")
print()
print(results.to_string(index=False))
