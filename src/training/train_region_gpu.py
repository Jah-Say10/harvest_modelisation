# ============================================================
# python -m training.train_region_gpu
#
# GLOBAL REGIONAL MALARIA FORECASTING — RUNPOD A40 (48GB VRAM) EDITION
#
# ------------------------------------------------------------
# ONE-TIME ENVIRONMENT SETUP ON THE POD (run in the pod terminal,
# NOT inside this script). The pod image is "Runpod PyTorch 2.8.0"
# which ships NVIDIA drivers + CUDA 12.8 runtime, but this script
# uses TensorFlow + XGBoost/LightGBM/CatBoost, not PyTorch, so you
# still need to install a CUDA-enabled TF build yourself:
#
#   pip install -U "tensorflow[and-cuda]"          # TF w/ CUDA 12 wheels
#   pip install -U "xgboost>=2.0"                  # has device="cuda" support
#   pip install -U lightgbm                         # CPU wheel by default;
#       # GPU/CUDA wheel needs a source build:
#       # pip install lightgbm --config-settings=cmake.define.USE_CUDA=ON
#       # if that fails, the script auto-falls-back to CPU (device_type="cpu")
#   pip install -U catboost                         # GPU support works out of the box
#   pip install -U statsmodels scikit-learn prophet joblib openpyxl
#
# Verify GPU is visible before training:
#   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
#   nvidia-smi
# ------------------------------------------------------------
#
# MODELS:
#
# 1. MACHINE LEARNING
#    - Random Forest      (CPU, sklearn — parallelized across 9 vCPUs)
#    - XGBoost             (GPU via device="cuda" when available)
#    - LightGBM             (GPU via device_type="cuda"/"gpu" when available, else CPU)
#    - CatBoost              (GPU via task_type="GPU" when available)   <- added, GPU-native
#    - SVR                  (CPU, sklearn)
#
# 2. TIME SERIES (CPU-only libraries — statsmodels/Prophet have no GPU path;
#    parallelized across regions using all 9 vCPUs instead)
#    - ARIMA
#    - SARIMA
#    - SARIMAX
#    - Holt-Winters / ETS
#    - VAR
#    - Prophet
#
# 3. DEEP LEARNING (GPU via TensorFlow, mixed precision on the A40's Tensor Cores)
#    - RNN
#    - LSTM
#    - GRU
#
# ============================================================

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless pod — no display server
import matplotlib.pyplot as plt
import seaborn as sns

import os
import time
from pathlib import Path
from datetime import datetime
from itertools import product

import pickle
import json
import warnings

from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

# ============================================================
# HARDWARE / DEVICE CONFIGURATION (RunPod: 1x A40, 48GB VRAM,
# 9 vCPU, 50GB RAM)
# ============================================================

# Reasonable TF allocator behavior on shared GPU pods
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # quiet TF INFO/WARNING spam
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_cuda_data_dir=/usr/local/cuda")

import tensorflow as tf

# All CPU-bound work (sklearn GridSearch/RandomizedSearch, statsmodels
# per-region loops) is parallelized across the pod's 9 vCPUs.
N_JOBS = max(1, (os.cpu_count() or 9) - 1)  # leave 1 core free for the OS/driver

GPU_AVAILABLE = False
gpus = tf.config.list_physical_devices("GPU")

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        # A40 has full Tensor Core support -> mixed precision gives a real
        # throughput boost for the RNN/LSTM/GRU training below.
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

        GPU_AVAILABLE = True

        print(f"[GPU] {len(gpus)} GPU(s) detected: {[g.name for g in gpus]}")
        print("[GPU] Memory growth enabled, mixed_float16 policy active")

    except RuntimeError as e:
        print(f"[GPU] Could not configure GPU, falling back to CPU: {e}")

else:
    print("[GPU] No GPU detected by TensorFlow — training will run on CPU. "
          "If you're on the RunPod A40 pod, check `nvidia-smi` and make sure "
          "tensorflow[and-cuda] is installed inside this environment.")

print(f"[CPU] Using N_JOBS={N_JOBS} for parallel CPU work "
      f"(sklearn search, per-region time-series fits)")

# XGBoost / LightGBM / CatBoost device strings, resolved once here so every
# model spec below just references these constants.
XGB_DEVICE = "cuda" if GPU_AVAILABLE else "cpu"
XGB_TREE_METHOD = "hist"  # required for device="cuda" on xgboost>=2.0

LGBM_DEVICE = "cpu"
if GPU_AVAILABLE:
    # LightGBM only actually uses the GPU if it was compiled with GPU/CUDA
    # support — a normal `pip install lightgbm` is CPU-only. We probe once
    # and fall back silently rather than crash mid-training.
    try:
        import lightgbm as _lgb_probe
        _test_model = _lgb_probe.LGBMRegressor(device_type="cuda", n_estimators=1, verbose=-1)
        _test_model.fit(np.random.rand(20, 3), np.random.rand(20))
        LGBM_DEVICE = "cuda"
        print("[GPU] LightGBM CUDA build detected — using device_type='cuda'")
    except Exception:
        try:
            _test_model = _lgb_probe.LGBMRegressor(device_type="gpu", n_estimators=1, verbose=-1)
            _test_model.fit(np.random.rand(20, 3), np.random.rand(20))
            LGBM_DEVICE = "gpu"
            print("[GPU] LightGBM OpenCL GPU build detected — using device_type='gpu'")
        except Exception:
            LGBM_DEVICE = "cpu"
            print("[GPU] LightGBM has no GPU build available — using device_type='cpu' "
                  "(install a CUDA-compiled lightgbm to change this)")

CATBOOST_TASK_TYPE = "GPU" if GPU_AVAILABLE else "CPU"

# ============================================================
# MACHINE LEARNING
# ============================================================

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.svm import SVR

from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error
)

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# ============================================================
# TIME SERIES
# ============================================================

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.api import VAR


# ============================================================
# PROJECT UTILITIES
# ============================================================

from utils.sort_year import sort_and_replace
from utils.save_metadata import save_metadata

# ============================================================
# PROPHET
# ============================================================

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("WARNING: Prophet is not installed.")

# ============================================================
# RANDOM SEEDS
# ============================================================

tf.random.set_seed(42)
np.random.seed(42)

# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = ("../data/features/Base_MALARIA_CS_CLEAN_2021_2024_with_weather.xlsx")
DATA_PATH_TEST = ("../data/features/Base_MALARIA_CS_CLEAN_2025_with_weather.xlsx")

REGISTRY_DIR = Path("../models/registry")
REGISTRY_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REGISTRY_DIR_META = Path("../models/metadata")
REGISTRY_DIR_META.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# TARGET
# ============================================================

target = (
    "Cas Confirmés (par TDR) palu consultations externes"
)

# ============================================================
# WEATHER VARIABLES
# ============================================================

meteo_vars = [
    "temperature_moyenne",
    "temperature_max",
    "temperature_min",
    "precipitation",
    "humidite",
    "vent",
    "rayonnement_solaire"
]

# ============================================================
# GLOBAL PARAMETERS
# ============================================================

TEST_RATIO = 0.20

# Monthly seasonality
SEASONAL_PERIOD = 12

# Number of target lags
N_LAGS = 12

# Rolling windows
ROLLING_WINDOWS = [3, 6, 12]

# Deep learning window
WINDOW = 6

RANDOM_STATE = 42

# Training version
VERSION = "1.0.0-gpu"

# Aggressive-search knobs — tuned for the A40's throughput.
# GPU-accelerated tree models (XGBoost/CatBoost, LightGBM if CUDA build) are
# cheap per fit, so they get a much larger randomized-search budget than the
# CPU-only estimators (RandomForest, SVR).
CV_SPLITS = 5
N_ITER_GPU_MODEL = 60   # XGBoost / CatBoost / LightGBM(if GPU)
N_ITER_CPU_MODEL = 30   # RandomForest / SVR / LightGBM(if CPU fallback)

# ============================================================
# 1. LOAD TRAIN / TEST DATA SEPARATELY
# ============================================================

print("\n" + "=" * 80)
print("1. LOAD TRAIN / TEST DATA")
print("=" * 80)

train_raw = pd.read_excel(DATA_PATH)
test_raw = pd.read_excel(DATA_PATH_TEST)

print("Train raw:", train_raw.shape)
print("Test raw :", test_raw.shape)

print("\nTrain periods:")
print(train_raw["PERIODE"].unique())

print("\nTest periods:")
print(test_raw["PERIODE"].unique())


# ============================================================
# 2. AGGREGATE TRAIN / TEST DATA
# ============================================================

print("\n" + "=" * 80)
print("2. AGGREGATE TRAIN / TEST DATA")
print("=" * 80)

agg_dict = {
    target: "sum",
    **{
        col: "mean"
        for col in meteo_vars
        if col in train_raw.columns
    }
}

# -----------------------------
# Training data: 2021-2024
# -----------------------------

train_df = (
    train_raw
    .groupby(["DRS", "PERIODE"])
    .agg(agg_dict)
    .reset_index()
)

# -----------------------------
# Test data: 2025
# -----------------------------

test_df = (
    test_raw
    .groupby(["DRS", "PERIODE"])
    .agg(agg_dict)
    .reset_index()
)

# Convert periods to datetime
train_df["PERIODE"] = pd.to_datetime(train_df["PERIODE"])
test_df["PERIODE"] = pd.to_datetime(test_df["PERIODE"])

# Sort
train_df = (
    train_df
    .sort_values(["DRS", "PERIODE"])
    .reset_index(drop=True)
)

test_df = (
    test_df
    .sort_values(["DRS", "PERIODE"])
    .reset_index(drop=True)
)

print("Aggregated train:", train_df.shape)
print("Aggregated test :", test_df.shape)

print("\nTrain period:")
print(train_df["PERIODE"].min(), "->", train_df["PERIODE"].max())

print("\nTest period:")
print(test_df["PERIODE"].min(), "->", test_df["PERIODE"].max())

print("\nRegions:")
print(train_df["DRS"].unique())


# ============================================================
# 3. DEFINE TRAIN / TEST PERIODS
# ============================================================

print("\n" + "=" * 80)
print("3. TRAIN / TEST SPLIT")
print("=" * 80)

TRAIN_END = train_df["PERIODE"].max()

TEST_START = test_df["PERIODE"].min()
TEST_END = test_df["PERIODE"].max()

print("Training:", train_df["PERIODE"].min(), "->", TRAIN_END)
print("Testing :", TEST_START, "->", TEST_END)

print("Train:", train_df.shape)
print("Test :", test_df.shape)


# ============================================================
# 4. COMBINE FOR FEATURE ENGINEERING
# ============================================================
#
# IMPORTANT:
# We combine 2021-2024 + 2025 BEFORE creating lags.
#
# This allows:
#
# January 2025 lag_1  -> December 2024
# January 2025 lag_2  -> November 2024
# ...
# January 2025 lag_12 -> January 2024
#
# Therefore, there is no leakage from future 2025 observations.
# ============================================================

all_df = pd.concat(
    [train_df, test_df],
    axis=0,
    ignore_index=True
)

all_df = (
    all_df
    .sort_values(["DRS", "PERIODE"])
    .reset_index(drop=True)
)

print("\nCombined dataset for feature engineering:")
print(all_df.shape)


# ============================================================
# 5. CREATE LAG FEATURES FOR ML
# ============================================================

LAGS = [1, 2, 3, 6, 12]


def create_ml_features(dataframe):

    data = dataframe.copy()

    data = data.sort_values(
        ["DRS", "PERIODE"]
    )

    # ----------------------------------------
    # Target lags
    # ----------------------------------------

    for lag in LAGS:

        data[f"target_lag_{lag}"] = (
            data
            .groupby("DRS")[target]
            .shift(lag)
        )

    # ----------------------------------------
    # Rolling features
    # ----------------------------------------

    data["target_roll_mean_3"] = (
        data
        .groupby("DRS")[target]
        .transform(
            lambda x:
            x.shift(1)
             .rolling(3)
             .mean()
        )
    )

    data["target_roll_mean_6"] = (
        data
        .groupby("DRS")[target]
        .transform(
            lambda x:
            x.shift(1)
             .rolling(6)
             .mean()
        )
    )

    # ----------------------------------------
    # Calendar features
    # ----------------------------------------

    data["month"] = data["PERIODE"].dt.month
    data["year"] = data["PERIODE"].dt.year

    # ----------------------------------------
    # Cyclic month encoding
    # ----------------------------------------

    data["month_sin"] = np.sin(
        2 * np.pi * data["month"] / 12
    )

    data["month_cos"] = np.cos(
        2 * np.pi * data["month"] / 12
    )

    return data


ml_df = create_ml_features(all_df)

print("\nFeature-engineered dataset:")
print(ml_df.head())
print(ml_df.shape)


# ============================================================
# 6. REMOVE ROWS WITHOUT REQUIRED LAGS
# ============================================================

ml_df = ml_df.dropna().copy()

print("\nAfter dropping missing lag/rolling values:")
print(ml_df.shape)


# ============================================================
# 7. SEPARATE TRAIN / TEST AGAIN
# ============================================================

ml_train = ml_df[
    ml_df["PERIODE"] <= TRAIN_END
].copy()

ml_test = ml_df[
    (ml_df["PERIODE"] >= TEST_START)
    &
    (ml_df["PERIODE"] <= TEST_END)
].copy()

print("\nML train:", ml_train.shape)
print("ML test :", ml_test.shape)

print(
    "\nML train period:",
    ml_train["PERIODE"].min(),
    "->",
    ml_train["PERIODE"].max()
)

print(
    "ML test period :",
    ml_test["PERIODE"].min(),
    "->",
    ml_test["PERIODE"].max()
)


# ============================================================
# 8. DEFINE ML FEATURES
# ============================================================

feature_cols = [
    "DRS",

    "temperature_moyenne",
    "temperature_max",
    "temperature_min",
    "precipitation",
    "humidite",
    "vent",
    "rayonnement_solaire",

    "target_lag_1",
    "target_lag_2",
    "target_lag_3",
    "target_lag_6",
    "target_lag_12",

    "target_roll_mean_3",
    "target_roll_mean_6",

    "month",
    "year",
    "month_sin",
    "month_cos"
]

feature_cols = [
    col
    for col in feature_cols
    if col in ml_df.columns
]

print("\nML features:")
print(feature_cols)


# ============================================================
# 9. X / y
# ============================================================

X_train = ml_train[feature_cols]
y_train = ml_train[target]

X_test = ml_test[feature_cols]
y_test = ml_test[target]

print("\nX_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_test :", X_test.shape)
print("y_test :", y_test.shape)


# ============================================================
# 10. ONE-HOT ENCODE DRS
# ============================================================

categorical_features = ["DRS"]

numeric_features = [
    col
    for col in feature_cols
    if col not in categorical_features
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        ),
        (
            "num",
            "passthrough",
            numeric_features
        )
    ]
)

# ===================================================
# Evaluation function
# ===================================================

def smape(y_true, y_pred):
    """
    Symmetric Mean Absolute Percentage Error (sMAPE)
    """
    denominator = np.abs(y_true) + np.abs(y_pred)

    mask = denominator != 0

    return (
        100
        * np.mean(
            2 * np.abs(y_pred[mask] - y_true[mask])
            / denominator[mask]
        )
    )


def evaluate_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    model_name
):

    print("\n" + "=" * 80)
    print(model_name)
    print("=" * 80)

    # Train model
    model.fit(
        X_train,
        y_train
    )

    # Predictions
    predictions = model.predict(X_test)

    # Prevent negative predictions
    predictions = np.maximum(
        predictions,
        0
    )

    # MAE
    mae = mean_absolute_error(
        y_test,
        predictions
    )

    # RMSE
    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    # sMAPE
    smape_value = smape(
        y_test.to_numpy(),
        predictions
    )

    # R²
    r2 = r2_score(
        y_test,
        predictions
    )

    # MAPE
    y_true = y_test.to_numpy()

    mask = y_true != 0

    mape = (
        100
        * np.mean(
            np.abs(
                (y_true[mask] - predictions[mask])
                / y_true[mask]
            )
        )
    )

    # Custom regression accuracy
    accuracy = max(
        0,
        100 - mape
    )

    print(f"MAE      : {mae:.4f}")
    print(f"RMSE     : {rmse:.4f}")
    print(f"sMAPE    : {smape_value:.2f}%")
    print(f"MAPE     : {mape:.2f}%")
    print(f"Accuracy : {accuracy:.2f}%")
    print(f"R²       : {r2:.4f}")

    return {
        "model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "sMAPE": smape_value,
        "MAPE": mape,
        "Accuracy": accuracy,
        "R2": r2,
        "model_object": model,
        "predictions": predictions
    }


def plot_model_predictions(
    full_df,
    test_df,
    predictions,
    model_name,
    target,
    test_start,
    save_plot=False
):
    """
    Plot historical target values and test predictions
    for every DRS separately.
    """

    if save_plot:
        current_date = pd.Timestamp.today().strftime("%Y-%m-%d")

        output_dir = (
            Path("../models/registry")
            / f"Plot_DRS_{current_date}"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    plot_test = test_df[
        ["DRS", "PERIODE", target]
    ].copy()

    plot_test["prediction"] = np.asarray(
        predictions
    )

    plot_test["PERIODE"] = pd.to_datetime(
        plot_test["PERIODE"]
    )

    plot_test = plot_test.sort_values(
        ["DRS", "PERIODE"]
    )

    regions = (
        plot_test["DRS"]
        .dropna()
        .unique()
    )

    for region in regions:

        historical = (
            full_df[
                full_df["DRS"] == region
            ]
            .copy()
            .sort_values("PERIODE")
        )

        historical["PERIODE"] = pd.to_datetime(
            historical["PERIODE"]
        )

        test_region = (
            plot_test[
                plot_test["DRS"] == region
            ]
            .sort_values("PERIODE")
        )

        plt.figure(
            figsize=(14, 7)
        )

        plt.plot(
            historical["PERIODE"],
            historical[target],
            color="black",
            linewidth=2,
            label="Historical"
        )

        plt.plot(
            test_region["PERIODE"],
            test_region[target],
            color="royalblue",
            linewidth=2.5,
            marker="o",
            markersize=5,
            label="Test actual"
        )

        plt.plot(
            test_region["PERIODE"],
            test_region["prediction"],
            color="red",
            linewidth=2.5,
            linestyle="--",
            marker="o",
            markersize=5,
            label=f"{model_name} prediction"
        )

        plt.axvline(
            test_start,
            color="gray",
            linestyle=":",
            linewidth=2,
            label="Test start"
        )

        plt.axvspan(
            test_start,
            historical["PERIODE"].max(),
            color="orange",
            alpha=0.08
        )

        plt.title(
            f"{model_name} — {region}",
            fontsize=16,
            fontweight="bold"
        )

        plt.xlabel(
            "PERIODE",
            fontsize=12
        )

        plt.ylabel(
            target,
            fontsize=12
        )

        plt.grid(
            True,
            alpha=0.25
        )

        plt.legend(
            loc="best",
            frameon=True
        )

        plt.tight_layout()

        if save_plot:

            safe_model_name = str(model_name).replace(
                "/", "_"
            ).replace(
                "\\", "_"
            ).replace(
                " ", "_"
            )

            safe_region = str(region).replace(
                "/", "_"
            ).replace(
                "\\", "_"
            ).replace(
                " ", "_"
            )

            filename = (
                f"{safe_model_name}_"
                f"{safe_region}_test_predictions.png"
            )

            filepath = output_dir / filename

            plt.savefig(
                filepath,
                dpi=300,
                bbox_inches="tight"
            )

        plt.close()


def forecast_next_year_ml(
    model,
    df,
    feature_cols,
    target,
    horizon=12,
    weather_vars=None
):
    """
    Forecast the next `horizon` months for every DRS.
    """

    data = df.copy()

    data["PERIODE"] = pd.to_datetime(
        data["PERIODE"]
    )

    data = data.sort_values(
        ["DRS", "PERIODE"]
    )

    last_date = data["PERIODE"].max()

    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=horizon,
        freq="MS"
    )

    all_forecasts = []

    for region in data["DRS"].unique():

        region_history = (
            data[
                data["DRS"] == region
            ]
            .sort_values("PERIODE")
            .copy()
        )

        last_12 = region_history.tail(12).copy()

        future = pd.DataFrame({
            "PERIODE": future_dates
        })

        future["DRS"] = region

        if weather_vars is not None:

            for col in weather_vars:

                if col in region_history.columns:

                    values = (
                        last_12[col]
                        .to_numpy()
                    )

                    future[col] = np.resize(
                        values,
                        horizon
                    )

        combined = pd.concat(
            [
                region_history,
                future
            ],
            ignore_index=True
        )

        combined = combined.sort_values(
            "PERIODE"
        ).reset_index(drop=True)

        for lag in LAGS:

            combined[
                f"target_lag_{lag}"
            ] = (
                combined[target]
                .shift(lag)
            )

        combined[
            "target_roll_mean_3"
        ] = (
            combined[target]
            .shift(1)
            .rolling(3)
            .mean()
        )

        combined[
            "target_roll_mean_6"
        ] = (
            combined[target]
            .shift(1)
            .rolling(6)
            .mean()
        )

        combined["month"] = (
            combined["PERIODE"].dt.month
        )

        combined["year"] = (
            combined["PERIODE"].dt.year
        )

        combined["month_sin"] = np.sin(
            2 * np.pi *
            combined["month"] / 12
        )

        combined["month_cos"] = np.cos(
            2 * np.pi *
            combined["month"] / 12
        )

        predictions = []

        future_indices = combined.index[
            combined["PERIODE"] > last_date
        ]

        for idx in future_indices:

            row = combined.loc[
                [idx],
                feature_cols
            ]

            prediction = model.predict(
                row
            )[0]

            prediction = max(
                0,
                prediction
            )

            predictions.append(
                prediction
            )

            combined.loc[
                idx,
                target
            ] = prediction

            combined[
                "target_roll_mean_3"
            ] = (
                combined[target]
                .shift(1)
                .rolling(3)
                .mean()
            )

            combined[
                "target_roll_mean_6"
            ] = (
                combined[target]
                .shift(1)
                .rolling(6)
                .mean()
            )

            for lag in LAGS:

                combined[
                    f"target_lag_{lag}"
                ] = (
                    combined[target]
                    .shift(lag)
                )

        region_forecast = pd.DataFrame({
            "DRS": region,
            "PERIODE": future_dates,
            "prediction": predictions
        })

        all_forecasts.append(
            region_forecast
        )

    return pd.concat(
        all_forecasts,
        ignore_index=True
    )


def plot_next_year_forecast(
    historical_df,
    forecast_df,
    model_name,
    target,
    save_plot=False
):
    """
    Plot complete historical data followed by
    next-year forecast for every DRS.
    """

    historical_df = historical_df.copy()
    forecast_df = forecast_df.copy()

    if save_plot:
        current_date = pd.Timestamp.today().strftime("%Y-%m-%d")

        output_dir = (
            Path("../models/registry")
            / f"Plot_DRS_{current_date}"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    historical_df["PERIODE"] = pd.to_datetime(
        historical_df["PERIODE"]
    )

    forecast_df["PERIODE"] = pd.to_datetime(
        forecast_df["PERIODE"]
    )

    regions = (
        historical_df["DRS"]
        .dropna()
        .unique()
    )

    for region in regions:

        history = (
            historical_df[
                historical_df["DRS"] == region
            ]
            .sort_values("PERIODE")
        )

        forecast = (
            forecast_df[
                forecast_df["DRS"] == region
            ]
            .sort_values("PERIODE")
        )

        if forecast.empty:
            continue

        plt.figure(
            figsize=(15, 7)
        )

        plt.plot(
            history["PERIODE"],
            history[target],
            color="black",
            linewidth=2.5,
            label="Historical"
        )

        plt.plot(
            forecast["PERIODE"],
            forecast["prediction"],
            color="red",
            linewidth=2.5,
            linestyle="--",
            marker="o",
            markersize=5,
            label=f"{model_name} — Next 12 Months"
        )

        forecast_start = forecast[
            "PERIODE"
        ].min()

        plt.axvline(
            forecast_start,
            color="gray",
            linestyle=":",
            linewidth=2,
            label="Forecast start"
        )

        plt.axvspan(
            forecast_start,
            forecast["PERIODE"].max(),
            color="red",
            alpha=0.06
        )

        plt.title(
            f"{model_name} — "
            f"{region}\n"
            f"Historical + Next 12 Months Forecast",
            fontsize=16,
            fontweight="bold"
        )

        plt.xlabel(
            "PERIODE",
            fontsize=12
        )

        plt.ylabel(
            target,
            fontsize=12
        )

        plt.grid(
            alpha=0.25
        )

        plt.legend(
            fontsize=11
        )

        plt.tight_layout()

        if save_plot:

            safe_model_name = str(model_name).replace(
                "/", "_"
            ).replace(
                "\\", "_"
            ).replace(
                " ", "_"
            )

            safe_region = str(region).replace(
                "/", "_"
            ).replace(
                "\\", "_"
            ).replace(
                " ", "_"
            )

            filename = (
                f"{safe_model_name}_"
                f"{safe_region}_next_year_forecast.png"
            )

            filepath = output_dir / filename

            plt.savefig(
                filepath,
                dpi=300,
                bbox_inches="tight"
            )

        plt.close()

# ============================================================
# PART 2 — CONTINUATION
# ============================================================

import itertools

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.api import VAR


def compute_metrics(y_true, y_pred, model_name):

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 0)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    smape_value = smape(y_true, y_pred)

    mask = y_true != 0
    if mask.any():
        mape = 100 * np.mean(
            np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
        )
        accuracy = max(0, 100 - mape)
    else:
        mape = np.nan
        accuracy = np.nan

    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan

    print(f"\n{model_name}")
    print(f"MAE      : {mae:.4f}")
    print(f"RMSE     : {rmse:.4f}")
    print(f"sMAPE    : {smape_value:.2f}%")
    print(f"MAPE     : {mape:.2f}%")
    print(f"Accuracy : {accuracy:.2f}%")
    print(f"R²       : {r2:.4f}")

    return {
        "model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "sMAPE": smape_value,
        "MAPE": mape,
        "Accuracy": accuracy,
        "R2": r2,
    }


def results_table(results_dict):
    return (
        pd.DataFrame(list(results_dict.values()))
        .sort_values("RMSE")
        .reset_index(drop=True)
    )


def align_predictions(preds_long_df):
    merged = (
        test_df[["DRS", "PERIODE", target]]
        .merge(preds_long_df, on=["DRS", "PERIODE"], how="inner")
        .sort_values(["DRS", "PERIODE"])
        .reset_index(drop=True)
    )

    return merged[["DRS", "PERIODE", target]], merged["prediction"].to_numpy()


def plot_best(model_name, preds_long_df, label_prefix=""):

    aligned_test_df, aligned_preds = align_predictions(preds_long_df)

    plot_model_predictions(
        full_df=all_df,
        test_df=aligned_test_df,
        predictions=aligned_preds,
        model_name=f"{label_prefix}{model_name}",
        target=target,
        test_start=TEST_START,
        save_plot=True,
    )


# ============================================================
# 10. MACHINE LEARNING — AGGRESSIVE RANDOMIZED SEARCH
#     (Random Forest, XGBoost[GPU], LightGBM[GPU if available],
#      CatBoost[GPU], SVR)
#
#     Switched from GridSearchCV -> RandomizedSearchCV: the GPU
#     tree models (XGBoost/CatBoost, LightGBM if CUDA build) fit
#     fast enough on the A40 that a much wider search space is
#     affordable. TimeSeriesSplit bumped 3 -> 5 folds.
# ============================================================

print("\n" + "=" * 80)
print("10. MACHINE LEARNING MODELS — AGGRESSIVE RANDOMIZED SEARCH")
print(f"    (device: XGBoost={XGB_DEVICE}, LightGBM={LGBM_DEVICE}, "
      f"CatBoost={CATBOOST_TASK_TYPE})")
print("=" * 80)

tscv = TimeSeriesSplit(n_splits=CV_SPLITS)

ml_specs = {
    "RandomForest": {
        "estimator": RandomForestRegressor(
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        ),
        "param_distributions": {
            "model__n_estimators": [200, 300, 400, 600, 800, 1000],
            "model__max_depth": [None, 6, 10, 15, 20, 30],
            "model__min_samples_split": [2, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 4, 8],
            "model__max_features": ["sqrt", "log2", None, 0.5, 0.7],
        },
        "n_iter": N_ITER_CPU_MODEL,
    },
    "XGBoost": {
        "estimator": XGBRegressor(
            random_state=RANDOM_STATE,
            objective="reg:squarederror",
            tree_method=XGB_TREE_METHOD,
            device=XGB_DEVICE,
            n_jobs=N_JOBS if XGB_DEVICE == "cpu" else 1,
        ),
        "param_distributions": {
            "model__n_estimators": [200, 400, 600, 800, 1000, 1500],
            "model__max_depth": [3, 4, 5, 6, 7, 8, 9, 10],
            "model__learning_rate": [0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2],
            "model__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "model__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            "model__min_child_weight": [1, 2, 3, 5, 7],
            "model__gamma": [0, 0.1, 0.2, 0.5],
            "model__reg_alpha": [0, 0.01, 0.1, 1],
            "model__reg_lambda": [0.5, 1, 1.5, 2],
        },
        "n_iter": N_ITER_GPU_MODEL if XGB_DEVICE == "cuda" else N_ITER_CPU_MODEL,
    },
    "LightGBM": {
        "estimator": LGBMRegressor(
            random_state=RANDOM_STATE,
            verbose=-1,
            device_type=LGBM_DEVICE,
            n_jobs=N_JOBS if LGBM_DEVICE == "cpu" else 1,
        ),
        "param_distributions": {
            "model__n_estimators": [200, 400, 600, 800, 1000, 1500],
            "model__num_leaves": [15, 31, 63, 127, 255],
            "model__max_depth": [-1, 4, 6, 8, 10, 15],
            "model__learning_rate": [0.01, 0.02, 0.05, 0.08, 0.1, 0.15],
            "model__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "model__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            "model__min_child_samples": [5, 10, 20, 30],
            "model__reg_alpha": [0, 0.01, 0.1, 1],
            "model__reg_lambda": [0, 0.01, 0.1, 1],
        },
        "n_iter": N_ITER_GPU_MODEL if LGBM_DEVICE != "cpu" else N_ITER_CPU_MODEL,
    },
    "CatBoost": {
        # Added: catboost was already imported but unused in the original
        # script. It has first-class, zero-config GPU support, so it's a
        # natural fit for the A40 and a cheap addition to the comparison.
        "estimator": CatBoostRegressor(
            random_state=RANDOM_STATE,
            task_type=CATBOOST_TASK_TYPE,
            devices="0" if CATBOOST_TASK_TYPE == "GPU" else None,
            verbose=False,
        ),
        "param_distributions": {
            "model__iterations": [300, 500, 800, 1200, 1600],
            "model__depth": [4, 6, 8, 10],
            "model__learning_rate": [0.01, 0.02, 0.05, 0.08, 0.1],
            "model__l2_leaf_reg": [1, 3, 5, 7, 9],
            "model__border_count": [32, 64, 128, 254],
        },
        "n_iter": N_ITER_GPU_MODEL if CATBOOST_TASK_TYPE == "GPU" else N_ITER_CPU_MODEL,
    },
    "SVR": {
        "estimator": SVR(),
        "param_distributions": {
            "model__C": [0.1, 1, 5, 10, 50, 100],
            "model__epsilon": [0.01, 0.05, 0.1, 0.3, 0.5, 1.0],
            "model__kernel": ["rbf", "linear", "poly"],
            "model__gamma": ["scale", "auto"],
        },
        "n_iter": N_ITER_CPU_MODEL,
    },
}

ml_results = {}
ml_best_estimators = {}

for name, spec in ml_specs.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", spec["estimator"]),
        ]
    )

    # GPU-backed estimators (XGBoost cuda / CatBoost GPU) generally can't
    # be fit concurrently by multiple sklearn worker processes without
    # contending for the same device, so we cap search-level parallelism
    # to 1 in that case and rely on each GPU fit being fast; CPU models
    # use every vCPU for the search itself.
    uses_gpu_device = (
        (name == "XGBoost" and XGB_DEVICE == "cuda")
        or (name == "LightGBM" and LGBM_DEVICE != "cpu")
        or (name == "CatBoost" and CATBOOST_TASK_TYPE == "GPU")
    )
    search_n_jobs = 1 if uses_gpu_device else N_JOBS

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=spec["param_distributions"],
        n_iter=spec["n_iter"],
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=search_n_jobs,
        random_state=RANDOM_STATE,
        verbose=1,
    )

    t0 = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - t0

    print(f"\n{name} best params ({spec['n_iter']} candidates, "
          f"{elapsed:.1f}s): {search.best_params_}")

    result = evaluate_model(
        search.best_estimator_,
        X_train,
        y_train,
        X_test,
        y_test,
        name,
    )

    ml_results[name] = result
    ml_best_estimators[name] = search.best_estimator_

ml_results_df = results_table(ml_results)

print("\nML MODEL COMPARISON (sorted by RMSE):")
print(ml_results_df.drop(columns=["model_object", "predictions"], errors="ignore"))

best_ml_name = ml_results_df.iloc[0]["model"]
best_ml_result = ml_results[best_ml_name]
best_ml_estimator = ml_best_estimators[best_ml_name]

print(f"\nBest ML model: {best_ml_name}")

plot_model_predictions(
    full_df=all_df,
    test_df=ml_test,
    predictions=best_ml_result["predictions"],
    model_name=f"Best ML Model: {best_ml_name}",
    target=target,
    test_start=TEST_START,
    save_plot=True,
)


# ============================================================
# 11. TIME SERIES MODELS — PER-REGION FORECASTING
#     (ARIMA, SARIMA, SARIMAX, Holt-Winters/ETS, VAR, Prophet)
#
#     statsmodels/Prophet have no GPU path, so the per-region
#     loops (independent across DRS) are parallelized across the
#     pod's 9 vCPUs with joblib instead.
# ============================================================

print("\n" + "=" * 80)
print("11. TIME SERIES MODELS")
print("=" * 80)

ts_horizon = test_df["PERIODE"].nunique()
regions = sorted(train_df["DRS"].dropna().unique())


def get_region_frame(df, region):
    frame = (
        df[df["DRS"] == region]
        .sort_values("PERIODE")
        .set_index("PERIODE")
    )

    frame.index = pd.DatetimeIndex(frame.index)
    frame = frame.asfreq("MS")

    return frame


ts_results = {}
ts_predictions_long = {}

# ------------------------------------------------------------
# 11a. ARIMA — light-ish grid over (p, d, q), fit in parallel
#      across regions (each region's model search is independent)
# ------------------------------------------------------------

print("\n--- ARIMA (parallel across regions) ---")

arima_orders = list(itertools.product([0, 1, 2, 3], [0, 1, 2], [0, 1, 2, 3]))


def _fit_arima_region(region):
    train_frame = get_region_frame(train_df, region)
    test_frame = get_region_frame(test_df, region)
    y_hist = train_frame[target]

    best_aic, best_fit = np.inf, None

    for order in arima_orders:
        try:
            fit = ARIMA(y_hist, order=order).fit()
            if fit.aic < best_aic:
                best_aic, best_fit = fit.aic, fit
        except Exception:
            continue

    if best_fit is None:
        return None

    forecast = np.maximum(best_fit.forecast(steps=ts_horizon).to_numpy(), 0)

    return pd.DataFrame(
        {
            "DRS": region,
            "PERIODE": test_frame.index[:ts_horizon],
            "prediction": forecast,
        }
    )


arima_preds = Parallel(n_jobs=N_JOBS)(
    delayed(_fit_arima_region)(region) for region in regions
)
arima_preds = [df for df in arima_preds if df is not None]

ts_predictions_long["ARIMA"] = pd.concat(arima_preds, ignore_index=True)
aligned_df, aligned_preds = align_predictions(ts_predictions_long["ARIMA"])
ts_results["ARIMA"] = compute_metrics(aligned_df[target], aligned_preds, "ARIMA")

# ------------------------------------------------------------
# 11b. SARIMA — seasonal grid, fit in parallel across regions
# ------------------------------------------------------------

print("\n--- SARIMA (parallel across regions) ---")

sarima_base_orders = [(1, 1, 1), (1, 1, 0), (0, 1, 1), (2, 1, 1), (1, 1, 2)]
sarima_seasonal_orders = [
    (P, 1, Q, SEASONAL_PERIOD) for P, Q in itertools.product([0, 1, 2], [0, 1, 2])
]


def _fit_sarima_region(region):
    train_frame = get_region_frame(train_df, region)
    test_frame = get_region_frame(test_df, region)
    y_hist = train_frame[target]

    best_aic, best_fit = np.inf, None

    for order, seasonal_order in itertools.product(
        sarima_base_orders, sarima_seasonal_orders
    ):
        try:
            fit = SARIMAX(
                y_hist,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
            if fit.aic < best_aic:
                best_aic, best_fit = fit.aic, fit
        except Exception:
            continue

    if best_fit is None:
        return None

    forecast = np.maximum(best_fit.forecast(steps=ts_horizon).to_numpy(), 0)

    return pd.DataFrame(
        {
            "DRS": region,
            "PERIODE": test_frame.index[:ts_horizon],
            "prediction": forecast,
        }
    )


sarima_preds = Parallel(n_jobs=N_JOBS)(
    delayed(_fit_sarima_region)(region) for region in regions
)
sarima_preds = [df for df in sarima_preds if df is not None]

ts_predictions_long["SARIMA"] = pd.concat(sarima_preds, ignore_index=True)
aligned_df, aligned_preds = align_predictions(ts_predictions_long["SARIMA"])
ts_results["SARIMA"] = compute_metrics(aligned_df[target], aligned_preds, "SARIMA")

# ------------------------------------------------------------
# 11c. SARIMAX — same seasonal grid + weather exog, parallel
# ------------------------------------------------------------

print("\n--- SARIMAX (parallel across regions) ---")

exog_cols = [c for c in meteo_vars if c in train_df.columns]

if not exog_cols:
    print("SARIMAX skipped: no exogenous columns available.")
    ts_predictions_long["SARIMAX"] = pd.DataFrame(columns=["DRS", "PERIODE", "prediction"])

else:

    def _fit_sarimax_region(region):
        train_frame = get_region_frame(train_df, region)
        test_frame = get_region_frame(test_df, region)

        y_hist = train_frame[target]
        exog_hist = train_frame[exog_cols]
        exog_future = test_frame[exog_cols].iloc[:ts_horizon]

        if y_hist.empty:
            return None
        if len(exog_future) < ts_horizon:
            return None
        if y_hist.isna().any():
            return None
        if exog_hist.isna().any().any():
            return None
        if exog_future.isna().any().any():
            return None

        best_aic, best_fit = np.inf, None

        for order, seasonal_order in itertools.product(
            sarima_base_orders, sarima_seasonal_orders
        ):
            try:
                fit = SARIMAX(
                    y_hist,
                    exog=exog_hist,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)

                if np.isfinite(fit.aic) and fit.aic < best_aic:
                    best_aic, best_fit = fit.aic, fit

            except Exception:
                continue

        if best_fit is None:
            return None

        forecast = np.maximum(
            best_fit.forecast(steps=ts_horizon, exog=exog_future).to_numpy(), 0
        )

        return pd.DataFrame(
            {
                "DRS": region,
                "PERIODE": test_frame.index[:ts_horizon],
                "prediction": forecast,
            }
        )

    sarimax_preds = Parallel(n_jobs=N_JOBS)(
        delayed(_fit_sarimax_region)(region) for region in regions
    )
    sarimax_preds = [df for df in sarimax_preds if df is not None]

    if sarimax_preds:
        ts_predictions_long["SARIMAX"] = pd.concat(sarimax_preds, ignore_index=True)
        aligned_df, aligned_preds = align_predictions(ts_predictions_long["SARIMAX"])
        ts_results["SARIMAX"] = compute_metrics(aligned_df[target], aligned_preds, "SARIMAX")
    else:
        print("SARIMAX: no predictions generated.")
        ts_predictions_long["SARIMAX"] = pd.DataFrame(columns=["DRS", "PERIODE", "prediction"])

# ------------------------------------------------------------
# 11d. Holt-Winters / ETS — grid over trend/seasonal, parallel
# ------------------------------------------------------------

print("\n--- Holt-Winters / ETS (parallel across regions) ---")

ets_grid = list(itertools.product([None, "add", "mul"], [None, "add", "mul"]))


def _fit_ets_region(region):
    train_frame = get_region_frame(train_df, region)
    test_frame = get_region_frame(test_df, region)
    y_hist = train_frame[target]

    best_aic, best_fit = np.inf, None

    for trend, seasonal in ets_grid:
        try:
            fit = ExponentialSmoothing(
                y_hist,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=SEASONAL_PERIOD if seasonal else None,
                initialization_method="estimated",
            ).fit()
            if fit.aic < best_aic:
                best_aic, best_fit = fit.aic, fit
        except Exception:
            continue

    if best_fit is None:
        return None

    forecast = np.maximum(best_fit.forecast(ts_horizon).to_numpy(), 0)

    return pd.DataFrame(
        {
            "DRS": region,
            "PERIODE": test_frame.index[:ts_horizon],
            "prediction": forecast,
        }
    )


ets_preds = Parallel(n_jobs=N_JOBS)(
    delayed(_fit_ets_region)(region) for region in regions
)
ets_preds = [df for df in ets_preds if df is not None]

ts_predictions_long["ETS"] = pd.concat(ets_preds, ignore_index=True)
aligned_df, aligned_preds = align_predictions(ts_predictions_long["ETS"])
ts_results["ETS"] = compute_metrics(aligned_df[target], aligned_preds, "Holt-Winters / ETS")

# ------------------------------------------------------------
# 11e. VAR — fit ONCE, jointly across all DRS regions
#      (single global fit — nothing to parallelize)
# ------------------------------------------------------------

print("\n--- VAR (joint across regions) ---")

wide_train = (
    train_df.pivot(index="PERIODE", columns="DRS", values=target)
    .sort_index()
    .asfreq("MS")
    .dropna(axis=1, how="any")
)

wide_test_index = (
    test_df["PERIODE"].drop_duplicates().sort_values().reset_index(drop=True)
)

var_lags = [1, 2, 3, 6, 9, 12]
best_ic, best_lag = np.inf, None

for lag in var_lags:
    try:
        candidate = VAR(wide_train).fit(lag)
        if candidate.aic < best_ic:
            best_ic, best_lag = candidate.aic, lag
    except Exception:
        continue

var_fit = VAR(wide_train).fit(best_lag)
print(f"VAR best lag order: {best_lag}")

var_forecast = var_fit.forecast(
    wide_train.values[-best_lag:], steps=ts_horizon
)

var_forecast_df = pd.DataFrame(
    np.maximum(var_forecast, 0),
    index=wide_test_index.iloc[:ts_horizon],
    columns=wide_train.columns,
)

var_preds = (
    var_forecast_df.reset_index()
    .melt(id_vars="PERIODE", var_name="DRS", value_name="prediction")
)

ts_predictions_long["VAR"] = var_preds
aligned_df, aligned_preds = align_predictions(ts_predictions_long["VAR"])
ts_results["VAR"] = compute_metrics(aligned_df[target], aligned_preds, "VAR")

# ------------------------------------------------------------
# 11f. Prophet — light grid over seasonality settings
#      Kept sequential: Prophet's cmdstanpy backend does not
#      reliably parallelize fits across processes (compilation
#      lock contention), so we leave this loop as-is rather than
#      risk flaky failures on the pod.
# ------------------------------------------------------------

if PROPHET_AVAILABLE:

    print("\n--- Prophet ---")

    prophet_grid = list(
        itertools.product(
            ["additive", "multiplicative"],
            [0.01, 0.05, 0.1, 0.5],
        )
    )

    prophet_preds = []

    for region in regions:

        train_frame = get_region_frame(train_df, region).reset_index()
        test_frame = get_region_frame(test_df, region).reset_index()

        prophet_train = train_frame.rename(
            columns={"PERIODE": "ds", target: "y"}
        )
        prophet_future = test_frame.rename(columns={"PERIODE": "ds"}).iloc[
            :ts_horizon
        ]

        if prophet_future[exog_cols].isna().any().any():
            continue

        best_mae, best_forecast = np.inf, None

        for seasonality_mode, cps in prophet_grid:
            try:
                m = Prophet(
                    seasonality_mode=seasonality_mode,
                    changepoint_prior_scale=cps,
                    yearly_seasonality=True,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                )
                for col in exog_cols:
                    m.add_regressor(col)

                m.fit(prophet_train[["ds", "y"] + exog_cols])

                forecast = m.predict(prophet_future[["ds"] + exog_cols])
                pred_values = np.maximum(forecast["yhat"].to_numpy(), 0)

                train_fit = m.predict(prophet_train[["ds"] + exog_cols])
                mae = mean_absolute_error(
                    prophet_train["y"], np.maximum(train_fit["yhat"], 0)
                )

                if mae < best_mae:
                    best_mae, best_forecast = mae, pred_values

            except Exception:
                continue

        if best_forecast is None:
            continue

        prophet_preds.append(
            pd.DataFrame(
                {
                    "DRS": region,
                    "PERIODE": test_frame["PERIODE"].iloc[:ts_horizon].values,
                    "prediction": best_forecast,
                }
            )
        )

    if prophet_preds:
        ts_predictions_long["Prophet"] = pd.concat(prophet_preds, ignore_index=True)
        aligned_df, aligned_preds = align_predictions(ts_predictions_long["Prophet"])
        ts_results["Prophet"] = compute_metrics(
            aligned_df[target], aligned_preds, "Prophet"
        )

else:
    print("\nSkipping Prophet (not installed).")

# ------------------------------------------------------------
# 11g. Compare all time-series models, plot the best one
# ------------------------------------------------------------

ts_results_df = results_table(ts_results)

print("\nTIME SERIES MODEL COMPARISON (sorted by RMSE):")
print(ts_results_df)

best_ts_name = ts_results_df.iloc[0]["model"]
best_ts_key = [k for k, v in ts_results.items() if v["model"] == best_ts_name][0]

print(f"\nBest time-series model: {best_ts_name}")

plot_best(best_ts_name, ts_predictions_long[best_ts_key], label_prefix="Best TS Model: ")


# ============================================================
# 12. DEEP LEARNING — RNN, LSTM, GRU (GPU, mixed precision)
# ============================================================

print("\n" + "=" * 80)
print("12. DEEP LEARNING MODELS")
print(f"    (device: {'GPU (mixed_float16)' if GPU_AVAILABLE else 'CPU'})")
print("=" * 80)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    SimpleRNN,
    LSTM,
    GRU,
    Dense,
    Dropout,
    Concatenate,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

dl_numeric_cols = [c for c in meteo_vars if c in all_df.columns] + [target]

dl_scaler = StandardScaler()
dl_scaler.fit(all_df.loc[all_df["PERIODE"] <= TRAIN_END, dl_numeric_cols])

dl_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
dl_encoder.fit(all_df[["DRS"]])


def build_sequences(df, window):

    X_seq, X_static, y, meta = [], [], [], []

    for region in sorted(df["DRS"].dropna().unique()):

        region_df = (
            df[df["DRS"] == region].sort_values("PERIODE").reset_index(drop=True)
        )

        scaled = dl_scaler.transform(region_df[dl_numeric_cols])
        static_vec = dl_encoder.transform(region_df[["DRS"]].iloc[:1])[0]

        for i in range(window, len(region_df)):
            X_seq.append(scaled[i - window : i])
            X_static.append(static_vec)
            y.append(region_df.loc[i, target])
            meta.append(
                {"DRS": region, "PERIODE": region_df.loc[i, "PERIODE"]}
            )

    return (
        np.asarray(X_seq, dtype="float32"),
        np.asarray(X_static, dtype="float32"),
        np.asarray(y, dtype="float32"),
        pd.DataFrame(meta),
    )


X_seq_all, X_static_all, y_seq_all, meta_all = build_sequences(all_df, WINDOW)

train_mask = meta_all["PERIODE"] <= TRAIN_END
test_mask = (meta_all["PERIODE"] >= TEST_START) & (meta_all["PERIODE"] <= TEST_END)

X_seq_train, X_static_train, y_seq_train = (
    X_seq_all[train_mask.values],
    X_static_all[train_mask.values],
    y_seq_all[train_mask.values],
)
X_seq_test, X_static_test, y_seq_test = (
    X_seq_all[test_mask.values],
    X_static_all[test_mask.values],
    y_seq_all[test_mask.values],
)
meta_test = meta_all[test_mask.values].reset_index(drop=True)

n_timesteps = X_seq_train.shape[1]
n_features = X_seq_train.shape[2]
n_static = X_static_train.shape[1]


def build_dl_model(recurrent_layer, units, dropout, num_layers=1):
    """
    Mixed-precision-safe model: internal layers compute in float16 on the
    A40's Tensor Cores (global policy set above), but the final Dense
    output is forced back to float32 for numerically stable loss/metrics —
    required practice under `mixed_float16`.
    """

    seq_input = Input(shape=(n_timesteps, n_features), name="sequence_input")
    static_input = Input(shape=(n_static,), name="region_input")

    x = seq_input
    for layer_idx in range(num_layers):
        return_sequences = layer_idx < num_layers - 1
        x = recurrent_layer(units, return_sequences=return_sequences)(x)
        x = Dropout(dropout)(x)

    merged = Concatenate()([x, static_input])
    merged = Dense(32, activation="relu")(merged)
    merged = Dropout(dropout)(merged)
    # Force float32 output under mixed precision
    output = Dense(1, activation="linear", dtype="float32")(merged)

    model = Model(inputs=[seq_input, static_input], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")

    return model


dl_specs = {
    "RNN": SimpleRNN,
    "LSTM": LSTM,
    "GRU": GRU,
}

# Wider search now that fits run on the A40 — still bounded via random
# sampling so the whole DL stage doesn't explode combinatorially.
dl_param_space = list(
    itertools.product([32, 64, 128, 256], [0.0, 0.1, 0.2, 0.3], [1, 2], [64, 128])
)  # (units, dropout, num_layers, batch_size)

rng = np.random.default_rng(RANDOM_STATE)
DL_N_ITER = 10  # sampled combos per architecture (RNN/LSTM/GRU)
dl_sampled_space = [
    dl_param_space[i]
    for i in rng.choice(len(dl_param_space), size=DL_N_ITER, replace=False)
]

dl_results = {}
dl_predictions_long = {}

for name, layer_cls in dl_specs.items():

    print(f"\n--- {name} ({DL_N_ITER} sampled configs) ---")

    best_val_loss, best_model = np.inf, None

    for units, dropout, num_layers, batch_size in dl_sampled_space:

        model = build_dl_model(layer_cls, units, dropout, num_layers)

        history = model.fit(
            [X_seq_train, X_static_train],
            y_seq_train,
            validation_split=0.15,
            epochs=100,
            batch_size=batch_size,  # A40's 48GB VRAM easily fits these batches
            shuffle=False,
            callbacks=[
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-5),
            ],
            verbose=0,
        )

        val_loss = min(history.history["val_loss"])

        if val_loss < best_val_loss:
            best_val_loss, best_model = val_loss, model
            print(f"    new best: units={units} dropout={dropout} "
                  f"layers={num_layers} batch={batch_size} val_loss={val_loss:.4f}")

    predictions = np.maximum(
        best_model.predict([X_seq_test, X_static_test], batch_size=256).flatten(), 0
    )

    result = compute_metrics(y_seq_test, predictions, name)
    dl_results[name] = result

    preds_long = meta_test.copy()
    preds_long["prediction"] = predictions
    dl_predictions_long[name] = preds_long

dl_results_df = results_table(dl_results)

print("\nDEEP LEARNING MODEL COMPARISON (sorted by RMSE):")
print(dl_results_df)

best_dl_name = dl_results_df.iloc[0]["model"]

print(f"\nBest deep learning model: {best_dl_name}")

plot_best(
    best_dl_name,
    dl_predictions_long[best_dl_name],
    label_prefix="Best DL Model: ",
)


# ============================================================
# 13. FINAL CROSS-CATEGORY SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("13. BEST MODEL PER CATEGORY")
print("=" * 80)

summary_rows = [
    {"category": "Machine Learning", **{k: v for k, v in best_ml_result.items() if k not in ("model_object", "predictions")}},
    {"category": "Time Series", **ts_results[best_ts_key]},
    {"category": "Deep Learning", **dl_results[best_dl_name]},
]

summary_df = pd.DataFrame(summary_rows).sort_values("RMSE").reset_index(drop=True)

print(summary_df)

overall_best_category = summary_df.iloc[0]["category"]
overall_best_model = summary_df.iloc[0]["model"]

print(
    f"\nOverall best model: {overall_best_model} "
    f"({overall_best_category}) — RMSE {summary_df.iloc[0]['RMSE']:.4f}"
)

# ------------------------------------------------------------
# Persist best-of-each-category to the model registry
# ------------------------------------------------------------

registry_snapshot = {
    "version": VERSION,
    "trained_at": datetime.now().isoformat(),
    "hardware": {
        "gpu_available": GPU_AVAILABLE,
        "gpu_devices": [g.name for g in gpus] if gpus else [],
        "xgboost_device": XGB_DEVICE,
        "lightgbm_device": LGBM_DEVICE,
        "catboost_task_type": CATBOOST_TASK_TYPE,
        "n_jobs_cpu": N_JOBS,
    },
    "best_overall": overall_best_model,
    "summary": summary_df.to_dict(orient="records"),
}

with open(REGISTRY_DIR_META / f"summary_{VERSION}.json", "w") as f:
    json.dump(registry_snapshot, f, indent=2, default=str)

with open(REGISTRY_DIR / f"best_ml_model_{VERSION}.pkl", "wb") as f:
    pickle.dump(best_ml_estimator, f)

print(f"\nSaved run summary + best ML model to {REGISTRY_DIR}")