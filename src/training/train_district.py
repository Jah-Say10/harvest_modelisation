# ============================================================
# python -m training.train_region
# 
# GLOBAL REGIONAL MALARIA FORECASTING
# MODELS:
#
# 1. MACHINE LEARNING
#    - Random Forest
#    - XGBoost
#    - LightGBM
#    - SVR
#
# 2. TIME SERIES
#    - ARIMA
#    - SARIMA
#    - SARIMAX
#    - Holt-Winters / ETS
#    - VAR
#    - Prophet
#
# 3. DEEP LEARNING
#    - RNN
#    - LSTM
#    - GRU
#
# ============================================================

import pandas as pd
import numpy as np

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

warnings.filterwarnings("ignore")

# ============================================================
# MACHINE LEARNING
# ============================================================

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
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


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# ============================================================
# EXTERNAL ML
# ============================================================

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
# DEEP LEARNING
# ============================================================

# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers

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

# tf.random.set_seed(42)
# np.random.seed(42)

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

# ============================================================
# TARGET
# ============================================================

target = ("Cas Confirmés (par TDR) palu consultations externes")

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
VERSION = "1.0.0"

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
# GLOBAL REGIONAL MALARIA FORECASTING
# DISTRICT-LEVEL MACHINE LEARNING
#
# MODELS:
#   - Random Forest
#   - XGBoost
#   - LightGBM
#   - CatBoost
#   - SVR
#
# GEOGRAPHIC LEVEL:
#   DISTRICT
#
# DRS:
#   Removed from the model
# ============================================================



# ============================================================
# 1. LOAD TRAIN / TEST DATA
# ============================================================

print("\n" + "=" * 80)
print("1. LOAD TRAIN / TEST DATA")
print("=" * 80)

train_raw = pd.read_excel(DATA_PATH)
test_raw = pd.read_excel(DATA_PATH_TEST)

print("Train raw:", train_raw.shape)
print("Test raw :", test_raw.shape)


# ============================================================
# 2. REMOVE DRS
# ============================================================

train_raw = train_raw.drop(
    columns=["DRS"],
    errors="ignore"
)

test_raw = test_raw.drop(
    columns=["DRS"],
    errors="ignore"
)

print("\nDRS removed.")

print("Train shape:", train_raw.shape)
print("Test shape :", test_raw.shape)


# ============================================================
# 3. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "DISTRICT",
    "PERIODE",
    target
]

for col in required_columns:

    if col not in train_raw.columns:
        raise ValueError(
            f"Missing column in training data: {col}"
        )

    if col not in test_raw.columns:
        raise ValueError(
            f"Missing column in test data: {col}"
        )


# ============================================================
# 4. CLEAN DISTRICT
# ============================================================

train_raw["DISTRICT"] = (
    train_raw["DISTRICT"]
    .astype(str)
    .str.strip()
)

test_raw["DISTRICT"] = (
    test_raw["DISTRICT"]
    .astype(str)
    .str.strip()
)


# ============================================================
# 5. CONVERT PERIOD
# ============================================================

train_raw["PERIODE"] = pd.to_datetime(
    train_raw["PERIODE"]
)

test_raw["PERIODE"] = pd.to_datetime(
    test_raw["PERIODE"]
)


# ============================================================
# 6. SORT BY DISTRICT + TIME
# ============================================================

train_raw = (
    train_raw
    .sort_values(
        ["DISTRICT", "PERIODE"]
    )
    .reset_index(drop=True)
)

test_raw = (
    test_raw
    .sort_values(
        ["DISTRICT", "PERIODE"]
    )
    .reset_index(drop=True)
)


# ============================================================
# 7. INFORMATION
# ============================================================

print("\nTrain periods:")
print(
    train_raw["PERIODE"]
    .sort_values()
    .unique()
)

print("\nTest periods:")
print(
    test_raw["PERIODE"]
    .sort_values()
    .unique()
)

print("\nNumber of districts:")
print(
    train_raw["DISTRICT"].nunique()
)

print("\nDistricts:")
print(
    train_raw["DISTRICT"]
    .unique()
)


# ============================================================
# 8. COMBINE DATA FOR FEATURE ENGINEERING
#
# Important:
# Lag features are created within DISTRICT.
#
# Training and test remain separated later.
# ============================================================

df = pd.concat(
    [
        train_raw,
        test_raw
    ],
    ignore_index=True
)

df = (
    df
    .sort_values(
        ["DISTRICT", "PERIODE"]
    )
    .reset_index(drop=True)
)


# ============================================================
# 9. LAG CONFIGURATION
# ============================================================

LAGS = [
    1,
    2,
    3,
    6,
    12
]


# ============================================================
# 10. CREATE ML FEATURES
# ============================================================

def create_ml_features(data):

    data = data.copy()

    data["PERIODE"] = pd.to_datetime(
        data["PERIODE"]
    )

    data = (
        data
        .sort_values(
            ["DISTRICT", "PERIODE"]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Target lags
    #
    # IMPORTANT:
    # Each district has its own time series.
    # --------------------------------------------------------

    for lag in LAGS:

        data[f"target_lag_{lag}"] = (
            data
            .groupby("DISTRICT")[target]
            .shift(lag)
        )

    # --------------------------------------------------------
    # Rolling means
    #
    # shift(1) prevents using the current target.
    # --------------------------------------------------------

    data["target_roll_mean_3"] = (
        data
        .groupby("DISTRICT")[target]
        .transform(
            lambda x:
            x.shift(1)
             .rolling(3)
             .mean()
        )
    )

    data["target_roll_mean_6"] = (
        data
        .groupby("DISTRICT")[target]
        .transform(
            lambda x:
            x.shift(1)
             .rolling(6)
             .mean()
        )
    )

    # --------------------------------------------------------
    # Calendar features
    # --------------------------------------------------------

    data["month"] = (
        data["PERIODE"].dt.month
    )

    data["year"] = (
        data["PERIODE"].dt.year
    )

    data["month_sin"] = np.sin(
        2 * np.pi *
        data["month"] / 12
    )

    data["month_cos"] = np.cos(
        2 * np.pi *
        data["month"] / 12
    )

    return data


# ============================================================
# 11. CREATE FEATURES
# ============================================================

ml_df = create_ml_features(df)

print("\nML dataframe:")
print(ml_df.shape)


# ============================================================
# 12. SEPARATE TRAIN / TEST
#
# IMPORTANT:
# Test data remains the externally defined 2025 dataset.
# ============================================================

train_periods = set(
    train_raw["PERIODE"]
)

test_periods = set(
    test_raw["PERIODE"]
)

ml_train = (
    ml_df[
        ml_df["PERIODE"].isin(train_periods)
    ]
    .copy()
)

ml_test = (
    ml_df[
        ml_df["PERIODE"].isin(test_periods)
    ]
    .copy()
)


# ============================================================
# 13. REMOVE ROWS WITH MISSING LAG FEATURES
# ============================================================

ml_train = (
    ml_train
    .dropna(subset=[
        target,
        "target_lag_1",
        "target_lag_2",
        "target_lag_3",
        "target_lag_6",
        "target_lag_12",
        "target_roll_mean_3",
        "target_roll_mean_6"
    ])
    .copy()
)

ml_test = (
    ml_test
    .dropna(subset=[
        target,
        "target_lag_1",
        "target_lag_2",
        "target_lag_3",
        "target_lag_6",
        "target_lag_12",
        "target_roll_mean_3",
        "target_roll_mean_6"
    ])
    .copy()
)


# ============================================================
# 14. DEFINE ML FEATURES
# ============================================================

feature_cols = [

    # Geographic variable
    "DISTRICT",

    # Weather
    "temperature_moyenne",
    "temperature_max",
    "temperature_min",
    "precipitation",
    "humidite",
    "vent",
    "rayonnement_solaire",

    # Target lags
    "target_lag_1",
    "target_lag_2",
    "target_lag_3",
    "target_lag_6",
    "target_lag_12",

    # Rolling statistics
    "target_roll_mean_3",
    "target_roll_mean_6",

    # Calendar
    "month",
    "year",
    "month_sin",
    "month_cos"
]


# Keep only columns actually available
feature_cols = [
    col
    for col in feature_cols
    if col in ml_df.columns
]

print("\nML features:")
print(feature_cols)


# ============================================================
# 15. X / Y
# ============================================================

X_train = ml_train[
    feature_cols
]

y_train = ml_train[
    target
]

X_test = ml_test[
    feature_cols
]

y_test = ml_test[
    target
]

print("\nX_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_test :", X_test.shape)
print("y_test :", y_test.shape)


# ============================================================
# 16. CATEGORICAL / NUMERICAL FEATURES
# ============================================================

categorical_features = [
    "DISTRICT"
]

numeric_features = [
    col
    for col in feature_cols
    if col not in categorical_features
]


# ============================================================
# 17. STANDARD PREPROCESSOR
#
# Used by:
#   Random Forest
#   XGBoost
#   LightGBM
#
# DISTRICT -> One Hot Encoding
# Numerical variables -> passthrough
# ============================================================

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


# ============================================================
# 18. SVR PREPROCESSOR
#
# SVR requires scaled numerical variables.
# ============================================================

svr_preprocessor = ColumnTransformer(
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
            StandardScaler(),
            numeric_features
        )
    ]
)


# ============================================================
# 19. EVALUATION FUNCTIONS
# ============================================================

def smape(
    y_true,
    y_pred
):

    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )

    denominator = (
        np.abs(y_true)
        +
        np.abs(y_pred)
    )

    mask = denominator != 0

    if not np.any(mask):
        return 0.0

    return (
        100
        *
        np.mean(
            2
            *
            np.abs(
                y_pred[mask]
                -
                y_true[mask]
            )
            /
            denominator[mask]
        )
    )


def calculate_metrics(
    y_true,
    y_pred
):

    y_true = np.asarray(
        y_true
    )

    y_pred = np.maximum(
        np.asarray(y_pred),
        0
    )

    # MAPE
    mask = y_true != 0

    if np.any(mask):

        mape = (
            100
            *
            np.mean(
                np.abs(
                    (
                        y_true[mask]
                        -
                        y_pred[mask]
                    )
                    /
                    y_true[mask]
                )
            )
        )

    else:

        mape = 0.0

    return {

        "MAE":
            mean_absolute_error(
                y_true,
                y_pred
            ),

        "RMSE":
            np.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred
                )
            ),

        "sMAPE":
            smape(
                y_true,
                y_pred
            ),

        "MAPE":
            mape,

        "Accuracy":
            max(
                0,
                100 - mape
            ),

        "R2":
            r2_score(
                y_true,
                y_pred
            )
    }


# ============================================================
# 20. TIME SERIES CROSS VALIDATION
#
# NOTE:
# The rows must already be sorted chronologically.
# ============================================================

tscv = TimeSeriesSplit(
    n_splits=5
)

# ============================================================
# 21. RANDOM FOREST
# ============================================================

print("\n" + "=" * 80)
print("RANDOM FOREST")
print("=" * 80)

rf = Pipeline(
    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            RandomForestRegressor(
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ]
)

# SMALL GRID FOR TESTING
rf_params = {

    "model__n_estimators": [
        200,
        500
    ],

    "model__max_depth": [
        None,
        10,
        20
    ],

    "model__min_samples_split": [
        2,
        5
    ],

    "model__min_samples_leaf": [
        1,
        2
    ],

    "model__max_features": [
        "sqrt",
        0.7
    ]
}

rf_search = GridSearchCV(
    estimator=rf,
    param_grid=rf_params,
    cv=tscv,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    verbose=1
)

rf_search.fit(
    X_train,
    y_train
)

print("\nBEST RANDOM FOREST PARAMETERS")
print(rf_search.best_params_)

rf_predictions = np.maximum(
    rf_search.predict(X_test),
    0
)

rf_metrics = calculate_metrics(
    y_test,
    rf_predictions
)

print("\nRANDOM FOREST TEST")

for key, value in rf_metrics.items():
    print(f"{key}: {value:.4f}")


# ============================================================
# 22. XGBOOST
# ============================================================

print("\n" + "=" * 80)
print("XGBOOST")
print("=" * 80)

xgb = Pipeline(
    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            XGBRegressor(
                objective="reg:squarederror",
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ]
)

# SMALL GRID FOR TESTING
xgb_params = {

    "model__n_estimators": [
        200,
        500
    ],

    "model__max_depth": [
        3,
        6
    ],

    "model__learning_rate": [
        0.05,
        0.1
    ],

    "model__subsample": [
        0.8,
        1.0
    ],

    "model__colsample_bytree": [
        0.8,
        1.0
    ]
}

xgb_search = GridSearchCV(
    xgb,
    xgb_params,
    cv=tscv,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    verbose=1
)

xgb_search.fit(
    X_train,
    y_train
)

print("\nBEST XGBOOST PARAMETERS")
print(xgb_search.best_params_)

xgb_predictions = np.maximum(
    xgb_search.predict(X_test),
    0
)

xgb_metrics = calculate_metrics(
    y_test,
    xgb_predictions
)

print("\nXGBOOST TEST")

for key, value in xgb_metrics.items():
    print(f"{key}: {value:.4f}")


# ============================================================
# 23. LIGHTGBM
# ============================================================

print("\n" + "=" * 80)
print("LIGHTGBM")
print("=" * 80)

lgbm = Pipeline(
    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            LGBMRegressor(
                objective="regression",
                random_state=RANDOM_STATE,
                verbosity=-1,
                n_jobs=-1
            )
        )
    ]
)

# SMALL GRID FOR TESTING
lgbm_params = {

    "model__n_estimators": [
        200,
        500
    ],

    "model__learning_rate": [
        0.05,
        0.1
    ],

    "model__num_leaves": [
        15,
        31
    ],

    "model__max_depth": [
        -1,
        7
    ],

    "model__min_child_samples": [
        10,
        20
    ]
}

lgbm_search = GridSearchCV(
    lgbm,
    lgbm_params,
    cv=tscv,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    verbose=1
)

lgbm_search.fit(
    X_train,
    y_train
)

print("\nBEST LIGHTGBM PARAMETERS")
print(lgbm_search.best_params_)

lgbm_predictions = np.maximum(
    lgbm_search.predict(X_test),
    0
)

lgbm_metrics = calculate_metrics(
    y_test,
    lgbm_predictions
)

print("\nLIGHTGBM TEST")

for key, value in lgbm_metrics.items():
    print(f"{key}: {value:.4f}")


# ============================================================
# 24. CATBOOST
#
# CatBoost handles DISTRICT directly.
# ============================================================

# print("\n" + "=" * 80)
# print("CATBOOST")
# print("=" * 80)

# catboost_model = CatBoostRegressor(
#     loss_function="RMSE",
#     random_seed=RANDOM_STATE,
#     verbose=False
# )

# # SMALL GRID FOR TESTING
# catboost_params = {

#     "iterations": [
#         300,
#         600
#     ],

#     "depth": [
#         4,
#         6
#     ],

#     "learning_rate": [
#         0.05,
#         0.1
#     ],

#     "l2_leaf_reg": [
#         3,
#         5
#     ]
# }

# catboost_search = GridSearchCV(
#     catboost_model,
#     catboost_params,
#     cv=tscv,
#     scoring="neg_root_mean_squared_error",
#     n_jobs=-1,
#     verbose=1
# )

# catboost_search.fit(
#     X_train,
#     y_train,
#     cat_features=[
#         "DISTRICT"
#     ]
# )

# print("\nBEST CATBOOST PARAMETERS")
# print(catboost_search.best_params_)

# cat_predictions = np.maximum(
#     catboost_search.predict(X_test),
#     0
# )

# cat_metrics = calculate_metrics(
#     y_test,
#     cat_predictions
# )

# print("\nCATBOOST TEST")

# for key, value in cat_metrics.items():
#     print(f"{key}: {value:.4f}")


# ============================================================
# 25. SVR
# ============================================================

print("\n" + "=" * 80)
print("SVR")
print("=" * 80)

svr = Pipeline(
    steps=[

        (
            "preprocessor",
            svr_preprocessor
        ),

        (
            "model",
            SVR()
        )
    ]
)

# SMALL GRID FOR TESTING
svr_params = {

    "model__kernel": [
        "rbf",
        "linear"
    ],

    "model__C": [
        1,
        10,
        100
    ],

    "model__epsilon": [
        0.1,
        0.2
    ],

    "model__gamma": [
        "scale",
        "auto"
    ]
}

svr_search = GridSearchCV(
    svr,
    svr_params,
    cv=tscv,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    verbose=1
)

svr_search.fit(
    X_train,
    y_train
)

print("\nBEST SVR PARAMETERS")
print(svr_search.best_params_)

svr_predictions = np.maximum(
    svr_search.predict(X_test),
    0
)

svr_metrics = calculate_metrics(
    y_test,
    svr_predictions
)

print("\nSVR TEST")

for key, value in svr_metrics.items():
    print(f"{key}: {value:.4f}")


# ============================================================
# 26. MODEL COMPARISON
# ============================================================

print("\n" + "=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

model_results = [

    {
        "model": "Random Forest",
        **rf_metrics,
        "model_object":
            rf_search.best_estimator_,
        "search":
            rf_search,
        "predictions":
            rf_predictions
    },

    {
        "model": "XGBoost",
        **xgb_metrics,
        "model_object":
            xgb_search.best_estimator_,
        "search":
            xgb_search,
        "predictions":
            xgb_predictions
    },

    {
        "model": "LightGBM",
        **lgbm_metrics,
        "model_object":
            lgbm_search.best_estimator_,
        "search":
            lgbm_search,
        "predictions":
            lgbm_predictions
    },

    # {
    #     "model": "CatBoost",
    #     **cat_metrics,
    #     "model_object":
    #         catboost_search.best_estimator_,
    #     "search":
    #         catboost_search,
    #     "predictions":
    #         cat_predictions
    # },

    {
        "model": "SVR",
        **svr_metrics,
        "model_object":
            svr_search.best_estimator_,
        "search":
            svr_search,
        "predictions":
            svr_predictions
    }
]


ml_results_df = pd.DataFrame([

    {
        "model": result["model"],
        "MAE": result["MAE"],
        "RMSE": result["RMSE"],
        "sMAPE": result["sMAPE"],
        "MAPE": result["MAPE"],
        "Accuracy": result["Accuracy"],
        "R2": result["R2"]
    }

    for result in model_results
])


print(
    ml_results_df
    .sort_values("RMSE")
    .to_string(index=False)
)


# ============================================================
# 27. SELECT BEST MODEL
# ============================================================

best_model_result = (
    ml_results_df
    .loc[
        ml_results_df["RMSE"].idxmin()
    ]
)

best_model_name = (
    best_model_result["model"]
)

best_model_rmse = (
    best_model_result["RMSE"]
)

print("\n" + "=" * 80)
print("BEST MODEL")
print("=" * 80)

print(
    f"Best ML model: {best_model_name}"
)

print(
    f"Test RMSE: {best_model_rmse:.4f}"
)


# ============================================================
# 28. GET BEST MODEL OBJECT
# ============================================================

best_result = next(
    result
    for result in model_results
    if result["model"] == best_model_name
)

best_model = (
    best_result["model_object"]
)

best_search = (
    best_result["search"]
)

best_predictions = (
    best_result["predictions"]
)


# ============================================================
# 29. DISTRICT TEST PREDICTION PLOT
# ============================================================

def plot_model_predictions(
    full_df,
    test_df,
    predictions,
    model_name,
    target,
    test_start,
    save_plot=False
):

    full_df = full_df.copy()
    test_df = test_df.copy()

    full_df["PERIODE"] = pd.to_datetime(
        full_df["PERIODE"]
    )

    test_df["PERIODE"] = pd.to_datetime(
        test_df["PERIODE"]
    )

    if save_plot:

        current_date = (
            pd.Timestamp.today()
            .strftime("%Y-%m-%d")
        )

        output_dir = (
            Path("../models/registry")
            /
            f"Plot_District_{current_date}"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    plot_test = test_df[
        [
            "DISTRICT",
            "PERIODE",
            target
        ]
    ].copy()

    plot_test["prediction"] = (
        np.asarray(predictions)
    )

    plot_test = (
        plot_test
        .sort_values(
            ["DISTRICT", "PERIODE"]
        )
    )

    districts = (
        plot_test["DISTRICT"]
        .dropna()
        .unique()
    )

    for district in districts:

        historical = (
            full_df[
                full_df["DISTRICT"] == district
            ]
            .sort_values("PERIODE")
        )

        test_district = (
            plot_test[
                plot_test["DISTRICT"] == district
            ]
            .sort_values("PERIODE")
        )

        if test_district.empty:
            continue

        plt.figure(
            figsize=(14, 7)
        )

        # Historical
        plt.plot(
            historical["PERIODE"],
            historical[target],
            color="black",
            linewidth=2,
            label="Historical"
        )

        # Test actual
        plt.plot(
            test_district["PERIODE"],
            test_district[target],
            color="royalblue",
            linewidth=2.5,
            marker="o",
            markersize=5,
            label="Test actual"
        )

        # Prediction
        plt.plot(
            test_district["PERIODE"],
            test_district["prediction"],
            color="red",
            linewidth=2.5,
            linestyle="--",
            marker="o",
            markersize=5,
            label=f"{model_name} prediction"
        )

        # Test boundary
        plt.axvline(
            test_start,
            color="gray",
            linestyle=":",
            linewidth=2,
            label="Test start"
        )

        plt.axvspan(
            test_start,
            test_district["PERIODE"].max(),
            color="orange",
            alpha=0.08
        )

        plt.title(
            f"{model_name} — {district}",
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
            loc="best"
        )

        plt.tight_layout()

        if save_plot:

            safe_model_name = (
                str(model_name)
                .replace("/", "_")
                .replace("\\", "_")
                .replace(" ", "_")
            )

            safe_district = (
                str(district)
                .replace("/", "_")
                .replace("\\", "_")
                .replace(" ", "_")
            )

            filepath = (
                output_dir
                /
                f"{safe_model_name}_"
                f"{safe_district}_"
                f"test_predictions.png"
            )

            plt.savefig(
                filepath,
                dpi=300,
                bbox_inches="tight"
            )

        plt.show()

        plt.close()


# ============================================================
# 30. PLOT BEST MODEL TEST RESULTS
# ============================================================

plot_model_predictions(
    full_df=df,
    test_df=ml_test,
    predictions=best_predictions,
    model_name=best_model_name,
    target=target,
    test_start=ml_test["PERIODE"].min(),
    save_plot=True
)


# ============================================================
# 31. FORECAST NEXT YEAR
# ============================================================

def forecast_next_year_ml(
    model,
    df,
    feature_cols,
    target,
    horizon=12,
    weather_vars=None
):

    data = df.copy()

    data["PERIODE"] = pd.to_datetime(
        data["PERIODE"]
    )

    data = (
        data
        .sort_values(
            ["DISTRICT", "PERIODE"]
        )
    )

    last_date = (
        data["PERIODE"].max()
    )

    future_dates = pd.date_range(
        start=
            last_date
            +
            pd.DateOffset(months=1),
        periods=horizon,
        freq="MS"
    )

    all_forecasts = []

    # --------------------------------------------------------
    # One recursive forecast for each DISTRICT
    # --------------------------------------------------------

    for district in data["DISTRICT"].dropna().unique():

        district_history = (
            data[
                data["DISTRICT"] == district
            ]
            .sort_values("PERIODE")
            .copy()
        )

        if district_history.empty:
            continue

        last_12 = (
            district_history
            .tail(12)
            .copy()
        )

        # ----------------------------------------------------
        # Future dataframe
        # ----------------------------------------------------

        future = pd.DataFrame({
            "PERIODE": future_dates
        })

        future["DISTRICT"] = district

        # ----------------------------------------------------
        # Future weather
        #
        # Repeat last 12 months
        # ----------------------------------------------------

        if weather_vars is not None:

            for col in weather_vars:

                if col in district_history.columns:

                    values = (
                        last_12[col]
                        .to_numpy()
                    )

                    future[col] = np.resize(
                        values,
                        horizon
                    )

        # ----------------------------------------------------
        # Combine history + future
        # ----------------------------------------------------

        combined = pd.concat(
            [
                district_history,
                future
            ],
            ignore_index=True
        )

        combined = (
            combined
            .sort_values("PERIODE")
            .reset_index(drop=True)
        )

        # ----------------------------------------------------
        # Recursive forecasting
        # ----------------------------------------------------

        predictions = []

        for step in range(horizon):

            future_idx = (
                len(district_history)
                + step
            )

            # ----------------------------------------------
            # Recalculate lags
            # ----------------------------------------------

            for lag in LAGS:

                combined[
                    f"target_lag_{lag}"
                ] = (
                    combined[target]
                    .shift(lag)
                )

            # ----------------------------------------------
            # Rolling features
            # ----------------------------------------------

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

            # ----------------------------------------------
            # Calendar features
            # ----------------------------------------------

            combined["month"] = (
                combined["PERIODE"]
                .dt.month
            )

            combined["year"] = (
                combined["PERIODE"]
                .dt.year
            )

            combined["month_sin"] = np.sin(
                2
                * np.pi
                * combined["month"]
                / 12
            )

            combined["month_cos"] = np.cos(
                2
                * np.pi
                * combined["month"]
                / 12
            )

            # ----------------------------------------------
            # Input row
            # ----------------------------------------------

            row = combined.loc[
                [future_idx],
                feature_cols
            ]

            # ----------------------------------------------
            # Prediction
            # ----------------------------------------------

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

            # ----------------------------------------------
            # Put prediction back into target
            # ----------------------------------------------

            combined.loc[
                future_idx,
                target
            ] = prediction

        # ----------------------------------------------------
        # Store district forecast
        # ----------------------------------------------------

        district_forecast = pd.DataFrame({

            "DISTRICT":
                district,

            "PERIODE":
                future_dates,

            "prediction":
                predictions
        })

        all_forecasts.append(
            district_forecast
        )

    if not all_forecasts:

        return pd.DataFrame(
            columns=[
                "DISTRICT",
                "PERIODE",
                "prediction"
            ]
        )

    return pd.concat(
        all_forecasts,
        ignore_index=True
    )


# ============================================================
# 32. REFIT BEST MODEL ON ALL AVAILABLE HISTORICAL DATA
# ============================================================

ml_full = create_ml_features(
    df
)

ml_full = (
    ml_full
    .dropna(
        subset=[
            target,
            "target_lag_1",
            "target_lag_2",
            "target_lag_3",
            "target_lag_6",
            "target_lag_12",
            "target_roll_mean_3",
            "target_roll_mean_6"
        ]
    )
    .sort_values(
        ["DISTRICT", "PERIODE"]
    )
)

X_full = ml_full[
    feature_cols
]

y_full = ml_full[
    target
]


# ------------------------------------------------------------
# IMPORTANT FOR CATBOOST
# ------------------------------------------------------------

if best_model_name == "CatBoost":

    model_final = CatBoostRegressor(
        **best_search.best_params_,
        loss_function="RMSE",
        random_seed=RANDOM_STATE,
        verbose=False
    )

    model_final.fit(
        X_full,
        y_full,
        cat_features=[
            "DISTRICT"
        ]
    )

else:

    model_final = best_search.best_estimator_

    model_final.fit(
        X_full,
        y_full
    )


# ============================================================
# 33. NEXT 12 MONTH FORECAST
# ============================================================

district_forecast = forecast_next_year_ml(
    model=model_final,
    df=df,
    feature_cols=feature_cols,
    target=target,
    horizon=12,
    weather_vars=meteo_vars
)


print("\n" + "=" * 80)
print("NEXT 12 MONTH DISTRICT FORECAST")
print("=" * 80)

print(
    district_forecast.head(20)
)


# ============================================================
# 34. PLOT NEXT YEAR FORECAST
# ============================================================

def plot_next_year_forecast(
    historical_df,
    forecast_df,
    model_name,
    target,
    save_plot=False
):

    historical_df = historical_df.copy()
    forecast_df = forecast_df.copy()

    historical_df["PERIODE"] = (
        pd.to_datetime(
            historical_df["PERIODE"]
        )
    )

    forecast_df["PERIODE"] = (
        pd.to_datetime(
            forecast_df["PERIODE"]
        )
    )

    if save_plot:

        current_date = (
            pd.Timestamp.today()
            .strftime("%Y-%m-%d")
        )

        output_dir = (
            Path("../models/registry")
            /
            f"Plot_District_{current_date}"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    districts = (
        historical_df["DISTRICT"]
        .dropna()
        .unique()
    )

    for district in districts:

        history = (
            historical_df[
                historical_df["DISTRICT"] == district
            ]
            .sort_values("PERIODE")
        )

        forecast = (
            forecast_df[
                forecast_df["DISTRICT"] == district
            ]
            .sort_values("PERIODE")
        )

        if forecast.empty:
            continue

        plt.figure(
            figsize=(15, 7)
        )

        # Historical
        plt.plot(
            history["PERIODE"],
            history[target],
            color="black",
            linewidth=2.5,
            label="Historical"
        )

        # Forecast
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

        forecast_start = (
            forecast["PERIODE"].min()
        )

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
            f"{model_name} — {district}\n"
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

            safe_model_name = (
                str(model_name)
                .replace("/", "_")
                .replace("\\", "_")
                .replace(" ", "_")
            )

            safe_district = (
                str(district)
                .replace("/", "_")
                .replace("\\", "_")
                .replace(" ", "_")
            )

            filename = (
                f"{safe_model_name}_"
                f"{safe_district}_"
                f"next_year_forecast.png"
            )

            filepath = (
                output_dir
                /
                filename
            )

            plt.savefig(
                filepath,
                dpi=300,
                bbox_inches="tight"
            )

        plt.show()

        plt.close()


plot_next_year_forecast(
    historical_df=df,
    forecast_df=district_forecast,
    model_name=best_model_name,
    target=target,
    save_plot=True
)


# ============================================================
# 35. SAVE BEST DISTRICT MODEL
# ============================================================

MODEL_PATH = (
    REGISTRY_DIR
    /
    "district_model.pkl"
)

FEATURE_PATH = (
    REGISTRY_DIR
    /
    "feature_columns_district.pkl"
)

HISTORY_PATH = (
    REGISTRY_DIR
    /
    "history_df_district.pkl"
)

FORECAST_PATH = (
    REGISTRY_DIR
    /
    "district_forecast_next_12_months.pkl"
)


# ------------------------------------------------------------
# Save model
# ------------------------------------------------------------

with open(
    MODEL_PATH,
    "wb"
) as f:

    pickle.dump(
        model_final,
        f
    )


# ------------------------------------------------------------
# Save feature columns
# ------------------------------------------------------------

with open(
    FEATURE_PATH,
    "wb"
) as f:

    pickle.dump(
        feature_cols,
        f
    )


# ------------------------------------------------------------
# Save historical dataframe
# ------------------------------------------------------------

with open(
    HISTORY_PATH,
    "wb"
) as f:

    pickle.dump(
        df,
        f
    )


# ------------------------------------------------------------
# Save forecast
# ------------------------------------------------------------

with open(
    FORECAST_PATH,
    "wb"
) as f:

    pickle.dump(
        district_forecast,
        f
    )


# ============================================================
# 36. SAVE METADATA
# ============================================================

save_metadata(
    district="DISTRICT",
    algorithm=best_model_name,
    rmse=best_model_rmse,
    training_rows=len(X_full),
    version=VERSION,
    best_params=best_search.best_params_,
    features=feature_cols
)


# ============================================================
# 37. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("DISTRICT MODEL TRAINING COMPLETE")
print("=" * 80)

print(
    f"Best model : {best_model_name}"
)

print(
    f"Test RMSE  : {best_model_rmse:.4f}"
)

print(
    f"Districts  : "
    f"{df['DISTRICT'].nunique()}"
)

print(
    f"Features   : "
    f"{len(feature_cols)}"
)

print(
    f"Model saved : {MODEL_PATH}"
)

print(
    f"Forecast saved : {FORECAST_PATH}"
)