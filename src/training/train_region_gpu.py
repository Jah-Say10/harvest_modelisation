# ============================================================
# python -m training.train_region
#
# GLOBAL REGIONAL MALARIA FORECASTING
#
# GPU-ACCELERATED VERSION
#
# MACHINE LEARNING
#   - RAPIDS cuML Random Forest       -> GPU
#   - XGBoost                         -> GPU
#   - LightGBM                        -> GPU
#   - CatBoost                        -> GPU
#   - RAPIDS cuML SVR                 -> GPU
#
# TIME SERIES
#   - ARIMA
#   - SARIMA
#   - SARIMAX
#   - Holt-Winters / ETS
#   - VAR
#   - Prophet
#
# DEEP LEARNING
#   - RNN
#   - LSTM
#   - GRU
#   - TensorFlow -> GPU
#
# ============================================================


# ============================================================
# 0. GPU ENVIRONMENT
# ============================================================

import os

# Select GPU 0
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# TensorFlow memory growth
os.environ.setdefault(
    "TF_FORCE_GPU_ALLOW_GROWTH",
    "true"
)


# ============================================================
# STANDARD LIBRARY
# ============================================================

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# WARNINGS
# ============================================================

warnings.filterwarnings("ignore")


# ============================================================
# SCIKIT-LEARN
# ============================================================

from sklearn.model_selection import (
    TimeSeriesSplit,
    RandomizedSearchCV
)

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# GPU ML
# ============================================================

# ------------------------------------------------------------
# RAPIDS cuML
# ------------------------------------------------------------

try:

    from cuml.ensemble import (
        RandomForestRegressor as cuRFRegressor
    )

    from cuml.svm import (
        SVR as cuSVR
    )

    CUML_AVAILABLE = True

except ImportError:

    CUML_AVAILABLE = False

    print(
        "\nWARNING: RAPIDS cuML is not installed."
    )

    print(
        "GPU Random Forest and GPU SVR will not be available."
    )


# ------------------------------------------------------------
# XGBoost
# ------------------------------------------------------------

try:

    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True

except ImportError:

    XGBOOST_AVAILABLE = False

    print(
        "\nWARNING: XGBoost is not installed."
    )


# ------------------------------------------------------------
# LightGBM
# ------------------------------------------------------------

try:

    from lightgbm import LGBMRegressor

    LIGHTGBM_AVAILABLE = True

except ImportError:

    LIGHTGBM_AVAILABLE = False

    print(
        "\nWARNING: LightGBM is not installed."
    )


# ------------------------------------------------------------
# CatBoost
# ------------------------------------------------------------

try:

    from catboost import CatBoostRegressor

    CATBOOST_AVAILABLE = True

except ImportError:

    CATBOOST_AVAILABLE = False

    print(
        "\nWARNING: CatBoost is not installed."
    )


# ============================================================
# TIME SERIES
# ============================================================

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.api import VAR


# ============================================================
# TENSORFLOW
# ============================================================

try:

    import tensorflow as tf

    TENSORFLOW_AVAILABLE = True

except ImportError:

    TENSORFLOW_AVAILABLE = False

    print(
        "\nWARNING: TensorFlow is not installed."
    )


# ============================================================
# PROPHET
# ============================================================

try:

    from prophet import Prophet

    PROPHET_AVAILABLE = True

except ImportError:

    PROPHET_AVAILABLE = False

    print(
        "\nWARNING: Prophet is not installed."
    )


# ============================================================
# PROJECT UTILITIES
# ============================================================

from utils.sort_year import sort_and_replace
from utils.save_metadata import save_metadata


# ============================================================
# RANDOM SEEDS
# ============================================================

RANDOM_STATE = 42

np.random.seed(
    RANDOM_STATE
)

if TENSORFLOW_AVAILABLE:

    tf.random.set_seed(
        RANDOM_STATE
    )


# ============================================================
# GPU INFORMATION
# ============================================================

print("\n" + "=" * 80)
print("GPU CONFIGURATION")
print("=" * 80)

if TENSORFLOW_AVAILABLE:

    gpus = tf.config.list_physical_devices(
        "GPU"
    )

    if gpus:

        print(
            f"TensorFlow detected {len(gpus)} GPU(s):"
        )

        for gpu in gpus:

            print(
                "  ",
                gpu
            )

            try:

                tf.config.experimental.set_memory_growth(
                    gpu,
                    True
                )

            except RuntimeError:
                pass

    else:

        print(
            "WARNING: TensorFlow did not detect a GPU."
        )

else:

    print(
        "TensorFlow unavailable."
    )


print(
    f"cuML available      : {CUML_AVAILABLE}"
)

print(
    f"XGBoost available    : {XGBOOST_AVAILABLE}"
)

print(
    f"LightGBM available   : {LIGHTGBM_AVAILABLE}"
)

print(
    f"CatBoost available   : {CATBOOST_AVAILABLE}"
)

print(
    f"Prophet available    : {PROPHET_AVAILABLE}"
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = (
    "../data/features/"
    "Base_MALARIA_CS_CLEAN_2021_2024_with_weather_2.xlsx"
)


REGISTRY_DIR = Path(
    "../models/registry"
)

REGISTRY_DIR.mkdir(
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

SEASONAL_PERIOD = 12

N_LAGS = 12

LAGS = [
    1,
    2,
    3,
    6,
    12
]

ROLLING_WINDOWS = [
    3,
    6,
    12
]

WINDOW = 6

VERSION = "2.0.0-GPU"


# ============================================================
# HYPERPARAMETER SEARCH
# ============================================================

# Number of random configurations
# Keep this reasonable because GPU models are being trained.
N_ITER_RF = 40
N_ITER_XGB = 80
N_ITER_LGBM = 80
N_ITER_CAT = 40
N_ITER_SVR = 40

CV_SPLITS = 5


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n" + "=" * 80)
print("1. LOAD DATA")
print("=" * 80)


data = pd.read_excel(
    DATA_PATH
)


print(
    data.head()
)

print(
    data.info()
)


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = [
    "DRS",
    "PERIODE",
    target
]


missing_columns = [
    col
    for col in required_columns
    if col not in data.columns
]


if missing_columns:

    raise ValueError(
        "Missing required columns: "
        + str(missing_columns)
    )


# ============================================================
# 2. AGGREGATE DATA BY DRS AND MONTH
# ============================================================

print("\n" + "=" * 80)
print("2. AGGREGATE DATA")
print("=" * 80)


agg_dict = {

    target: "sum",

    **{
        col: "mean"
        for col in meteo_vars
        if col in data.columns
    }

}


df = (
    data
    .groupby(
        [
            "DRS",
            "PERIODE"
        ]
    )
    .agg(
        agg_dict
    )
    .reset_index()
)


print(
    df.head()
)

print(
    "Aggregated dataset:",
    df.shape
)

print(
    "\nRegions:"
)

print(
    df["DRS"].unique()
)


df["PERIODE"] = pd.to_datetime(
    df["PERIODE"]
)


df = (
    df
    .sort_values(
        [
            "PERIODE",
            "DRS"
        ]
    )
    .reset_index(
        drop=True
    )
)


print(
    df.head()
)

print(
    df.dtypes
)


# ============================================================
# 3. COMMON TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 80)
print("3. TRAIN / TEST SPLIT")
print("=" * 80)


unique_periods = (
    df["PERIODE"]
    .drop_duplicates()
    .sort_values()
    .reset_index(
        drop=True
    )
)


n_test = max(
    1,
    int(
        len(unique_periods)
        * TEST_RATIO
    )
)


test_periods = (
    unique_periods
    .iloc[-n_test:]
)


train_periods = (
    unique_periods
    .iloc[:-n_test]
)


TRAIN_END = (
    train_periods.max()
)


TEST_START = (
    test_periods.min()
)


TEST_END = (
    test_periods.max()
)


print(
    "Training:",
    train_periods.min(),
    "->",
    TRAIN_END
)

print(
    "Testing :",
    TEST_START,
    "->",
    TEST_END
)


train_df = (
    df[
        df["PERIODE"].isin(
            train_periods
        )
    ]
    .copy()
)


test_df = (
    df[
        df["PERIODE"].isin(
            test_periods
        )
    ]
    .copy()
)


print(
    "Train:",
    train_df.shape
)

print(
    "Test :",
    test_df.shape
)


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 80)
print("4. FEATURE ENGINEERING")
print("=" * 80)


def create_ml_features(
    dataframe
):

    data = (
        dataframe
        .copy()
        .sort_values(
            [
                "DRS",
                "PERIODE"
            ]
        )
    )


    # --------------------------------------------------------
    # Target lags
    # --------------------------------------------------------

    for lag in LAGS:

        data[
            f"target_lag_{lag}"
        ] = (
            data
            .groupby("DRS")[
                target
            ]
            .shift(lag)
        )


    # --------------------------------------------------------
    # Rolling features
    # --------------------------------------------------------

    data[
        "target_roll_mean_3"
    ] = (
        data
        .groupby("DRS")[
            target
        ]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(3)
            .mean()
        )
    )


    data[
        "target_roll_mean_6"
    ] = (
        data
        .groupby("DRS")[
            target
        ]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(6)
            .mean()
        )
    )


    data[
        "target_roll_mean_12"
    ] = (
        data
        .groupby("DRS")[
            target
        ]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(12)
            .mean()
        )
    )


    # --------------------------------------------------------
    # Calendar
    # --------------------------------------------------------

    data["month"] = (
        data["PERIODE"]
        .dt.month
    )


    data["year"] = (
        data["PERIODE"]
        .dt.year
    )


    data["month_sin"] = np.sin(
        2
        * np.pi
        * data["month"]
        / 12
    )


    data["month_cos"] = np.cos(
        2
        * np.pi
        * data["month"]
        / 12
    )


    return data


ml_df = create_ml_features(
    df
)


ml_df = (
    ml_df
    .dropna()
    .copy()
)


print(
    ml_df.head()
)

print(
    "ML dataset:",
    ml_df.shape
)


# ============================================================
# 5. ML TRAIN / TEST
# ============================================================

ml_train = (
    ml_df[
        ml_df["PERIODE"] <= TRAIN_END
    ]
    .copy()
)


ml_test = (
    ml_df[
        ml_df["PERIODE"] >= TEST_START
    ]
    .copy()
)


print(
    "ML train:",
    ml_train.shape
)

print(
    "ML test :",
    ml_test.shape
)


# ============================================================
# FEATURES
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

    "target_roll_mean_12",

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


X_train = (
    ml_train[
        feature_cols
    ]
    .copy()
)


y_train = (
    ml_train[
        target
    ]
    .copy()
)


X_test = (
    ml_test[
        feature_cols
    ]
    .copy()
)


y_test = (
    ml_test[
        target
    ]
    .copy()
)


# ============================================================
# CATEGORICAL / NUMERIC FEATURES
# ============================================================

categorical_features = [
    "DRS"
]


numeric_features = [
    col
    for col in feature_cols
    if col not in categorical_features
]


# ============================================================
# ONE-HOT ENCODING
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
# SVR PREPROCESSOR
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
# METRICS
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
        + np.abs(y_pred)
    )


    mask = (
        denominator != 0
    )


    if not np.any(mask):

        return 0.0


    return (
        100
        * np.mean(
            2
            * np.abs(
                y_pred[mask]
                - y_true[mask]
            )
            / denominator[mask]
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


    mask = (
        y_true != 0
    )


    if np.any(mask):

        mape = (
            100
            * np.mean(
                np.abs(
                    (
                        y_true[mask]
                        - y_pred[mask]
                    )
                    / y_true[mask]
                )
            )
        )

    else:

        mape = np.nan


    accuracy = (
        max(
            0,
            100 - mape
        )
        if not np.isnan(mape)
        else np.nan
    )


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
            accuracy,

        "R2":
            r2_score(
                y_true,
                y_pred
            )

    }


# ============================================================
# MODEL RESULT CONTAINER
# ============================================================

all_model_results = []


# ============================================================
# TIME SERIES CROSS VALIDATION
# ============================================================

tscv = TimeSeriesSplit(
    n_splits=CV_SPLITS
)


# ============================================================
# 6. GPU RANDOM FOREST
# ============================================================

if CUML_AVAILABLE:

    print("\n" + "=" * 80)
    print("RANDOM FOREST - RAPIDS GPU")
    print("=" * 80)


    rf = Pipeline(

        steps=[

            (
                "preprocessor",
                preprocessor
            ),

            (
                "model",

                cuRFRegressor(

                    random_state=RANDOM_STATE,

                    n_streams=4,

                    n_bins=128

                )
            )

        ]

    )


    rf_params = {

        "model__n_estimators": [
            200,
            400,
            600,
            800
        ],

        "model__max_depth": [
            8,
            12,
            16,
            20,
            24,
            None
        ],

        "model__min_samples_split": [
            2,
            5,
            10,
            15
        ],

        "model__min_samples_leaf": [
            1,
            2,
            4,
            8
        ],

        "model__max_features": [
            0.5,
            0.7,
            1.0
        ]

    }


    rf_search = RandomizedSearchCV(

        estimator=rf,

        param_distributions=rf_params,

        n_iter=N_ITER_RF,

        cv=tscv,

        scoring="neg_root_mean_squared_error",

        n_jobs=1,

        random_state=RANDOM_STATE,

        verbose=2

    )


    rf_search.fit(
        X_train,
        y_train
    )


    print(
        "\nBEST RANDOM FOREST PARAMETERS"
    )

    print(
        rf_search.best_params_
    )


    rf_predictions = (
        rf_search
        .best_estimator_
        .predict(X_test)
    )


    rf_predictions = np.maximum(
        np.asarray(
            rf_predictions
        ),
        0
    )


    rf_metrics = calculate_metrics(
        y_test,
        rf_predictions
    )


    print(
        "\nRANDOM FOREST TEST"
    )

    print(
        rf_metrics
    )


    all_model_results.append({

        "model":
            "Random Forest",

        "search":
            rf_search,

        "predictions":
            rf_predictions,

        **rf_metrics

    })


else:

    print(
        "\nSkipping GPU Random Forest."
    )


# ============================================================
# 7. GPU XGBOOST
# ============================================================

if XGBOOST_AVAILABLE:

    print("\n" + "=" * 80)
    print("XGBOOST - CUDA GPU")
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

                    tree_method="hist",

                    device="cuda",

                    random_state=RANDOM_STATE,

                    n_jobs=1

                )

            )

        ]

    )


    xgb_params = {

        "model__n_estimators": [
            200,
            400,
            600,
            800,
            1200
        ],

        "model__max_depth": [
            2,
            3,
            4,
            5,
            6,
            8,
            10
        ],

        "model__learning_rate": [
            0.01,
            0.02,
            0.03,
            0.05,
            0.07,
            0.10
        ],

        "model__subsample": [
            0.7,
            0.8,
            0.9,
            1.0
        ],

        "model__colsample_bytree": [
            0.7,
            0.8,
            0.9,
            1.0
        ],

        "model__min_child_weight": [
            1,
            3,
            5,
            10
        ],

        "model__gamma": [
            0,
            0.1,
            0.3,
            0.5
        ],

        "model__reg_alpha": [
            0,
            0.01,
            0.1,
            1.0
        ],

        "model__reg_lambda": [
            1,
            2,
            5,
            10
        ]

    }


    xgb_search = RandomizedSearchCV(

        estimator=xgb,

        param_distributions=xgb_params,

        n_iter=N_ITER_XGB,

        cv=tscv,

        scoring="neg_root_mean_squared_error",

        n_jobs=1,

        random_state=RANDOM_STATE,

        verbose=2

    )


    xgb_search.fit(
        X_train,
        y_train
    )


    print(
        "\nBEST XGBOOST PARAMETERS"
    )

    print(
        xgb_search.best_params_
    )


    xgb_predictions = (
        xgb_search
        .best_estimator_
        .predict(X_test)
    )


    xgb_predictions = np.maximum(
        np.asarray(
            xgb_predictions
        ),
        0
    )


    xgb_metrics = calculate_metrics(
        y_test,
        xgb_predictions
    )


    print(
        "\nXGBOOST TEST"
    )

    print(
        xgb_metrics
    )


    all_model_results.append({

        "model":
            "XGBoost",

        "search":
            xgb_search,

        "predictions":
            xgb_predictions,

        **xgb_metrics

    })


else:

    print(
        "\nSkipping XGBoost."
    )


# ============================================================
# 8. GPU LIGHTGBM
# ============================================================

if LIGHTGBM_AVAILABLE:

    print("\n" + "=" * 80)
    print("LIGHTGBM - CUDA GPU")
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

                    device_type="cuda",

                    random_state=RANDOM_STATE,

                    verbosity=-1,

                    n_jobs=1,

                    max_bin=63

                )

            )

        ]

    )


    lgbm_params = {

        "model__n_estimators": [
            200,
            400,
            600,
            800,
            1200
        ],

        "model__learning_rate": [
            0.01,
            0.02,
            0.03,
            0.05,
            0.07,
            0.10
        ],

        "model__num_leaves": [
            7,
            15,
            31,
            47,
            63,
            127
        ],

        "model__max_depth": [
            -1,
            3,
            5,
            7,
            10,
            15
        ],

        "model__min_child_samples": [
            5,
            10,
            20,
            30,
            40,
            60
        ],

        "model__subsample": [
            0.7,
            0.8,
            0.9,
            1.0
        ],

        "model__colsample_bytree": [
            0.7,
            0.8,
            0.9,
            1.0
        ],

        "model__reg_alpha": [
            0,
            0.01,
            0.1,
            1.0
        ],

        "model__reg_lambda": [
            0,
            0.1,
            1.0,
            5.0,
            10.0
        ]

    }


    lgbm_search = RandomizedSearchCV(

        estimator=lgbm,

        param_distributions=lgbm_params,

        n_iter=N_ITER_LGBM,

        cv=tscv,

        scoring="neg_root_mean_squared_error",

        n_jobs=1,

        random_state=RANDOM_STATE,

        verbose=2

    )


    try:

        lgbm_search.fit(
            X_train,
            y_train
        )


        print(
            "\nBEST LIGHTGBM PARAMETERS"
        )

        print(
            lgbm_search.best_params_
        )


        lgbm_predictions = (
            lgbm_search
            .best_estimator_
            .predict(X_test)
        )


        lgbm_predictions = np.maximum(
            np.asarray(
                lgbm_predictions
            ),
            0
        )


        lgbm_metrics = calculate_metrics(
            y_test,
            lgbm_predictions
        )


        print(
            "\nLIGHTGBM TEST"
        )

        print(
            lgbm_metrics
        )


        all_model_results.append({

            "model":
                "LightGBM",

            "search":
                lgbm_search,

            "predictions":
                lgbm_predictions,

            **lgbm_metrics

        })


    except Exception as e:

        print(
            "\nWARNING: LightGBM CUDA training failed."
        )

        print(
            "Reason:",
            repr(e)
        )

        print(
            "Make sure LightGBM was built with CUDA support."
        )


else:

    print(
        "\nSkipping LightGBM."
    )


# ============================================================
# 9. GPU CATBOOST
# ============================================================

if CATBOOST_AVAILABLE:

    print("\n" + "=" * 80)
    print("CATBOOST - GPU")
    print("=" * 80)


    catboost_model = CatBoostRegressor(

        loss_function="RMSE",

        random_seed=RANDOM_STATE,

        task_type="GPU",

        devices="0",

        verbose=False

    )


    catboost_params = {

        "iterations": [
            300,
            500,
            700,
            1000
        ],

        "depth": [
            4,
            6,
            8,
            10
        ],

        "learning_rate": [
            0.02,
            0.03,
            0.05,
            0.07,
            0.10
        ],

        "l2_leaf_reg": [
            1,
            3,
            5,
            10
        ]

    }


    cat_features = [
        X_train.columns.get_loc(
            "DRS"
        )
    ]


    catboost_search = RandomizedSearchCV(

        estimator=catboost_model,

        param_distributions=catboost_params,

        n_iter=N_ITER_CAT,

        cv=tscv,

        scoring="neg_root_mean_squared_error",

        n_jobs=1,

        random_state=RANDOM_STATE,

        verbose=2

    )


    try:

        catboost_search.fit(

            X_train,

            y_train,

            cat_features=cat_features

        )


        print(
            "\nBEST CATBOOST PARAMETERS"
        )

        print(
            catboost_search.best_params_
        )


        cat_predictions = (
            catboost_search
            .best_estimator_
            .predict(X_test)
        )


        cat_predictions = np.maximum(
            np.asarray(
                cat_predictions
            ),
            0
        )


        cat_metrics = calculate_metrics(
            y_test,
            cat_predictions
        )


        print(
            "\nCATBOOST TEST"
        )

        print(
            cat_metrics
        )


        all_model_results.append({

            "model":
                "CatBoost",

            "search":
                catboost_search,

            "predictions":
                cat_predictions,

            **cat_metrics

        })


    except Exception as e:

        print(
            "\nWARNING: CatBoost GPU training failed."
        )

        print(
            "Reason:",
            repr(e)
        )


else:

    print(
        "\nSkipping CatBoost."
    )


# ============================================================
# 10. GPU SVR
# ============================================================

if CUML_AVAILABLE:

    print("\n" + "=" * 80)
    print("SVR - RAPIDS GPU")
    print("=" * 80)


    # --------------------------------------------------------
    # cuML SVR does not use the sklearn Pipeline.
    #
    # We preprocess once and send the resulting matrix
    # to the GPU model.
    # --------------------------------------------------------

    svr_transformer = ColumnTransformer(

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


    X_train_svr = (
        svr_transformer
        .fit_transform(
            X_train
        )
    )


    X_test_svr = (
        svr_transformer
        .transform(
            X_test
        )
    )


    # Convert sparse matrix to dense.
    #
    # cuML SVR expects dense data for ordinary kernels.
    # This is reasonable here because the DRS cardinality
    # is expected to be small.
    #
    if hasattr(
        X_train_svr,
        "toarray"
    ):

        X_train_svr = (
            X_train_svr
            .toarray()
        )


    if hasattr(
        X_test_svr,
        "toarray"
    ):

        X_test_svr = (
            X_test_svr
            .toarray()
        )


    X_train_svr = np.asarray(
        X_train_svr,
        dtype=np.float32
    )


    X_test_svr = np.asarray(
        X_test_svr,
        dtype=np.float32
    )


    svr_model = cuSVR()


    svr_params = {

        "kernel": [
            "rbf",
            "linear",
            "poly",
            "sigmoid"
        ],

        "C": [
            0.1,
            0.5,
            1,
            5,
            10,
            50,
            100,
            500
        ],

        "epsilon": [
            0.001,
            0.01,
            0.05,
            0.1,
            0.2,
            0.3,
            0.5
        ],

        "gamma": [
            "scale",
            "auto",
            0.001,
            0.01,
            0.05,
            0.1
        ],

        "degree": [
            2,
            3,
            4
        ]

    }


    # --------------------------------------------------------
    # cuML SVR is GPU based, but sklearn's RandomizedSearchCV
    # can still orchestrate the parameter search.
    #
    # n_jobs=1 is intentional.
    # --------------------------------------------------------

    svr_search = RandomizedSearchCV(

        estimator=svr_model,

        param_distributions=svr_params,

        n_iter=N_ITER_SVR,

        cv=tscv,

        scoring="neg_root_mean_squared_error",

        n_jobs=1,

        random_state=RANDOM_STATE,

        verbose=2

    )


    try:

        svr_search.fit(
            X_train_svr,
            y_train
        )


        print(
            "\nBEST GPU SVR PARAMETERS"
        )

        print(
            svr_search.best_params_
        )


        svr_predictions = (
            svr_search
            .best_estimator_
            .predict(
                X_test_svr
            )
        )


        svr_predictions = np.maximum(
            np.asarray(
                svr_predictions
            ),
            0
        )


        svr_metrics = calculate_metrics(
            y_test,
            svr_predictions
        )


        print(
            "\nGPU SVR TEST"
        )

        print(
            svr_metrics
        )


        all_model_results.append({

            "model":
                "SVR",

            "search":
                svr_search,

            "predictions":
                svr_predictions,

            "svr_transformer":
                svr_transformer,

            **svr_metrics

        })


    except Exception as e:

        print(
            "\nWARNING: cuML SVR failed."
        )

        print(
            "Reason:",
            repr(e)
        )


else:

    print(
        "\nSkipping GPU SVR."
    )


# ============================================================
# 11. MODEL COMPARISON
# ============================================================

print("\n" + "=" * 80)
print("ML MODEL COMPARISON")
print("=" * 80)


if not all_model_results:

    raise RuntimeError(
        "No ML model was successfully trained."
    )


ml_results_df = pd.DataFrame([

    {

        "model":
            result["model"],

        "MAE":
            result["MAE"],

        "RMSE":
            result["RMSE"],

        "sMAPE":
            result["sMAPE"],

        "MAPE":
            result["MAPE"],

        "Accuracy":
            result["Accuracy"],

        "R2":
            result["R2"]

    }

    for result in all_model_results

])


print(

    ml_results_df
    .sort_values(
        "RMSE"
    )
    .to_string(
        index=False
    )

)


# ============================================================
# 12. GET BEST MODEL
# ============================================================

best_result = min(

    all_model_results,

    key=lambda x: x["RMSE"]

)


best_model_name = (
    best_result["model"]
)


best_model_rmse = (
    best_result["RMSE"]
)


best_search = (
    best_result["search"]
)


best_model = (
    best_search
    .best_estimator_
)


print("\n" + "=" * 80)
print("BEST MODEL")
print("=" * 80)


print(
    "Best ML model:",
    best_model_name
)


print(
    "Test RMSE:",
    f"{best_model_rmse:.4f}"
)


print(
    "Best parameters:"
)


print(
    best_search.best_params_
)


# ============================================================
# 13. PLOT MODEL TEST PREDICTIONS
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


    if save_plot:

        current_date = (
            pd.Timestamp
            .today()
            .strftime(
                "%Y-%m-%d"
            )
        )


        output_dir = (
            REGISTRY_DIR
            / f"Plot_DRS_{current_date}"
        )


        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )


    plot_test = (
        test_df[
            [
                "DRS",
                "PERIODE",
                target
            ]
        ]
        .copy()
    )


    plot_test[
        "prediction"
    ] = np.asarray(
        predictions
    )


    plot_test[
        "PERIODE"
    ] = pd.to_datetime(
        plot_test["PERIODE"]
    )


    plot_test = (
        plot_test
        .sort_values(
            [
                "DRS",
                "PERIODE"
            ]
        )
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
            .sort_values(
                "PERIODE"
            )
        )


        historical[
            "PERIODE"
        ] = pd.to_datetime(
            historical["PERIODE"]
        )


        test_region = (
            plot_test[
                plot_test["DRS"] == region
            ]
            .sort_values(
                "PERIODE"
            )
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
            "PERIODE"
        )


        plt.ylabel(
            target
        )


        plt.grid(
            True,
            alpha=0.25
        )


        plt.legend()


        plt.tight_layout()


        if save_plot:

            safe_model_name = (
                str(model_name)
                .replace("/", "_")
                .replace("\\", "_")
                .replace(" ", "_")
            )


            safe_region = (
                str(region)
                .replace("/", "_")
                .replace("\\", "_")
                .replace(" ", "_")
            )


            filename = (

                f"{safe_model_name}_"
                f"{safe_region}_"
                f"test_predictions.png"

            )


            filepath = (
                output_dir
                / filename
            )


            plt.savefig(

                filepath,

                dpi=300,

                bbox_inches="tight"

            )


        plt.show()

        plt.close()


# ============================================================
# 14. PREPARE FULL DATASET
# ============================================================

ml_full = create_ml_features(
    df
)


ml_full = (

    ml_full
    .dropna()
    .sort_values(
        [
            "DRS",
            "PERIODE"
        ]
    )

)


X_full = (
    ml_full[
        feature_cols
    ]
)


y_full = (
    ml_full[
        target
    ]
)


# ============================================================
# 15. PLOT TEST RESULTS
# ============================================================

best_test_predictions = (
    best_result["predictions"]
)


plot_model_predictions(

    full_df=df,

    test_df=ml_test,

    predictions=best_test_predictions,

    model_name=best_model_name,

    target=target,

    test_start=TEST_START,

    save_plot=True

)


# ============================================================
# 16. REFIT BEST MODEL ON FULL DATA
# ============================================================

print("\n" + "=" * 80)
print("REFIT BEST MODEL ON FULL DATA")
print("=" * 80)


model_final = (
    best_search
    .best_estimator_
)


# ------------------------------------------------------------
# Special case: GPU SVR
# ------------------------------------------------------------

if best_model_name == "SVR":

    full_svr_transformer = ColumnTransformer(

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


    X_full_svr = (
        full_svr_transformer
        .fit_transform(
            X_full
        )
    )


    if hasattr(
        X_full_svr,
        "toarray"
    ):

        X_full_svr = (
            X_full_svr
            .toarray()
        )


    X_full_svr = np.asarray(

        X_full_svr,

        dtype=np.float32

    )


    model_final.fit(

        X_full_svr,

        y_full

    )


else:

    model_final.fit(

        X_full,

        y_full

    )


# ============================================================
# 17. NEXT 12 MONTH FORECAST
# ============================================================

def forecast_next_year_ml(

    model,

    df,

    feature_cols,

    target,

    horizon=12,

    weather_vars=None,

    model_name=None,

    svr_transformer=None

):

    data = (
        df
        .copy()
    )


    data["PERIODE"] = (
        pd.to_datetime(
            data["PERIODE"]
        )
    )


    data = (
        data
        .sort_values(
            [
                "DRS",
                "PERIODE"
            ]
        )
    )


    last_date = (
        data["PERIODE"].max()
    )


    future_dates = pd.date_range(

        start=(
            last_date
            + pd.DateOffset(
                months=1
            )
        ),

        periods=horizon,

        freq="MS"

    )


    all_forecasts = []


    for region in (
        data["DRS"].unique()
    ):


        region_history = (

            data[
                data["DRS"] == region
            ]

            .sort_values(
                "PERIODE"
            )

            .copy()

        )


        # ----------------------------------------------------
        # Last 12 months
        # ----------------------------------------------------

        last_12 = (
            region_history
            .tail(12)
            .copy()
        )


        future = pd.DataFrame({

            "PERIODE":
                future_dates

        })


        future["DRS"] = (
            region
        )


        # ----------------------------------------------------
        # Future weather
        # ----------------------------------------------------

        if weather_vars is not None:

            for col in weather_vars:

                if col in region_history.columns:

                    values = (
                        last_12[col]
                        .to_numpy()
                    )


                    if len(values) > 0:

                        future[col] = np.resize(

                            values,

                            horizon

                        )

                    else:

                        future[col] = 0.0


        # ----------------------------------------------------
        # Add target column
        # ----------------------------------------------------

        future[target] = np.nan


        # ----------------------------------------------------
        # Combine
        # ----------------------------------------------------

        combined = pd.concat(

            [

                region_history,

                future

            ],

            ignore_index=True

        )


        combined = (
            combined
            .sort_values(
                "PERIODE"
            )
            .reset_index(
                drop=True
            )
        )


        # ----------------------------------------------------
        # Recursive forecasting
        # ----------------------------------------------------

        predictions = []


        for step in range(
            horizon
        ):


            future_index = (
                len(region_history)
                + step
            )


            # ------------------------------------------------
            # Recalculate lag features
            # ------------------------------------------------

            for lag in LAGS:

                lag_index = (
                    future_index
                    - lag
                )


                if lag_index >= 0:

                    combined.loc[
                        future_index,
                        f"target_lag_{lag}"
                    ] = (
                        combined.loc[
                            lag_index,
                            target
                        ]
                    )

                else:

                    combined.loc[
                        future_index,
                        f"target_lag_{lag}"
                    ] = np.nan


            # ------------------------------------------------
            # Rolling means
            # ------------------------------------------------

            history_targets = (
                combined.loc[
                    :future_index - 1,
                    target
                ]
                .dropna()
            )


            if len(
                history_targets
            ) >= 3:

                combined.loc[
                    future_index,
                    "target_roll_mean_3"
                ] = (
                    history_targets
                    .tail(3)
                    .mean()
                )

            else:

                combined.loc[
                    future_index,
                    "target_roll_mean_3"
                ] = (
                    history_targets
                    .mean()
                    if len(history_targets)
                    else 0
                )


            if len(
                history_targets
            ) >= 6:

                combined.loc[
                    future_index,
                    "target_roll_mean_6"
                ] = (
                    history_targets
                    .tail(6)
                    .mean()
                )

            else:

                combined.loc[
                    future_index,
                    "target_roll_mean_6"
                ] = (
                    history_targets
                    .mean()
                    if len(history_targets)
                    else 0
                )


            if len(
                history_targets
            ) >= 12:

                combined.loc[
                    future_index,
                    "target_roll_mean_12"
                ] = (
                    history_targets
                    .tail(12)
                    .mean()
                )

            else:

                combined.loc[
                    future_index,
                    "target_roll_mean_12"
                ] = (
                    history_targets
                    .mean()
                    if len(history_targets)
                    else 0
                )


            # ------------------------------------------------
            # Calendar features
            # ------------------------------------------------

            current_date = (
                combined.loc[
                    future_index,
                    "PERIODE"
                ]
            )


            combined.loc[
                future_index,
                "month"
            ] = (
                current_date.month
            )


            combined.loc[
                future_index,
                "year"
            ] = (
                current_date.year
            )


            combined.loc[
                future_index,
                "month_sin"
            ] = np.sin(

                2
                * np.pi
                * current_date.month
                / 12

            )


            combined.loc[
                future_index,
                "month_cos"
            ] = np.cos(

                2
                * np.pi
                * current_date.month
                / 12

            )


            # ------------------------------------------------
            # Prepare feature row
            # ------------------------------------------------

            row = (
                combined
                .loc[
                    [future_index],
                    feature_cols
                ]
                .copy()
            )


            # ------------------------------------------------
            # GPU SVR special handling
            # ------------------------------------------------

            if model_name == "SVR":

                row_transformed = (
                    svr_transformer
                    .transform(
                        row
                    )
                )


                if hasattr(
                    row_transformed,
                    "toarray"
                ):

                    row_transformed = (
                        row_transformed
                        .toarray()
                    )


                row_transformed = np.asarray(

                    row_transformed,

                    dtype=np.float32

                )


                prediction = (
                    model
                    .predict(
                        row_transformed
                    )
                )[0]


            else:

                prediction = (
                    model
                    .predict(
                        row
                    )
                )[0]


            prediction = max(
                0,
                float(prediction)
            )


            predictions.append(
                prediction
            )


            # ------------------------------------------------
            # Feed prediction back
            # ------------------------------------------------

            combined.loc[
                future_index,
                target
            ] = prediction


        # ----------------------------------------------------
        # Store region forecast
        # ----------------------------------------------------

        region_forecast = pd.DataFrame({

            "DRS":
                region,

            "PERIODE":
                future_dates,

            "prediction":
                predictions

        })


        all_forecasts.append(
            region_forecast
        )


    return pd.concat(

        all_forecasts,

        ignore_index=True

    )


# ============================================================
# 18. GENERATE FORECAST
# ============================================================

if best_model_name == "SVR":

    forecast = forecast_next_year_ml(

        model=model_final,

        df=df,

        feature_cols=feature_cols,

        target=target,

        horizon=12,

        weather_vars=meteo_vars,

        model_name="SVR",

        svr_transformer=full_svr_transformer

    )

else:

    forecast = forecast_next_year_ml(

        model=model_final,

        df=df,

        feature_cols=feature_cols,

        target=target,

        horizon=12,

        weather_vars=meteo_vars,

        model_name=best_model_name

    )


print("\n" + "=" * 80)
print("NEXT 12 MONTH FORECAST")
print("=" * 80)


print(
    forecast.head(30)
)


# ============================================================
# 19. PLOT NEXT YEAR FORECAST
# ============================================================

def plot_next_year_forecast(

    historical_df,

    forecast_df,

    model_name,

    target,

    save_plot=False

):


    historical_df = (
        historical_df
        .copy()
    )


    forecast_df = (
        forecast_df
        .copy()
    )


    if save_plot:

        current_date = (
            pd.Timestamp
            .today()
            .strftime(
                "%Y-%m-%d"
            )
        )


        output_dir = (
            REGISTRY_DIR
            / f"Plot_DRS_{current_date}"
        )


        output_dir.mkdir(

            parents=True,

            exist_ok=True

        )


    historical_df[
        "PERIODE"
    ] = pd.to_datetime(

        historical_df[
            "PERIODE"
        ]

    )


    forecast_df[
        "PERIODE"
    ] = pd.to_datetime(

        forecast_df[
            "PERIODE"
        ]

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

            .sort_values(
                "PERIODE"
            )

        )


        forecast_region = (

            forecast_df[
                forecast_df["DRS"] == region
            ]

            .sort_values(
                "PERIODE"
            )

        )


        if forecast_region.empty:

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

            forecast_region["PERIODE"],

            forecast_region["prediction"],

            color="red",

            linewidth=2.5,

            linestyle="--",

            marker="o",

            markersize=5,

            label=(
                f"{model_name} — "
                "Next 12 Months"
            )

        )


        forecast_start = (
            forecast_region[
                "PERIODE"
            ].min()
        )


        forecast_end = (
            forecast_region[
                "PERIODE"
            ].max()
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

            forecast_end,

            color="red",

            alpha=0.06

        )


        plt.title(

            f"{model_name} — "
            f"{region}\n"
            "Historical + Next 12 Months Forecast",

            fontsize=16,

            fontweight="bold"

        )


        plt.xlabel(
            "PERIODE"
        )


        plt.ylabel(
            target
        )


        plt.grid(
            alpha=0.25
        )


        plt.legend()


        plt.tight_layout()


        if save_plot:

            safe_model_name = (

                str(model_name)

                .replace("/", "_")

                .replace("\\", "_")

                .replace(" ", "_")

            )


            safe_region = (

                str(region)

                .replace("/", "_")

                .replace("\\", "_")

                .replace(" ", "_")

            )


            filename = (

                f"{safe_model_name}_"
                f"{safe_region}_"
                "next_year_forecast.png"

            )


            filepath = (
                output_dir
                / filename
            )


            plt.savefig(

                filepath,

                dpi=300,

                bbox_inches="tight"

            )


        plt.show()

        plt.close()


# ============================================================
# 20. PLOT FUTURE FORECAST
# ============================================================

plot_next_year_forecast(

    historical_df=df,

    forecast_df=forecast,

    model_name=best_model_name,

    target=target,

    save_plot=True

)


# ============================================================
# 21. SAVE BEST MODEL
# ============================================================

print("\n" + "=" * 80)
print("SAVE MODEL")
print("=" * 80)


model_path = (
    REGISTRY_DIR
    / "drs_model.pkl"
)


with open(
    model_path,
    "wb"
) as f:

    pickle.dump(
        model_final,
        f
    )


print(
    "Model saved:",
    model_path
)


# ============================================================
# 22. SAVE FEATURE COLUMNS
# ============================================================

feature_path = (
    REGISTRY_DIR
    / "feature_columns_drs.pkl"
)


with open(
    feature_path,
    "wb"
) as f:

    pickle.dump(
        list(feature_cols),
        f
    )


print(
    "Feature columns saved:",
    feature_path
)


# ============================================================
# 23. SAVE HISTORY DATAFRAME
# ============================================================

history_path = (
    REGISTRY_DIR
    / "history_df_drs.pkl"
)


with open(
    history_path,
    "wb"
) as f:

    pickle.dump(
        df,
        f
    )


print(
    "History saved:",
    history_path
)


# ============================================================
# 24. SAVE FORECAST
# ============================================================

forecast_path = (
    REGISTRY_DIR
    / "next_12_months_forecast_drs.csv"
)


forecast.to_csv(

    forecast_path,

    index=False

)


print(
    "Forecast saved:",
    forecast_path
)


# ============================================================
# 25. SAVE MODEL COMPARISON
# ============================================================

comparison_path = (
    REGISTRY_DIR
    / "ml_model_comparison_drs.csv"
)


ml_results_df.sort_values(
    "RMSE"
).to_csv(

    comparison_path,

    index=False

)


print(
    "Comparison saved:",
    comparison_path
)


# ============================================================
# 26. SAVE METADATA
# ============================================================

save_metadata(

    district="DRS",

    algorithm=best_model_name,

    rmse=best_model_rmse,

    training_rows=len(
        X_full
    ),

    version=VERSION,

    best_params=(
        best_search
        .best_params_
    ),

    features=list(
        feature_cols
    )

)


# ============================================================
# 27. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)


print(
    f"Best model       : {best_model_name}"
)


print(
    f"Test RMSE        : {best_model_rmse:.4f}"
)


best_metrics = calculate_metrics(

    y_test,

    best_test_predictions

)


print(
    f"Test MAE         : "
    f"{best_metrics['MAE']:.4f}"
)


print(
    f"Test sMAPE       : "
    f"{best_metrics['sMAPE']:.2f}%"
)


print(
    f"Test MAPE        : "
    f"{best_metrics['MAPE']:.2f}%"
)


print(
    f"Test R2          : "
    f"{best_metrics['R2']:.4f}"
)


print(
    "\nModel:",
    model_path
)


print(
    "Forecast:",
    forecast_path
)


print(
    "Comparison:",
    comparison_path
)


print(
    "\nGPU ML models used:"
)


for result in all_model_results:

    print(
        "  -",
        result["model"]
    )


print(
    "\nDone."
)