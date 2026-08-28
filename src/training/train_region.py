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

import tensorflow as tf
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

    Parameters
    ----------
    full_df : pd.DataFrame
        Complete historical dataframe.

    test_df : pd.DataFrame
        Test dataframe.

    predictions : array-like
        Predictions corresponding to test_df rows.

    model_name : str
        Name of the model.

    target : str
        Target column.

    test_start : datetime
        Beginning of test period.

    save_plot : bool, default=False
        If True, save each DRS plot as a PNG in:
        ../models/registry/Plot_DRS_currentdate/
    """

    # --------------------------------------------------------
    # Prepare output directory
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Prepare test predictions
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # One plot for each DRS
    # --------------------------------------------------------

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

        # ----------------------------------------------------
        # Create figure
        # ----------------------------------------------------

        plt.figure(
            figsize=(14, 7)
        )

        # ----------------------------------------------------
        # Historical values
        # ----------------------------------------------------

        plt.plot(
            historical["PERIODE"],
            historical[target],
            color="black",
            linewidth=2,
            label="Historical"
        )

        # ----------------------------------------------------
        # Test actual values
        # ----------------------------------------------------

        plt.plot(
            test_region["PERIODE"],
            test_region[target],
            color="royalblue",
            linewidth=2.5,
            marker="o",
            markersize=5,
            label="Test actual"
        )

        # ----------------------------------------------------
        # Test predictions
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Test-period boundary
        # ----------------------------------------------------

        plt.axvline(
            test_start,
            color="gray",
            linestyle=":",
            linewidth=2,
            label="Test start"
        )

        # ----------------------------------------------------
        # Shading test period
        # ----------------------------------------------------

        plt.axvspan(
            test_start,
            historical["PERIODE"].max(),
            color="orange",
            alpha=0.08
        )

        # ----------------------------------------------------
        # Titles and labels
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Grid
        # ----------------------------------------------------

        plt.grid(
            True,
            alpha=0.25
        )

        # ----------------------------------------------------
        # Legend
        # ----------------------------------------------------

        plt.legend(
            loc="best",
            frameon=True
        )

        # ----------------------------------------------------
        # Layout
        # ----------------------------------------------------

        plt.tight_layout()

        # ----------------------------------------------------
        # Save plot if requested
        # ----------------------------------------------------

        if save_plot:

            # Replace potentially problematic characters
            # in model/DRS names
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

        plt.show()

        # Close figure to avoid accumulating figures
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

    The model is retrained using all available historical data.

    Future weather is approximated by repeating the
    last 12 months of weather.

    Returns
    -------
    forecast_df : DataFrame
        DRS, PERIODE, prediction
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

        # ----------------------------------------------------
        # Last 12 months
        # ----------------------------------------------------

        last_12 = region_history.tail(12).copy()

        future = pd.DataFrame({
            "PERIODE": future_dates
        })

        future["DRS"] = region

        # ----------------------------------------------------
        # Future weather
        #
        # Repeat the last 12 months as a simple proxy
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Combine history + future
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Create lag features
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Calendar features
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Recursive prediction
        # ----------------------------------------------------

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

            # Malaria indicator cannot be negative
            prediction = max(
                0,
                prediction
            )

            predictions.append(
                prediction
            )

            # Put prediction back into target
            #
            # This is important because the next
            # month's lag features depend on this
            # prediction.
            combined.loc[
                idx,
                target
            ] = prediction

            # Recalculate future rolling features
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

            # Recalculate lags
            for lag in LAGS:

                combined[
                    f"target_lag_{lag}"
                ] = (
                    combined[target]
                    .shift(lag)
                )

        # ----------------------------------------------------
        # Store forecast
        # ----------------------------------------------------

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

    Parameters
    ----------
    historical_df : pd.DataFrame
        Historical dataframe.

    forecast_df : pd.DataFrame
        Forecast dataframe containing:
        DRS, PERIODE and prediction.

    model_name : str
        Name of the model.

    target : str
        Target column.

    save_plot : bool, default=False
        If True, save each DRS plot as a PNG in:
        ../models/registry/Plot_DRS_currentdate/
    """

    historical_df = historical_df.copy()
    forecast_df = forecast_df.copy()

    # --------------------------------------------------------
    # Prepare output directory
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Prepare dates
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # One plot for each DRS
    # --------------------------------------------------------

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

        # Skip DRS with no forecast
        if forecast.empty:
            continue

        # ----------------------------------------------------
        # Plot
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Forecast boundary
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Shade forecast area
        # ----------------------------------------------------

        plt.axvspan(
            forecast_start,
            forecast["PERIODE"].max(),
            color="red",
            alpha=0.06
        )

        # ----------------------------------------------------
        # Title
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Save plot if requested
        # ----------------------------------------------------

        if save_plot:

            # Replace potentially problematic characters
            # in model/DRS names
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

        plt.show()

        # Close figure to avoid accumulating figures
        plt.close()
        
# ============================================================
# PART 2 — CONTINUATION OF train_region.py
#
# This file picks up right after `plot_next_year_forecast(...)`
# is defined in your original script. Paste everything below
# at the end of train_region.py (it reuses target, meteo_vars,
# train_df, test_df, all_df, ml_train, ml_test, X_train, y_train,
# X_test, y_test, feature_cols, preprocessor, evaluate_model,
# plot_model_predictions, smape, LAGS, WINDOW, SEASONAL_PERIOD,
# TRAIN_END, TEST_START, TEST_END, REGISTRY_DIR, RANDOM_STATE,
# PROPHET_AVAILABLE, etc. defined earlier in the file).
#
# ASSUMPTIONS (documented once, so nothing is a silent guess):
#
# 1. "2025 prediction" = the ml_test / test_df period, which is
#    already isolated as the true out-of-sample test set. No
#    separate "next-year" recursive forecast is produced here —
#    `forecast_next_year_ml` / `plot_next_year_forecast` from
#    Part 1 remain available if you want a true 2026 forecast.
#
# 2. Weather (meteo_vars) for 2025 is treated as *known* at
#    prediction time — this matches your ML feature set, which
#    already uses contemporaneous (not lagged) weather. SARIMAX
#    and Prophet regressors reuse actual 2025 weather the same
#    way.
#
# 3. VAR is inherently multivariate, so it is fit ONCE across all
#    DRS regions jointly (each region = one VAR variable), not
#    per-region like the other time-series models.
#
# 4. PERIODE is assumed monthly. Each regional series is coerced
#    to a strict "MS" (month-start) DatetimeIndex before being
#    handed to statsmodels/Prophet.
#
# 5. "Light" grid search = a handful of hand-picked candidate
#    values per hyperparameter, not exhaustive tuning. Good
#    enough to beat an untuned default, not meant to be a final
#    production search.
# ============================================================

import itertools

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.api import VAR

# ============================================================
# SHARED METRIC HELPER (mirrors the metric block inside
# evaluate_model, factored out so time-series / deep-learning
# predictions can be scored the same way without a sklearn-style
# .fit/.predict estimator).
# ============================================================


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
    """
    preds_long_df must have columns: DRS, PERIODE, prediction

    Merges onto the real 2025 test_df so plot_model_predictions
    (already defined in Part 1) can be reused unchanged, no
    matter which model family produced the predictions.
    """

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
# 10. MACHINE LEARNING — LIGHT GRID SEARCH
#     (Random Forest, XGBoost, LightGBM, SVR)
# ============================================================

print("\n" + "=" * 80)
print("10. MACHINE LEARNING MODELS — LIGHT GRID SEARCH")
print("=" * 80)

tscv = TimeSeriesSplit(n_splits=3)

ml_specs = {
    "RandomForest": {
        "estimator": RandomForestRegressor(random_state=RANDOM_STATE),
        "param_grid": {
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 10],
        },
    },
    "XGBoost": {
        "estimator": XGBRegressor(
            random_state=RANDOM_STATE,
            objective="reg:squarederror",
        ),
        "param_grid": {
            "model__n_estimators": [200, 400],
            "model__max_depth": [3, 6],
            "model__learning_rate": [0.05, 0.1],
        },
    },
    "LightGBM": {
        "estimator": LGBMRegressor(random_state=RANDOM_STATE, verbose=-1),
        "param_grid": {
            "model__n_estimators": [200, 400],
            "model__num_leaves": [15, 31],
            "model__learning_rate": [0.05, 0.1],
        },
    },
    "SVR": {
        "estimator": SVR(),
        "param_grid": {
            "model__C": [1, 10],
            "model__epsilon": [0.1, 0.5],
            "model__kernel": ["rbf"],
        },
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

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=spec["param_grid"],
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )

    grid.fit(X_train, y_train)

    print(f"\n{name} best params: {grid.best_params_}")

    result = evaluate_model(
        grid.best_estimator_,
        X_train,
        y_train,
        X_test,
        y_test,
        name,
    )

    ml_results[name] = result
    ml_best_estimators[name] = grid.best_estimator_

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
# ============================================================

print("\n" + "=" * 80)
print("11. TIME SERIES MODELS")
print("=" * 80)

ts_horizon = test_df["PERIODE"].nunique()
regions = sorted(train_df["DRS"].dropna().unique())


def get_region_frame(df, region):
    """
    Monthly-indexed frame (target + weather) for one DRS,
    forced onto a strict month-start frequency.
    """

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
# 11a. ARIMA — light grid over (p, d, q)
# ------------------------------------------------------------

print("\n--- ARIMA ---")

arima_orders = list(itertools.product([0, 1, 2], [0, 1], [0, 1, 2]))
arima_preds = []

for region in regions:

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
        continue

    forecast = np.maximum(best_fit.forecast(steps=ts_horizon).to_numpy(), 0)

    arima_preds.append(
        pd.DataFrame(
            {
                "DRS": region,
                "PERIODE": test_frame.index[:ts_horizon],
                "prediction": forecast,
            }
        )
    )

ts_predictions_long["ARIMA"] = pd.concat(arima_preds, ignore_index=True)
aligned_df, aligned_preds = align_predictions(ts_predictions_long["ARIMA"])
ts_results["ARIMA"] = compute_metrics(aligned_df[target], aligned_preds, "ARIMA")

# ------------------------------------------------------------
# 11b. SARIMA — light grid over seasonal (P, D, Q, 12)
# ------------------------------------------------------------

print("\n--- SARIMA ---")

sarima_base_orders = [(1, 1, 1), (1, 1, 0), (0, 1, 1)]
sarima_seasonal_orders = [
    (P, 1, Q, SEASONAL_PERIOD) for P, Q in itertools.product([0, 1], [0, 1])
]

sarima_preds = []

for region in regions:

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
        continue

    forecast = np.maximum(
        best_fit.forecast(steps=ts_horizon).to_numpy(), 0
    )

    sarima_preds.append(
        pd.DataFrame(
            {
                "DRS": region,
                "PERIODE": test_frame.index[:ts_horizon],
                "prediction": forecast,
            }
        )
    )

ts_predictions_long["SARIMA"] = pd.concat(sarima_preds, ignore_index=True)
aligned_df, aligned_preds = align_predictions(ts_predictions_long["SARIMA"])
ts_results["SARIMA"] = compute_metrics(aligned_df[target], aligned_preds, "SARIMA")

# ------------------------------------------------------------
# 11c. SARIMAX — same light seasonal grid, + weather exog
# ------------------------------------------------------------

print("\n--- SARIMAX ---")

sarimax_preds = []
exog_cols = [c for c in meteo_vars if c in train_df.columns]

if not exog_cols:
    print("SARIMAX skipped: no exogenous columns available.")

else:
    for region in regions:

        train_frame = get_region_frame(train_df, region)
        test_frame = get_region_frame(test_df, region)

        y_hist = train_frame[target]
        exog_hist = train_frame[exog_cols]
        exog_future = test_frame[exog_cols].iloc[:ts_horizon]

        # Validate data
        if y_hist.empty:
            print(f"{region}: empty training data")
            continue

        if len(exog_future) < ts_horizon:
            print(
                f"{region}: insufficient future exogenous data "
                f"({len(exog_future)}/{ts_horizon})"
            )
            continue

        if y_hist.isna().any():
            print(f"{region}: NaNs in target")
            continue

        if exog_hist.isna().any().any():
            print(f"{region}: NaNs in historical exogenous variables")
            continue

        if exog_future.isna().any().any():
            print(f"{region}: NaNs in future exogenous variables")
            continue

        best_aic = np.inf
        best_fit = None

        for order, seasonal_order in itertools.product(
            sarima_base_orders,
            sarima_seasonal_orders
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
                    best_aic = fit.aic
                    best_fit = fit

            except Exception as e:
                print(
                    f"{region} | {order} | {seasonal_order} "
                    f"failed: {e}"
                )

        if best_fit is None:
            print(f"{region}: no valid SARIMAX model")
            continue

        forecast = np.maximum(
            best_fit.forecast(
                steps=ts_horizon,
                exog=exog_future
            ).to_numpy(),
            0
        )

        sarimax_preds.append(
            pd.DataFrame({
                "DRS": region,
                "PERIODE": test_frame.index[:ts_horizon],
                "prediction": forecast,
            })
        )

    # Never concatenate an empty list
    if sarimax_preds:
        ts_predictions_long["SARIMAX"] = pd.concat(
            sarimax_preds,
            ignore_index=True
        )

        aligned_df, aligned_preds = align_predictions(
            ts_predictions_long["SARIMAX"]
        )

        ts_results["SARIMAX"] = compute_metrics(
            aligned_df[target],
            aligned_preds,
            "SARIMAX"
        )

    else:
        print("SARIMAX: no predictions generated.")
        ts_predictions_long["SARIMAX"] = pd.DataFrame(
            columns=["DRS", "PERIODE", "prediction"]
        )

# ------------------------------------------------------------
# 11d. Holt-Winters / ETS — light grid over trend/seasonal
# ------------------------------------------------------------

print("\n--- Holt-Winters / ETS ---")

ets_grid = list(itertools.product([None, "add"], [None, "add", "mul"]))
ets_preds = []

for region in regions:

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
        continue

    forecast = np.maximum(best_fit.forecast(ts_horizon).to_numpy(), 0)

    ets_preds.append(
        pd.DataFrame(
            {
                "DRS": region,
                "PERIODE": test_frame.index[:ts_horizon],
                "prediction": forecast,
            }
        )
    )

ts_predictions_long["ETS"] = pd.concat(ets_preds, ignore_index=True)
aligned_df, aligned_preds = align_predictions(ts_predictions_long["ETS"])
ts_results["ETS"] = compute_metrics(aligned_df[target], aligned_preds, "Holt-Winters / ETS")

# ------------------------------------------------------------
# 11e. VAR — fit ONCE, jointly across all DRS regions
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

var_lags = [1, 2, 3, 6, 12]
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
#      (skipped automatically if prophet isn't installed)
# ------------------------------------------------------------

if PROPHET_AVAILABLE:

    print("\n--- Prophet ---")

    prophet_grid = list(
        itertools.product(
            ["additive", "multiplicative"],
            [0.05, 0.5],
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

                # quick in-sample-style check using train residuals
                # as a light selection criterion across the grid
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
# results dict was keyed by short names (ARIMA, SARIMA, ...) but
# compute_metrics stores the pretty label in "model" — map back:
best_ts_key = [k for k, v in ts_results.items() if v["model"] == best_ts_name][0]

print(f"\nBest time-series model: {best_ts_name}")

plot_best(best_ts_name, ts_predictions_long[best_ts_key], label_prefix="Best TS Model: ")


# ============================================================
# 12. DEEP LEARNING — RNN, LSTM, GRU
# ============================================================

print("\n" + "=" * 80)
print("12. DEEP LEARNING MODELS")
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
from tensorflow.keras.callbacks import EarlyStopping

dl_numeric_cols = [c for c in meteo_vars if c in all_df.columns] + [target]

dl_scaler = StandardScaler()
dl_scaler.fit(all_df.loc[all_df["PERIODE"] <= TRAIN_END, dl_numeric_cols])

dl_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
dl_encoder.fit(all_df[["DRS"]])


def build_sequences(df, window):
    """
    Sliding windows of length `window` over [meteo..., target]
    per DRS, predicting the next month's target. Returns:
      X_seq   : (n_samples, window, n_numeric_features)
      X_static: (n_samples, n_regions)  one-hot DRS
      y       : (n_samples,)
      meta    : DataFrame with DRS, PERIODE of the predicted month
    """

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
        np.asarray(X_seq),
        np.asarray(X_static),
        np.asarray(y),
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


def build_dl_model(recurrent_layer, units, dropout):

    seq_input = Input(shape=(n_timesteps, n_features), name="sequence_input")
    static_input = Input(shape=(n_static,), name="region_input")

    x = recurrent_layer(units)(seq_input)
    x = Dropout(dropout)(x)

    merged = Concatenate()([x, static_input])
    merged = Dense(16, activation="relu")(merged)
    output = Dense(1, activation="linear")(merged)

    model = Model(inputs=[seq_input, static_input], outputs=output)
    model.compile(optimizer="adam", loss="mse")

    return model


dl_specs = {
    "RNN": SimpleRNN,
    "LSTM": LSTM,
    "GRU": GRU,
}

dl_param_grid = list(itertools.product([32, 64], [0.0, 0.2]))  # (units, dropout)

dl_results = {}
dl_predictions_long = {}

for name, layer_cls in dl_specs.items():

    print(f"\n--- {name} ---")

    best_val_loss, best_model = np.inf, None

    for units, dropout in dl_param_grid:

        model = build_dl_model(layer_cls, units, dropout)

        history = model.fit(
            [X_seq_train, X_static_train],
            y_seq_train,
            validation_split=0.15,
            epochs=50,
            batch_size=16,
            shuffle=False,
            callbacks=[
                EarlyStopping(patience=5, restore_best_weights=True)
            ],
            verbose=0,
        )

        val_loss = min(history.history["val_loss"])

        if val_loss < best_val_loss:
            best_val_loss, best_model = val_loss, model

    predictions = np.maximum(
        best_model.predict([X_seq_test, X_static_test]).flatten(), 0
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
#
# NOTE: ML metrics are computed on the tabular ml_test set,
# TS/DL metrics on their own aligned test sets. All three are
# pooled across every DRS/month in 2025, so RMSE/MAE are
# comparable in scale even though the underlying test rows
# aren't 100% identical row-for-row (a handful of TS regions
# can silently drop out if a fit fails).
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
    "best_overall": overall_best_model,
    "summary": summary_df.to_dict(orient="records"),
}

# Save metadata
with open(REGISTRY_DIR_META / f"summary_{VERSION}.json", "w") as f:
    json.dump(registry_snapshot, f, indent=2, default=str)

# Save models
with open(REGISTRY_DIR / f"best_ml_model_{VERSION}.pkl", "wb") as f:
    pickle.dump(best_ml_estimator, f)

print(f"\nSaved run summary + best ML model to {REGISTRY_DIR}")