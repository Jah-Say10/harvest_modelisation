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
#    - CatBoost
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

DATA_PATH = (
    "../data/features/"
    "Base_MALARIA_CS_CLEAN_2021_2024_with_weather_2.xlsx"
)

REGISTRY_DIR = Path("../models/registry")
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
# 1. LOAD DATA
# ============================================================

print("\n" + "=" * 80)
print("1. LOAD DATA")
print("=" * 80)

data = pd.read_excel(DATA_PATH)

print(data.head())
print(data.info())

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
    .groupby(["DRS", "PERIODE"])
    .agg(agg_dict)
    .reset_index()
)

print(df.head())
print("Aggregated dataset:", df.shape)
print("\nRegions:")
print(df["DRS"].unique())

print("\nPeriods:")
print(df["PERIODE"].unique())

df["PERIODE"] = pd.to_datetime(df["PERIODE"])

df = (
    df
    .sort_values(["PERIODE", "DRS"])
    .reset_index(drop=True)
)

print(df.head())
print(df.dtypes)

# ===================================================
# Create one common test period
# ===================================================

print("\n" + "=" * 80)
print("3. TRAIN / TEST SPLIT")
print("=" * 80)

unique_periods = (
    df["PERIODE"]
    .drop_duplicates()
    .sort_values()
    .reset_index(drop=True)
)

TEST_SIZE = 0.20

n_test = max(
    1,
    int(len(unique_periods) * TEST_SIZE)
)

test_periods = unique_periods.iloc[-n_test:]
train_periods = unique_periods.iloc[:-n_test]

TRAIN_END = train_periods.max()
TEST_START = test_periods.min()
TEST_END = test_periods.max()

print("Training:", train_periods.min(), "->", TRAIN_END)
print("Testing :", TEST_START, "->", TEST_END)

train_df = df[df["PERIODE"].isin(train_periods)].copy()
test_df = df[df["PERIODE"].isin(test_periods)].copy()

print("Train:", train_df.shape)
print("Test :", test_df.shape)

# ===================================================
# Create lag features for ML
# ===================================================

LAGS = [1, 2, 3, 6, 12]


def create_ml_features(dataframe):

    data = dataframe.copy()

    data = data.sort_values(
        ["DRS", "PERIODE"]
    )

    # Target lags
    for lag in LAGS:
        data[f"target_lag_{lag}"] = (
            data
            .groupby("DRS")[target]
            .shift(lag)
        )

    # Rolling features
    data["target_roll_mean_3"] = (
        data
        .groupby("DRS")[target]
        .transform(
            lambda x: x.shift(1).rolling(3).mean()
        )
    )

    data["target_roll_mean_6"] = (
        data
        .groupby("DRS")[target]
        .transform(
            lambda x: x.shift(1).rolling(6).mean()
        )
    )

    # Calendar features
    data["month"] = data["PERIODE"].dt.month
    data["year"] = data["PERIODE"].dt.year

    # Cyclic month encoding
    data["month_sin"] = np.sin(
        2 * np.pi * data["month"] / 12
    )

    data["month_cos"] = np.cos(
        2 * np.pi * data["month"] / 12
    )

    return data

ml_df = create_ml_features(df)

ml_df = ml_df.dropna().copy()

print(ml_df.head())
print(ml_df.shape)

ml_train = ml_df[
    ml_df["PERIODE"] <= TRAIN_END
].copy()

ml_test = ml_df[
    ml_df["PERIODE"] >= TEST_START
].copy()

print("ML train:", ml_train.shape)
print("ML test :", ml_test.shape)

# ===================================================
# Define ML features
# ===================================================

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
    col for col in feature_cols
    if col in ml_df.columns
]

X_train = ml_train[feature_cols]
y_train = ml_train[target]

X_test = ml_test[feature_cols]
y_test = ml_test[target]

# ===================================================
# One-hot encode DRS
# ===================================================

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
    
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
        
# ===================================================
# Random Forest
# ===================================================

tscv = TimeSeriesSplit(
    n_splits=5
)

print("\n" + "=" * 80)
print("Random Forest")
print("=" * 80)

rf = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestRegressor(
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

rf_params = {
    "model__n_estimators": [
        200,
        500,
        800,
        1200
    ],

    "model__max_depth": [
        None,
        5,
        10,
        15,
        20,
        30,
        50
    ],

    "model__min_samples_split": [
        2,
        5,
        10,
        15,
        20
    ],

    "model__min_samples_leaf": [
        1,
        2,
        4,
        8,
        12
    ],

    "model__max_features": [
        "sqrt",
        "log2",
        0.3,
        0.5,
        0.7,
        1.0
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

print("Best CV RMSE:", -rf_search.best_score_)

rf_predictions = rf_search.predict(X_test)

rf_predictions = np.maximum(rf_predictions, 0)

rf_mae = mean_absolute_error(y_test, rf_predictions)

rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))

rf_smape = smape(y_test.to_numpy(), rf_predictions)

print("\nRANDOM FOREST TEST")
print("MAE:", rf_mae)
print("RMSE:", rf_rmse)
print("sMAPE:", rf_smape)

# # Plot Test
# plot_model_predictions(
#     full_df=df,
#     test_df=ml_test,
#     predictions=rf_predictions,
#     model_name="Random Forest",
#     target=target,
#     test_start=TEST_START
# )

# # Plot historical and prediction
# ml_full = create_ml_features(df)

# ml_full = (
#     ml_full
#     .dropna()
#     .sort_values(
#         ["DRS", "PERIODE"]
#     )
# )

# X_full = ml_full[feature_cols]
# y_full = ml_full[target]

# rf_final = rf_search.best_estimator_

# rf_final.fit(
#     X_full,
#     y_full
# )

# rf_future = forecast_next_year_ml(
#     model=rf_final,
#     df=df,
#     feature_cols=feature_cols,
#     target=target,
#     horizon=12,
#     weather_vars=meteo_vars
# )

# plot_next_year_forecast(
#     historical_df=df,
#     forecast_df=rf_future,
#     model_name="Random Forest",
#     target=target
# )

# ===================================================
# XGBoost
# ===================================================

print("\n" + "=" * 80)
print("XGBoost")
print("=" * 80)

xgb = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            XGBRegressor(
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

xgb_params = {
    "model__n_estimators": [
        200,
        500,
        800,
        1200
    ],

    "model__max_depth": [
        2,
        3,
        4,
        5,
        6,
        7,
        9,
        12
    ],

    "model__learning_rate": [
        0.01,
        0.02,
        0.03,
        0.05,
        0.07,
        0.1,
        0.15
    ],

    "model__subsample": [
        0.6,
        0.7,
        0.8,
        0.9,
        1.0
    ],

    "model__colsample_bytree": [
        0.6,
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
        0.5,
        1.0
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

xgb_predictions = xgb_search.predict(
    X_test
)

xgb_predictions = np.maximum(
    xgb_predictions,
    0
)

print("\nXGBOOST TEST")

print(
    "MAE:",
    mean_absolute_error(
        y_test,
        xgb_predictions
    )
)

print(
    "RMSE:",
    np.sqrt(
        mean_squared_error(
            y_test,
            xgb_predictions
        )
    )
)

print(
    "sMAPE:",
    smape(
        y_test.to_numpy(),
        xgb_predictions
    )
)

# # Plot Test
# plot_model_predictions(
#     full_df=df,
#     test_df=ml_test,
#     predictions=xgb_predictions,
#     model_name="XGBoost",
#     target=target,
#     test_start=TEST_START
# )

# # Plot historical and prediction
# ml_full = create_ml_features(df)

# ml_full = (
#     ml_full
#     .dropna()
#     .sort_values(
#         ["DRS", "PERIODE"]
#     )
# )

# X_full = ml_full[feature_cols]
# y_full = ml_full[target]

# xgb_final = xgb_search.best_estimator_

# xgb_final.fit(
#     X_full,
#     y_full
# )

# xgb_final = forecast_next_year_ml(
#     model=xgb_final,
#     df=df,
#     feature_cols=feature_cols,
#     target=target,
#     horizon=12,
#     weather_vars=meteo_vars
# )

# plot_next_year_forecast(
#     historical_df=df,
#     forecast_df=xgb_final,
#     model_name="Random Forest",
#     target=target
# )

# ===================================================
# LightGBM
# ===================================================

print("\n" + "=" * 80)
print("LightGBM")
print("=" * 80)

lgbm = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            LGBMRegressor(
                objective="regression",
                random_state=42,
                verbosity=-1,
                n_jobs=-1
            )
        )
    ]
)

lgbm_params = {
    "model__n_estimators": [
        200,
        500,
        800,
        1200
    ],

    "model__learning_rate": [
        0.01,
        0.02,
        0.03,
        0.05,
        0.07,
        0.1,
        0.15
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
        0.6,
        0.7,
        0.8,
        0.9,
        1.0
    ],

    "model__colsample_bytree": [
        0.6,
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

lgbm_predictions = lgbm_search.predict(
    X_test
)

lgbm_predictions = np.maximum(
    lgbm_predictions,
    0
)

print("\nLIGHTGBM TEST")

print(
    "MAE:",
    mean_absolute_error(
        y_test,
        lgbm_predictions
    )
)

print(
    "RMSE:",
    np.sqrt(
        mean_squared_error(
            y_test,
            lgbm_predictions
        )
    )
)

print(
    "sMAPE:",
    smape(
        y_test.to_numpy(),
        lgbm_predictions
    )
)

# # Plot Test
# plot_model_predictions(
#     full_df=df,
#     test_df=ml_test,
#     predictions=lgbm_predictions,
#     model_name="LGBM",
#     target=target,
#     test_start=TEST_START
# )

# # Plot historical and prediction
# ml_full = create_ml_features(df)

# ml_full = (
#     ml_full
#     .dropna()
#     .sort_values(
#         ["DRS", "PERIODE"]
#     )
# )

# X_full = ml_full[feature_cols]
# y_full = ml_full[target]

# lgbm_final = lgbm_search.best_estimator_

# lgbm_final.fit(
#     X_full,
#     y_full
# )

# lgbm_final = forecast_next_year_ml(
#     model=lgbm_final,
#     df=df,
#     feature_cols=feature_cols,
#     target=target,
#     horizon=12,
#     weather_vars=meteo_vars
# )

# plot_next_year_forecast(
#     historical_df=df,
#     forecast_df=lgbm_final,
#     model_name="LGBM",
#     target=target
# )

# ===================================================
# CatBoost
# ===================================================

print("\n" + "=" * 80)
print("CatBoost")
print("=" * 80)

cat_features = [
    X_train.columns.get_loc("DRS")
]

catboost_model = CatBoostRegressor(
    loss_function="RMSE",
    random_seed=42,
    verbose=False
)

catboost_params = {
    "iterations": [
        300,
        600
    ],

    "depth": [
        4,
        6,
        8
    ],

    "learning_rate": [
        0.03,
        0.05,
        0.1
    ],

    "l2_leaf_reg": [
        1,
        3,
        5,
        10
    ]
}

catboost_search = GridSearchCV(
    catboost_model,
    catboost_params,
    cv=tscv,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    verbose=1
)

catboost_search.fit(
    X_train,
    y_train,
    cat_features=cat_features
)

print("\nBEST CATBOOST PARAMETERS")
print(catboost_search.best_params_)

cat_predictions = catboost_search.predict(
    X_test
)

cat_predictions = np.maximum(
    cat_predictions,
    0
)

print("\nCATBOOST TEST")

print(
    "MAE:",
    mean_absolute_error(
        y_test,
        cat_predictions
    )
)

print(
    "RMSE:",
    np.sqrt(
        mean_squared_error(
            y_test,
            cat_predictions
        )
    )
)

print(
    "sMAPE:",
    smape(
        y_test.to_numpy(),
        cat_predictions
    )
)

# # Plot historical and prediction
# ml_full = create_ml_features(df)

# ml_full = (
#     ml_full
#     .dropna()
#     .sort_values(
#         ["DRS", "PERIODE"]
#     )
# )

# X_full = ml_full[feature_cols]
# y_full = ml_full[target]

# cat_final = catboost_search.best_estimator_

# cat_final.fit(
#     X_full,
#     y_full
# )

# cat_final = forecast_next_year_ml(
#     model=cat_final,
#     df=df,
#     feature_cols=feature_cols,
#     target=target,
#     horizon=12,
#     weather_vars=meteo_vars
# )

# plot_next_year_forecast(
#     historical_df=df,
#     forecast_df=cat_final,
#     model_name="CatBoost",
#     target=target
# )

# ===================================================
# SVR
# ===================================================

print("\n" + "=" * 80)
print("SVR")
print("=" * 80)

svr_preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            ["DRS"]
        ),
        (
            "num",
            StandardScaler(),
            numeric_features
        )
    ]
)

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

svr_params = {
    "model__kernel": [
        "rbf",
        "linear",
        "poly",
        "sigmoid"
    ],

    "model__C": [
        0.01,
        0.1,
        0.5,
        1,
        5,
        10,
        50,
        100,
        500,
        1000
    ],

    "model__epsilon": [
        0.001,
        0.01,
        0.05,
        0.1,
        0.2,
        0.3,
        0.5
    ],

    "model__gamma": [
        "scale",
        "auto",
        0.001,
        0.01,
        0.05,
        0.1,
        0.5,
        1
    ],

    "model__degree": [
        2,
        3,
        4,
        5
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

svr_predictions = svr_search.predict(
    X_test
)

svr_predictions = np.maximum(
    svr_predictions,
    0
)

print("\nSVR TEST")

print(
    "MAE:",
    mean_absolute_error(
        y_test,
        svr_predictions
    )
)

print(
    "RMSE:",
    np.sqrt(
        mean_squared_error(
            y_test,
            svr_predictions
        )
    )
)

print(
    "sMAPE:",
    smape(
        y_test.to_numpy(),
        svr_predictions
    )
)

# Plot Test
# plot_model_predictions(
#     full_df=df,
#     test_df=ml_test,
#     predictions=svr_predictions,
#     model_name="SVR",
#     target=target,
#     test_start=TEST_START
# )

# # Plot historical and prediction
# ml_full = create_ml_features(df)

# ml_full = (
#     ml_full
#     .dropna()
#     .sort_values(
#         ["DRS", "PERIODE"]
#     )
# )

# X_full = ml_full[feature_cols]
# y_full = ml_full[target]

# svr_final = svr_search.best_estimator_

# svr_final.fit(
#     X_full,
#     y_full
# )

# svr_final = forecast_next_year_ml(
#     model=svr_final,
#     df=df,
#     feature_cols=feature_cols,
#     target=target,
#     horizon=12,
#     weather_vars=meteo_vars
# )

# plot_next_year_forecast(
#     historical_df=df,
#     forecast_df=svr_final,
#     model_name="SVR",
#     target=target
# )

# ===================================================
# BEST MODEL
# ===================================================

print("\n" + "=" * 80)
print("BEST MODEL")
print("=" * 80)

def calculate_metrics(
    y_true,
    y_pred
):

    y_pred = np.maximum(
        np.asarray(y_pred),
        0
    )

    return {
        "MAE": mean_absolute_error(
            y_true,
            y_pred
        ),

        "RMSE": np.sqrt(
            mean_squared_error(
                y_true,
                y_pred
            )
        ),

        "sMAPE": smape(
            np.asarray(y_true),
            y_pred
        ),

        "R2": r2_score(
            y_true,
            y_pred
        )
    }

ml_results = []

for name, model_search in [
    # ("Random Forest", rf_search),
    ("XGBoost", xgb_search),
    # ("LightGBM", lgbm_search),
    # ("CatBoost", catboost_search),
    # ("SVR", svr_search)
]:

    if name == "CatBoost":
        predictions = model_search.predict(
            X_test
        )
    else:
        predictions = model_search.predict(
            X_test
        )

    predictions = np.maximum(
        predictions,
        0
    )

    metrics = calculate_metrics(
        y_test,
        predictions
    )

    ml_results.append({
        "model": name,
        **metrics,
        "best_params": model_search.best_params_,
        "predictions": predictions,
        "model_object": model_search.best_estimator_
    })

ml_results_df = pd.DataFrame([
    {
        "model": result["model"],
        "MAE": result["MAE"],
        "RMSE": result["RMSE"],
        "sMAPE": result["sMAPE"],
        "R2": result["R2"],
        "model_object": result["model_object"],
    }
    for result in ml_results
])

# print("\n" + "=" * 80)
# print("ML MODEL COMPARISON")
# print("=" * 80)

# print(
#     ml_results_df
#     .sort_values("RMSE")
#     .to_string(index=False)
# )

def best_model(df_result):
    # ============================================================
    # SELECT BEST MODEL
    # ============================================================

    best_model_result = df_result.loc[df_result["RMSE"].idxmin()]

    best_model_name = best_model_result["model"]
    best_model_rmse = best_model_result["RMSE"]

    print(f"Best ML model: {best_model_name}")
    print(f"Test RMSE: {best_model_rmse:.4f}")

    # ============================================================
    # GET BEST TRAINED MODEL
    # ============================================================

    match best_model_name:
        # case "RandomForestRegressor":
        #     search = rf_search
            
        # case "RandomForest":
        #         search = rf_search
                    
        # case "Random Forest":
        #     search = rf_search

        case "XGBoost":
            search = xgb_search

        case "XGBRegressor":
            search = xgb_search

        # case "LightGBM":
        #     search = lgbm_search

        # case "LGBMRegressor":
        #     search = lgbm_search

        # case "CatBoost":
        #     search = catboost_search

        # case "CatBoostRegressor":
        #     search = catboost_search

        # case "SVR":
        #     search = svr_search

        case _:
            raise ValueError(
                f"Unknown model name: {best_model_name}"
            )

    best_model = search.best_estimator_

    # ============================================================
    # PREDICTION
    # ============================================================

    predictions = best_model.predict(X_test)

    # ============================================================
    # PLOT TEST RESULTS
    # ============================================================

    plot_model_predictions(
        full_df=df,
        test_df=ml_test,
        predictions=predictions,
        model_name=best_model_name,
        target=target,
        test_start=TEST_START,
        save_plot=True
    )

    # ============================================================
    # PREPARE FULL DATASET
    # ============================================================

    ml_full = create_ml_features(df)

    ml_full = (
        ml_full
        .dropna()
        .sort_values(["DRS", "PERIODE"])
    )

    X_full = ml_full[feature_cols]
    y_full = ml_full[target]

    # ============================================================
    # REFIT BEST MODEL ON FULL DATA
    # ============================================================

    model_final = search.best_estimator_

    model_final.fit(
        X_full,
        y_full
    )

    # ============================================================
    # FORECAST NEXT YEAR
    # ============================================================

    forecast = forecast_next_year_ml(
        model=model_final,
        df=df,
        feature_cols=feature_cols,
        target=target,
        horizon=12,
        weather_vars=meteo_vars
    )

    plot_next_year_forecast(
        historical_df=df,
        forecast_df=forecast,
        model_name=best_model_name,
        target=target,
        save_plot=True
    )
    
    # ============================================================
    # SAVE BEST MODEL
    # ============================================================
    
    # Save model
    with open("../models/registry/drs_model.pkl", "wb") as f:
        pickle.dump(model_final, f)

    # Save feature names
    with open("../models/registry/feature_columns_drs.pkl", "wb") as f:
        pickle.dump(list(ml_df.columns), f)

    # Save history dataframe
    with open("../models/registry/history_df_drs.pkl", "wb") as f:
        pickle.dump(df, f)
    
    # Save metadata
    save_metadata(
        district="DRS",
        algorithm=best_model_name,
        rmse=best_model_rmse,
        training_rows=len(X_train),
        version=VERSION,
        best_params=search.best_params_,
        features=list(df.columns)
    )

    # ============================================================
    # RETURN
    # ============================================================

    return model_final, forecast
    
best_model(ml_results_df)