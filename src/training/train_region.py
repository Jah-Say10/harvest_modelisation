# ============================================================
# python -m training.train_region
#
# GLOBAL REGIONAL MALARIA FORECASTING
#
# MACHINE LEARNING ONLY
#    - Random Forest
#    - XGBoost
#    - LightGBM
#    - SVR
#
# Workflow:
#    1. Load train/test data
#    2. Aggregate data by DRS/month
#    3. Create ML features
#    4. Split train/test
#    5. Grid search ML models
#    6. Evaluate models on 2025 test set
#    7. Select best ML model by RMSE
#    8. Plot best model predictions
#    9. Save best model + metadata
#
# ============================================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime

import pickle
import json
import warnings

warnings.filterwarnings("ignore")


# ============================================================
# MACHINE LEARNING
# ============================================================

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import (
    TimeSeriesSplit,
    GridSearchCV
)

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.svm import SVR

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


# ============================================================
# PROJECT UTILITIES
# ============================================================

from utils.sort_year import sort_and_replace
from utils.save_metadata import save_metadata


# ============================================================
# RANDOM SEEDS
# ============================================================

np.random.seed(42)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "../data/features/Base_MALARIA_CS_CLEAN_2021_2024_with_weather.xlsx"

DATA_PATH_TEST = "../data/features/Base_MALARIA_CS_CLEAN_2025_with_weather.xlsx"


REGISTRY_DIR = Path("../models/registry")
REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

REGISTRY_DIR_META = Path("../models/metadata")
REGISTRY_DIR_META.mkdir(parents=True, exist_ok=True)


# ============================================================
# TARGET
# ============================================================

target = "Cas Confirmés (par TDR) palu consultations externes"


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

# Number of target lags used as ML features
LAGS = [1, 2, 3, 6, 12]

# Random seed
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


# ------------------------------------------------------------
# Training data: 2021-2024
# ------------------------------------------------------------

train_df = (
    train_raw
    .groupby(["DRS", "PERIODE"])
    .agg(agg_dict)
    .reset_index()
)


# ------------------------------------------------------------
# Test data: 2025
# ------------------------------------------------------------

test_df = (
    test_raw
    .groupby(["DRS", "PERIODE"])
    .agg(agg_dict)
    .reset_index()
)


# ------------------------------------------------------------
# Convert periods to datetime
# ------------------------------------------------------------

train_df["PERIODE"] = pd.to_datetime(train_df["PERIODE"])

test_df["PERIODE"] = pd.to_datetime(test_df["PERIODE"])


# ------------------------------------------------------------
# Sort
# ------------------------------------------------------------

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
#
# We combine 2021-2024 + 2025 BEFORE creating lag features.
#
# This allows:
#
# January 2025 lag_1  -> December 2024
# January 2025 lag_2  -> November 2024
# ...
# January 2025 lag_12 -> January 2024
#
# Therefore, 2025 features use historical 2021-2024
# observations and do not use future 2025 target values.
#
# The actual 2025 target remains out-of-sample.
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
# 5. CREATE ML FEATURES
# ============================================================


def create_ml_features(dataframe):

    data = dataframe.copy()

    data = (
        data
        .sort_values(["DRS", "PERIODE"])
        .reset_index(drop=True)
    )


    # --------------------------------------------------------
    # Target lags
    # --------------------------------------------------------

    for lag in LAGS:

        data[f"target_lag_{lag}"] = (
            data
            .groupby("DRS")[target]
            .shift(lag)
        )


    # --------------------------------------------------------
    # Rolling features
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Calendar features
    # --------------------------------------------------------

    data["month"] = data["PERIODE"].dt.month

    data["year"] = data["PERIODE"].dt.year


    # --------------------------------------------------------
    # Cyclic month encoding
    # --------------------------------------------------------

    data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12)

    data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12)


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


print("\nML train period:",
      ml_train["PERIODE"].min(),
      "->",
      ml_train["PERIODE"].max())


print("ML test period :",
      ml_test["PERIODE"].min(),
      "->",
      ml_test["PERIODE"].max())


# ============================================================
# 8. DEFINE ML FEATURES
# ============================================================


feature_cols = [

    "DRS",

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

    # Rolling features
    "target_roll_mean_3",
    "target_roll_mean_6",

    # Calendar
    "month",
    "year",

    # Cyclic month
    "month_sin",
    "month_cos"
]


# Keep only columns that actually exist
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
            OneHotEncoder(handle_unknown="ignore"),
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
# 11. METRICS
# ============================================================


def smape(y_true, y_pred):
    """
    Symmetric Mean Absolute Percentage Error (sMAPE).
    """

    y_true = np.asarray(y_true, dtype=float)

    y_pred = np.asarray(y_pred, dtype=float)

    denominator = np.abs(y_true) + np.abs(y_pred)

    mask = denominator != 0

    if not mask.any():
        return np.nan

    return (
        100
        * np.mean(
            2
            * np.abs(y_pred[mask] - y_true[mask])
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


    # --------------------------------------------------------
    # Train final model
    # --------------------------------------------------------

    model.fit(X_train, y_train)


    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    predictions = model.predict(X_test)


    # --------------------------------------------------------
    # Prevent negative predictions
    # --------------------------------------------------------

    predictions = np.maximum(predictions, 0)


    # --------------------------------------------------------
    # MAE
    # --------------------------------------------------------

    mae = mean_absolute_error(y_test, predictions)


    # --------------------------------------------------------
    # RMSE
    # --------------------------------------------------------

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )


    # --------------------------------------------------------
    # sMAPE
    # --------------------------------------------------------

    smape_value = smape(y_test.to_numpy(), predictions)


    # --------------------------------------------------------
    # R²
    # --------------------------------------------------------

    r2 = r2_score(y_test, predictions)


    # --------------------------------------------------------
    # MAPE
    # --------------------------------------------------------

    y_true = y_test.to_numpy()

    mask = y_true != 0

    if mask.any():

        mape = (
            100
            * np.mean(
                np.abs(
                    (
                        y_true[mask]
                        -
                        predictions[mask]
                    )
                    /
                    y_true[mask]
                )
            )
        )

        accuracy = max(0, 100 - mape)

    else:

        mape = np.nan
        accuracy = np.nan


    # --------------------------------------------------------
    # Print metrics
    # --------------------------------------------------------

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


# ============================================================
# 12. PLOT FUNCTION
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
            /
            f"Plot_DRS_{current_date}"
        )

        output_dir.mkdir(parents=True, exist_ok=True)


    # --------------------------------------------------------
    # Prepare test predictions
    # --------------------------------------------------------

    plot_test = test_df[
        ["DRS", "PERIODE", target]
    ].copy()


    plot_test["prediction"] = np.asarray(predictions)

    plot_test["PERIODE"] = pd.to_datetime(plot_test["PERIODE"])


    plot_test = (
        plot_test
        .sort_values(["DRS", "PERIODE"])
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

        plt.figure(figsize=(14, 7))


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


        plt.xlabel("PERIODE", fontsize=12)

        plt.ylabel(target, fontsize=12)


        # ----------------------------------------------------
        # Grid
        # ----------------------------------------------------

        plt.grid(True, alpha=0.25)


        # ----------------------------------------------------
        # Legend
        # ----------------------------------------------------

        plt.legend(loc="best", frameon=True)


        # ----------------------------------------------------
        # Layout
        # ----------------------------------------------------

        plt.tight_layout()


        # ----------------------------------------------------
        # Save plot
        # ----------------------------------------------------

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
# 13. MACHINE LEARNING — GRID SEARCH
# ============================================================

print("\n" + "=" * 80)
print("13. MACHINE LEARNING MODELS — GRID SEARCH")
print("=" * 80)


# ------------------------------------------------------------
# Time-series-aware cross-validation
# ------------------------------------------------------------

tscv = TimeSeriesSplit(n_splits=3)


# ------------------------------------------------------------
# ML models and hyperparameter grids
# ------------------------------------------------------------

ml_specs = {

    "RandomForest": {

        "estimator": RandomForestRegressor(
            random_state=RANDOM_STATE
        ),

        "param_grid": {

            "model__n_estimators": [
                200,
                400
            ],

            "model__max_depth": [
                None,
                10
            ]
        }
    },


    "XGBoost": {

        "estimator": XGBRegressor(
            random_state=RANDOM_STATE,
            objective="reg:squarederror"
        ),

        "param_grid": {

            "model__n_estimators": [
                200,
                400
            ],

            "model__max_depth": [
                3,
                6
            ],

            "model__learning_rate": [
                0.05,
                0.1
            ]
        }
    },


    "LightGBM": {

        "estimator": LGBMRegressor(
            random_state=RANDOM_STATE,
            verbose=-1
        ),

        "param_grid": {

            "model__n_estimators": [
                200,
                400
            ],

            "model__num_leaves": [
                15,
                31
            ],

            "model__learning_rate": [
                0.05,
                0.1
            ]
        }
    },


    "SVR": {

        "estimator": SVR(),

        "param_grid": {

            "model__C": [
                1,
                10
            ],

            "model__epsilon": [
                0.1,
                0.5
            ],

            "model__kernel": [
                "rbf"
            ]
        }
    }
}


# ============================================================
# GRID SEARCH
# ============================================================

ml_results = {}

ml_best_estimators = {}


for name, spec in ml_specs.items():

    print("\n" + "-" * 80)
    print(f"GRID SEARCH: {name}")
    print("-" * 80)


    # --------------------------------------------------------
    # Pipeline
    # --------------------------------------------------------

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                spec["estimator"]
            )
        ]
    )


    # --------------------------------------------------------
    # GridSearchCV
    # --------------------------------------------------------

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=spec["param_grid"],
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=1
    )


    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    grid.fit(X_train, y_train)


    # --------------------------------------------------------
    # Best parameters
    # --------------------------------------------------------

    print(f"\n{name} best parameters:")

    print(grid.best_params_)

    print(
        f"{name} best CV RMSE: "
        f"{-grid.best_score_:.4f}"
    )


    # --------------------------------------------------------
    # Evaluate best estimator on 2025 test set
    # --------------------------------------------------------

    result = evaluate_model(
        grid.best_estimator_,
        X_train,
        y_train,
        X_test,
        y_test,
        name
    )


    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    ml_results[name] = result

    ml_best_estimators[name] = grid.best_estimator_


# ============================================================
# 14. ML MODEL COMPARISON
# ============================================================

ml_results_df = (
    pd.DataFrame(
        [
            {
                key: value
                for key, value in result.items()
                if key not in (
                    "model_object",
                    "predictions"
                )
            }
            for result in ml_results.values()
        ]
    )
    .sort_values("RMSE")
    .reset_index(drop=True)
)


print("\n" + "=" * 80)
print("ML MODEL COMPARISON")
print("=" * 80)


print(ml_results_df.to_string(index=False))


# ============================================================
# 15. SELECT BEST ML MODEL
# ============================================================

best_ml_name = (ml_results_df.iloc[0]["model"])

best_ml_result = ml_results[best_ml_name]

best_ml_estimator = ml_best_estimators[best_ml_name]


print("\n" + "=" * 80)
print("BEST MACHINE LEARNING MODEL")
print("=" * 80)


print(f"Best ML model: {best_ml_name}")

print(f"RMSE: {best_ml_result['RMSE']:.4f}")

print(f"MAE : {best_ml_result['MAE']:.4f}")

print(f"sMAPE: {best_ml_result['sMAPE']:.2f}%")

print(f"MAPE : {best_ml_result['MAPE']:.2f}%")

print(f"R²   : {best_ml_result['R2']:.4f}")


# ============================================================
# 16. PLOT BEST ML MODEL PREDICTIONS
# ============================================================

print("\n" + "=" * 80)
print("PLOT BEST ML MODEL PREDICTIONS")
print("=" * 80)


plot_model_predictions(
    full_df=all_df,
    test_df=ml_test,
    predictions=best_ml_result["predictions"],
    model_name=f"Best ML Model: {best_ml_name}",
    target=target,
    test_start=TEST_START,
    save_plot=True
)


# ============================================================
# 17. SAVE BEST ML MODEL
# ============================================================

print("\n" + "=" * 80)
print("SAVE BEST ML MODEL")
print("=" * 80)


model_path = (REGISTRY_DIR /  f"best_ml_model_{VERSION}.pkl")


with open(model_path, "wb") as f:
    pickle.dump(best_ml_estimator, f)

print("Best ML model saved to:")

print(model_path)


# ============================================================
# 18. SAVE RESULTS / METADATA
# ============================================================

registry_snapshot = {

    "version": VERSION,
    "date": datetime.now().strftime("%Y-%m-%d"),
    "trained_at": datetime.now().isoformat(),
    "best_ml_model": best_ml_name,
    "best_ml_metrics": {
        "MAE": float(best_ml_result["MAE"]),
        "RMSE": float(best_ml_result["RMSE"]),
        "sMAPE": float(best_ml_result["sMAPE"]),
        "MAPE": float(best_ml_result["MAPE"]),
        "Accuracy": float(best_ml_result["Accuracy"]),
        "R2": float(best_ml_result["R2"])
    },
    "all_ml_results": (
        ml_results_df
        .to_dict(orient="records")
    )
}


metadata_path = (REGISTRY_DIR_META / f"summary_{VERSION}.json")


with open(metadata_path, "w") as f:

    json.dump(
        registry_snapshot,
        f,
        indent=2,
        default=str
    )


print("\nMetadata saved to:")

print(metadata_path)


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 80)
print("ML TRAINING COMPLETE")
print("=" * 80)

print(f"Best model: {best_ml_name}")

print(f"Test RMSE: {best_ml_result['RMSE']:.4f}")

print(f"Model file: {model_path}")

print("Prediction plots have been saved.")

print("=" * 80)
