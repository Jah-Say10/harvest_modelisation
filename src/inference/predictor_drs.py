# ============================================================
# python -m inference.predictor_drs
#
# GLOBAL REGIONAL MALARIA FORECASTING
#
# MODEL PREDICTION
#    - Load saved ML model
#    - Load metadata
#    - Load historical + future data
#    - Recreate ML features
#    - Predict 2025
#    - Evaluate predictions
#    - Save predictions
#    - Plot predictions by DRS
#
# ============================================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from pathlib import Path

import pickle
import json
import warnings

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_VERSION = "1.0.0"

MODEL_PATH = (
    Path("../models/registry")
    / f"best_ml_model_{MODEL_VERSION}.pkl"
)

METADATA_PATH = (
    Path("../models/metadata")
    / f"summary_{MODEL_VERSION}.json"
)


# ------------------------------------------------------------
# Data
# ------------------------------------------------------------

DATA_PATH_HISTORICAL = (
    "../data/features/"
    "Base_MALARIA_CS_CLEAN_2021_2024_with_weather.xlsx"
)

DATA_PATH_TEST = (
    "../data/features/"
    "Base_MALARIA_CS_CLEAN_2025_with_weather.xlsx"
)


# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

OUTPUT_DIR = Path("../models/predictions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


PLOT_DIR = OUTPUT_DIR / f"plots_{MODEL_VERSION}"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. LOAD MODEL
# ============================================================

print("\n" + "=" * 80)
print("1. LOAD SAVED MODEL")
print("=" * 80)


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )


with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


print("Model loaded:")
print(MODEL_PATH)


# ============================================================
# 2. LOAD METADATA
# ============================================================

print("\n" + "=" * 80)
print("2. LOAD MODEL METADATA")
print("=" * 80)


if not METADATA_PATH.exists():
    raise FileNotFoundError(
        f"Metadata not found:\n{METADATA_PATH}"
    )


with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)


print("Metadata loaded:")
print(METADATA_PATH)


# ============================================================
# 3. READ MODEL CONFIGURATION
# ============================================================

target = metadata["target"]

feature_cols = metadata["feature_columns"]

categorical_features = metadata["categorical_features"]

numeric_features = metadata["numeric_features"]

weather_variables = metadata["weather_variables"]

LAGS = metadata["lags"]


print("\nModel version:")
print(metadata["version"])


print("\nBest model:")
print(metadata["best_ml_model"])


print("\nTarget:")
print(target)


print("\nFeatures:")
for feature in feature_cols:
    print("  -", feature)


print("\nLags:")
print(LAGS)


# ============================================================
# 4. LOAD DATA
# ============================================================

print("\n" + "=" * 80)
print("3. LOAD DATA")
print("=" * 80)


historical_raw = pd.read_excel(
    DATA_PATH_HISTORICAL
)

future_raw = pd.read_excel(
    DATA_PATH_TEST
)


print("Historical raw:", historical_raw.shape)

print("Future raw    :", future_raw.shape)


# ============================================================
# 5. AGGREGATE DATA
# ============================================================

print("\n" + "=" * 80)
print("4. AGGREGATE DATA")
print("=" * 80)


# ------------------------------------------------------------
# Verify required columns
# ------------------------------------------------------------

required_columns = [
    "DRS",
    "PERIODE",
    target,
    *weather_variables
]


missing_historical = [
    col
    for col in required_columns
    if col not in historical_raw.columns
]


missing_future = [
    col
    for col in required_columns
    if col not in future_raw.columns
]


if missing_historical:
    raise ValueError(
        "Missing columns in historical data:\n"
        + "\n".join(missing_historical)
    )


if missing_future:
    raise ValueError(
        "Missing columns in future data:\n"
        + "\n".join(missing_future)
    )


# ------------------------------------------------------------
# Aggregation
# ------------------------------------------------------------

agg_dict = {
    target: "sum",
    **{
        col: "mean"
        for col in weather_variables
    }
}


historical_df = (
    historical_raw
    .groupby(["DRS", "PERIODE"])
    .agg(agg_dict)
    .reset_index()
)


future_df = (
    future_raw
    .groupby(["DRS", "PERIODE"])
    .agg(agg_dict)
    .reset_index()
)


# ------------------------------------------------------------
# Convert dates
# ------------------------------------------------------------

historical_df["PERIODE"] = pd.to_datetime(
    historical_df["PERIODE"]
)

future_df["PERIODE"] = pd.to_datetime(
    future_df["PERIODE"]
)


# ------------------------------------------------------------
# Sort
# ------------------------------------------------------------

historical_df = (
    historical_df
    .sort_values(["DRS", "PERIODE"])
    .reset_index(drop=True)
)


future_df = (
    future_df
    .sort_values(["DRS", "PERIODE"])
    .reset_index(drop=True)
)


print("Historical aggregated:", historical_df.shape)

print("Future aggregated    :", future_df.shape)


print(
    "\nHistorical period:",
    historical_df["PERIODE"].min(),
    "->",
    historical_df["PERIODE"].max()
)


print(
    "Future period    :",
    future_df["PERIODE"].min(),
    "->",
    future_df["PERIODE"].max()
)


# ============================================================
# 6. COMBINE HISTORICAL + FUTURE
# ============================================================

print("\n" + "=" * 80)
print("5. COMBINE DATA FOR FEATURE ENGINEERING")
print("=" * 80)


all_df = pd.concat(
    [
        historical_df,
        future_df
    ],
    ignore_index=True
)


all_df = (
    all_df
    .sort_values(["DRS", "PERIODE"])
    .reset_index(drop=True)
)


print("Combined:", all_df.shape)


# ============================================================
# 7. CREATE ML FEATURES
# ============================================================

def create_ml_features(dataframe, target, lags):

    data = dataframe.copy()

    data = (
        data
        .sort_values(["DRS", "PERIODE"])
        .reset_index(drop=True)
    )


    # --------------------------------------------------------
    # Target lags
    # --------------------------------------------------------

    for lag in lags:

        data[f"target_lag_{lag}"] = (
            data
            .groupby("DRS")[target]
            .shift(lag)
        )


    # --------------------------------------------------------
    # Rolling features
    #
    # IMPORTANT:
    # shift(1) prevents current target from being used.
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Calendar features
    # --------------------------------------------------------

    data["month"] = (
        data["PERIODE"].dt.month
    )

    data["year"] = (
        data["PERIODE"].dt.year
    )


    # --------------------------------------------------------
    # Cyclic month encoding
    # --------------------------------------------------------

    data["month_sin"] = (
        np.sin(
            2
            * np.pi
            * data["month"]
            / 12
        )
    )


    data["month_cos"] = (
        np.cos(
            2
            * np.pi
            * data["month"]
            / 12
        )
    )


    return data


ml_df = create_ml_features(
    all_df,
    target,
    LAGS
)


print("\nFeature-engineered data:")
print(ml_df.shape)


# ============================================================
# 8. SELECT FUTURE DATA
# ============================================================

print("\n" + "=" * 80)
print("6. PREPARE PREDICTION DATA")
print("=" * 80)


future_start = future_df["PERIODE"].min()

future_end = future_df["PERIODE"].max()


prediction_df = ml_df[
    (ml_df["PERIODE"] >= future_start)
    &
    (ml_df["PERIODE"] <= future_end)
].copy()


# ------------------------------------------------------------
# Check required features
# ------------------------------------------------------------

missing_features = [
    col
    for col in feature_cols
    if col not in prediction_df.columns
]


if missing_features:

    raise ValueError(
        "Required model features are missing:\n"
        + "\n".join(missing_features)
    )


# ------------------------------------------------------------
# Check missing values
# ------------------------------------------------------------

missing_rows = prediction_df[
    feature_cols
].isna().any(axis=1)


if missing_rows.any():

    print(
        "\nWARNING:",
        missing_rows.sum(),
        "prediction rows contain missing features."
    )

    print(
        prediction_df.loc[
            missing_rows,
            ["DRS", "PERIODE"]
        ]
    )

    raise ValueError(
        "Cannot predict because required "
        "lag/rolling features contain NaN values."
    )


X_future = prediction_df[
    feature_cols
]


print("\nPrediction features:", X_future.shape)


print(
    "Prediction period:",
    prediction_df["PERIODE"].min(),
    "->",
    prediction_df["PERIODE"].max()
)


# ============================================================
# 9. PREDICT
# ============================================================

print("\n" + "=" * 80)
print("7. GENERATE PREDICTIONS")
print("=" * 80)


predictions = model.predict(
    X_future
)


# ------------------------------------------------------------
# Prevent negative malaria predictions
# ------------------------------------------------------------

predictions = np.maximum(
    predictions,
    0
)


prediction_df["prediction"] = predictions


print(
    "\nPredictions generated:",
    len(predictions)
)


print(
    "\nPrediction summary:"
)

print(
    pd.Series(predictions).describe()
)


# ============================================================
# 10. EVALUATE IF ACTUAL TARGET EXISTS
# ============================================================

print("\n" + "=" * 80)
print("8. EVALUATE PREDICTIONS")
print("=" * 80)


y_true = prediction_df[target].to_numpy()

y_pred = prediction_df["prediction"].to_numpy()


# ------------------------------------------------------------
# MAE
# ------------------------------------------------------------

mae = np.mean(
    np.abs(
        y_true - y_pred
    )
)


# ------------------------------------------------------------
# RMSE
# ------------------------------------------------------------

rmse = np.sqrt(
    np.mean(
        (y_true - y_pred) ** 2
    )
)


# ------------------------------------------------------------
# sMAPE
# ------------------------------------------------------------

denominator = (
    np.abs(y_true)
    +
    np.abs(y_pred)
)


mask_smape = denominator != 0


if mask_smape.any():

    smape_value = (
        100
        * np.mean(
            2
            * np.abs(
                y_pred[mask_smape]
                -
                y_true[mask_smape]
            )
            /
            denominator[mask_smape]
        )
    )

else:

    smape_value = np.nan


# ------------------------------------------------------------
# MAPE
# ------------------------------------------------------------

mask_mape = y_true != 0


if mask_mape.any():

    mape = (
        100
        * np.mean(
            np.abs(
                (
                    y_true[mask_mape]
                    -
                    y_pred[mask_mape]
                )
                /
                y_true[mask_mape]
            )
        )
    )

else:

    mape = np.nan


# ------------------------------------------------------------
# R²
# ------------------------------------------------------------

ss_res = np.sum(
    (y_true - y_pred) ** 2
)

ss_tot = np.sum(
    (y_true - np.mean(y_true)) ** 2
)


if ss_tot != 0:

    r2 = (
        1
        -
        ss_res / ss_tot
    )

else:

    r2 = np.nan


print(f"\nMAE   : {mae:.4f}")

print(f"RMSE  : {rmse:.4f}")

print(f"sMAPE : {smape_value:.2f}%")

print(f"MAPE  : {mape:.2f}%")

print(f"R²    : {r2:.4f}")


# ============================================================
# 11. SAVE PREDICTIONS
# ============================================================

print("\n" + "=" * 80)
print("9. SAVE PREDICTIONS")
print("=" * 80)


prediction_output = prediction_df[
    [
        "DRS",
        "PERIODE",
        target,
        "prediction"
    ]
].copy()


prediction_output = (
    prediction_output
    .sort_values(
        ["DRS", "PERIODE"]
    )
    .reset_index(drop=True)
)


prediction_path = (
    OUTPUT_DIR
    /
    f"predictions_{MODEL_VERSION}.csv"
)


prediction_output.to_csv(
    prediction_path,
    index=False
)


print(
    "Predictions saved to:"
)

print(prediction_path)


# ============================================================
# 12. SAVE PREDICTION METADATA
# ============================================================

prediction_metadata = {

    "model_version": MODEL_VERSION,

    "model": metadata["best_ml_model"],

    "prediction_date": (
        pd.Timestamp.now()
        .isoformat()
    ),

    "prediction_period": {
        "start": str(
            prediction_df["PERIODE"].min()
        ),
        "end": str(
            prediction_df["PERIODE"].max()
        )
    },

    "metrics": {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "sMAPE": float(smape_value),
        "MAPE": float(mape),
        "R2": float(r2)
    },

    "number_of_predictions": int(
        len(prediction_df)
    ),

    "number_of_regions": int(
        prediction_df["DRS"].nunique()
    )
}


prediction_metadata_path = (
    OUTPUT_DIR
    /
    f"prediction_summary_{MODEL_VERSION}.json"
)


with open(
    prediction_metadata_path,
    "w"
) as f:

    json.dump(
        prediction_metadata,
        f,
        indent=2
    )


print(
    "\nPrediction metadata saved to:"
)

print(prediction_metadata_path)


# ============================================================
# 13. PLOT PREDICTIONS
# ============================================================

print("\n" + "=" * 80)
print("10. PLOT PREDICTIONS")
print("=" * 80)


regions = (
    prediction_df["DRS"]
    .dropna()
    .unique()
)


for region in regions:

    historical = (
        historical_df[
            historical_df["DRS"] == region
        ]
        .sort_values("PERIODE")
    )


    region_prediction = (
        prediction_df[
            prediction_df["DRS"] == region
        ]
        .sort_values("PERIODE")
    )


    plt.figure(
        figsize=(14, 7)
    )


    # --------------------------------------------------------
    # Historical
    # --------------------------------------------------------

    plt.plot(
        historical["PERIODE"],
        historical[target],
        color="black",
        linewidth=2,
        label="Historical"
    )


    # --------------------------------------------------------
    # Actual 2025
    # --------------------------------------------------------

    plt.plot(
        region_prediction["PERIODE"],
        region_prediction[target],
        color="royalblue",
        linewidth=2.5,
        marker="o",
        markersize=5,
        label="2025 actual"
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    plt.plot(
        region_prediction["PERIODE"],
        region_prediction["prediction"],
        color="red",
        linewidth=2.5,
        linestyle="--",
        marker="o",
        markersize=5,
        label="2025 prediction"
    )


    # --------------------------------------------------------
    # Test boundary
    # --------------------------------------------------------

    plt.axvline(
        future_start,
        color="gray",
        linestyle=":",
        linewidth=2,
        label="Prediction start"
    )


    # --------------------------------------------------------
    # Test shading
    # --------------------------------------------------------

    plt.axvspan(
        future_start,
        future_end,
        color="orange",
        alpha=0.08
    )


    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    plt.title(
        f"{metadata['best_ml_model']} — {region}",
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


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    safe_region = (
        str(region)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )


    plot_path = (
        PLOT_DIR
        /
        f"{metadata['best_ml_model']}_"
        f"{safe_region}_"
        f"prediction.png"
    )


    plt.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.show()

    plt.close()


# ============================================================
# 14. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("PREDICTION COMPLETE")
print("=" * 80)


print(
    f"Model      : {metadata['best_ml_model']}"
)

print(
    f"Version    : {MODEL_VERSION}"
)

print(
    f"Period     : "
    f"{future_start} -> {future_end}"
)

print(
    f"Regions    : "
    f"{prediction_df['DRS'].nunique()}"
)

print(
    f"Predictions: "
    f"{len(prediction_df)}"
)

print(
    f"RMSE       : {rmse:.4f}"
)

print(
    f"MAE        : {mae:.4f}"
)

print(
    f"R²         : {r2:.4f}"
)

print(
    f"\nPrediction file:"
)

print(prediction_path)

print(
    "\nPlots:"
)

print(PLOT_DIR)

print("\n" + "=" * 80)
