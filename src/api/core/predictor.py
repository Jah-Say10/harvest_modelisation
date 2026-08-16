"""
python -m inference.predictor_drs

Regional malaria prediction module.

Usage
-----
from prediction_region import predict_region

forecast = predict_region(
    drs="Dakar",
    start_date="2025-01-01",
    end_date="2025-12-01",
    weather_df=weather_df
)
"""

from pathlib import Path
import os
import pickle

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

# Répertoire des artefacts. Surchargeable via la variable d'environnement
# PALU_MODEL_DIR ; par défaut : <racine du projet>/models/registry
MODEL_DIR = Path(
    os.environ.get(
        "PALU_MODEL_DIR",
        Path(__file__).resolve().parent.parent / "models" / "registry",
    )
)

MODEL_PATH = MODEL_DIR / "drs_model.pkl"
HISTORY_PATH = MODEL_DIR / "history_df_drs.pkl"

TARGET = "Cas Confirmés (par TDR) palu consultations externes"

LAGS = [1, 2, 3, 6, 12]

WEATHER_VARS = [
    "temperature_moyenne",
    "temperature_max",
    "temperature_min",
    "precipitation",
    "humidite",
    "vent",
    "rayonnement_solaire",
]


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

def load_model():
    """
    Load the trained regional model.

    Returns
    -------
    model
        Trained sklearn Pipeline.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    return model


# ============================================================
# LOAD HISTORY
# ============================================================

def load_history():
    """
    Load the historical regional dataset used for training.

    Returns
    -------
    pd.DataFrame
    """

    if not HISTORY_PATH.exists():
        raise FileNotFoundError(
            f"History dataframe not found: {HISTORY_PATH}"
        )

    with open(HISTORY_PATH, "rb") as f:
        history = pickle.load(f)

    history = history.copy()

    history["PERIODE"] = pd.to_datetime(
        history["PERIODE"]
    )

    history = history.sort_values(
        ["DRS", "PERIODE"]
    ).reset_index(drop=True)

    return history


# ============================================================
# CREATE FEATURES
# ============================================================

def create_features(data):
    """
    Create exactly the lag, rolling and calendar features
    used by the training pipeline.

    Parameters
    ----------
    data : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """

    data = data.copy()

    data["PERIODE"] = pd.to_datetime(
        data["PERIODE"]
    )

    data = data.sort_values(
        ["DRS", "PERIODE"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Target lags
    # --------------------------------------------------------

    for lag in LAGS:

        data[f"target_lag_{lag}"] = (
            data
            .groupby("DRS")[TARGET]
            .shift(lag)
        )

    # --------------------------------------------------------
    # Rolling means
    # --------------------------------------------------------

    data["target_roll_mean_3"] = (
        data
        .groupby("DRS")[TARGET]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(3)
            .mean()
        )
    )

    data["target_roll_mean_6"] = (
        data
        .groupby("DRS")[TARGET]
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
        2 * np.pi * data["month"] / 12
    )

    data["month_cos"] = np.cos(
        2 * np.pi * data["month"] / 12
    )

    return data


# ============================================================
# FEATURE COLUMNS
# ============================================================

def get_feature_columns(data):
    """
    Return the feature columns expected by the trained model.
    """

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
        "month_cos",
    ]

    return [
        col
        for col in feature_cols
        if col in data.columns
    ]


# ============================================================
# VALIDATE WEATHER DATA
# ============================================================

def validate_weather(
    weather_df,
    required_dates
):
    """
    Validate the future weather dataframe.

    weather_df must contain:
        PERIODE
        temperature_moyenne
        temperature_max
        temperature_min
        precipitation
        humidite
        vent
        rayonnement_solaire
    """

    if weather_df is None:
        raise ValueError(
            "weather_df is required for future prediction."
        )

    weather = weather_df.copy()

    if "PERIODE" not in weather.columns:
        raise ValueError(
            "weather_df must contain a 'PERIODE' column."
        )

    weather["PERIODE"] = pd.to_datetime(
        weather["PERIODE"]
    )

    # --------------------------------------------------------
    # Check weather variables
    # --------------------------------------------------------

    missing_weather = [
        col
        for col in WEATHER_VARS
        if col not in weather.columns
    ]

    if missing_weather:

        raise ValueError(
            "Missing weather variables: "
            + ", ".join(missing_weather)
        )

    # --------------------------------------------------------
    # Check duplicate dates
    # --------------------------------------------------------

    duplicates = weather[
        weather["PERIODE"].duplicated()
    ]

    if not duplicates.empty:

        raise ValueError(
            "weather_df contains duplicate dates."
        )

    # --------------------------------------------------------
    # Check requested dates
    # --------------------------------------------------------

    required_dates = pd.DatetimeIndex(
        required_dates
    )

    missing_dates = required_dates[
        ~required_dates.isin(
            weather["PERIODE"]
        )
    ]

    if len(missing_dates) > 0:

        raise ValueError(
            "Missing weather data for dates: "
            + ", ".join(
                date.strftime("%Y-%m-%d")
                for date in missing_dates
            )
        )

    return weather


# ============================================================
# PREDICTION
# ============================================================

def predict_region(
    drs,
    start_date,
    end_date,
    weather_df,
    model=None,
    history=None,
):
    """
    Predict the target for one DRS over a date range.

    Parameters
    ----------
    drs : str
        Name of the DRS.

    start_date : str or datetime
        First prediction month.

    end_date : str or datetime
        Last prediction month.

    weather_df : pd.DataFrame
        Future monthly weather data.

    model, history : optional
        Objets déjà chargés (évite de relire les pickles à chaque appel,
        utilisé par l'interface Streamlit). Si None, ils sont chargés
        depuis MODEL_DIR.

        Required columns:
            PERIODE
            temperature_moyenne
            temperature_max
            temperature_min
            precipitation
            humidite
            vent
            rayonnement_solaire

    Returns
    -------
    pd.DataFrame

        Columns:
            DRS
            PERIODE
            prediction
    """

    # ========================================================
    # LOAD MODEL AND HISTORY
    # ========================================================

    if model is None:
        model = load_model()

    if history is None:
        history = load_history()

    # ========================================================
    # VALIDATE DRS
    # ========================================================

    available_drs = (
        history["DRS"]
        .dropna()
        .unique()
    )

    if drs not in available_drs:

        raise ValueError(
            f"Unknown DRS: {drs}. "
            f"Available DRS: {list(available_drs)}"
        )

    # ========================================================
    # PREPARE DATES
    # ========================================================

    start_date = pd.Timestamp(
        start_date
    )

    end_date = pd.Timestamp(
        end_date
    )

    # Force monthly start
    start_date = start_date.to_period(
        "M"
    ).to_timestamp()

    end_date = end_date.to_period(
        "M"
    ).to_timestamp()

    if start_date > end_date:

        raise ValueError(
            "start_date must be <= end_date."
        )

    future_dates = pd.date_range(
        start=start_date,
        end=end_date,
        freq="MS"
    )

    if len(future_dates) == 0:

        raise ValueError(
            "No prediction dates generated."
        )

    # ========================================================
    # WEATHER
    # ========================================================

    weather = validate_weather(
        weather_df,
        future_dates
    )

    weather = (
        weather[
            weather["PERIODE"].isin(
                future_dates
            )
        ]
        .sort_values("PERIODE")
        .reset_index(drop=True)
    )

    # ========================================================
    # GET REGION HISTORY
    # ========================================================

    region_history = (
        history[
            history["DRS"] == drs
        ]
        .sort_values("PERIODE")
        .copy()
    )

    # --------------------------------------------------------
    # Important:
    #
    # We need historical observations immediately before
    # the requested prediction period because the model
    # uses lags up to 12 months.
    # --------------------------------------------------------

    region_history = region_history[
        region_history["PERIODE"] < start_date
    ].copy()

    if region_history.empty:

        raise ValueError(
            f"No historical observations before "
            f"{start_date.date()} for DRS '{drs}'."
        )

    # ========================================================
    # CHECK LAG HISTORY
    # ========================================================

    if len(region_history) < 12:

        raise ValueError(
            f"At least 12 historical months are required "
            f"before {start_date.date()} for recursive "
            f"prediction. Only {len(region_history)} found."
        )

    # ========================================================
    # CREATE FUTURE DATAFRAME
    # ========================================================

    future = pd.DataFrame({
        "DRS": drs,
        "PERIODE": future_dates
    })

    # Add weather
    future = future.merge(
        weather,
        on="PERIODE",
        how="left"
    )

    # Target is unknown for future months
    future[TARGET] = np.nan

    # ========================================================
    # COMBINE HISTORY + FUTURE
    # ========================================================

    combined = pd.concat(
        [
            region_history,
            future
        ],
        ignore_index=True
    )

    combined = (
        combined
        .sort_values("PERIODE")
        .reset_index(drop=True)
    )

    # ========================================================
    # RECURSIVE FORECAST
    # ========================================================

    predictions = []

    for future_date in future_dates:

        # ----------------------------------------------------
        # Recalculate features using all known observations
        # ----------------------------------------------------

        combined = create_features(
            combined
        )

        # ----------------------------------------------------
        # Locate current future row
        # ----------------------------------------------------

        idx = combined.index[
            combined["PERIODE"] == future_date
        ]

        if len(idx) != 1:

            raise RuntimeError(
                f"Could not locate prediction date "
                f"{future_date}."
            )

        idx = idx[0]

        # ----------------------------------------------------
        # Feature columns
        # ----------------------------------------------------

        feature_cols = get_feature_columns(
            combined
        )

        X_future = combined.loc[
            [idx],
            feature_cols
        ]

        # ----------------------------------------------------
        # Check missing features
        # ----------------------------------------------------

        missing = X_future.columns[
            X_future.isnull().any()
        ].tolist()

        if missing:

            raise ValueError(
                f"Missing feature values for "
                f"{future_date.date()}: "
                f"{missing}"
            )

        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

        prediction = model.predict(
            X_future
        )[0]

        # Target cannot be negative
        prediction = max(
            0,
            float(prediction)
        )

        predictions.append(
            prediction
        )

        # ----------------------------------------------------
        # Put prediction back into history
        #
        # This allows the next month to use:
        #
        # lag_1
        # lag_2
        # rolling_mean_3
        # rolling_mean_6
        #
        # based on the previous predictions.
        # ----------------------------------------------------

        combined.loc[
            idx,
            TARGET
        ] = prediction

    # ========================================================
    # RESULT
    # ========================================================

    forecast = pd.DataFrame({
        "DRS": drs,
        "PERIODE": future_dates,
        "prediction": predictions
    })

    return forecast


# ============================================================
# OPTIONAL: SAVE FORECAST
# ============================================================

def save_forecast(
    forecast_df,
    output_path
):
    """
    Save prediction dataframe to CSV.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    forecast_df.to_csv(
        output_path,
        index=False
    )

    return output_path


# ============================================================
# EXAMPLE
# ============================================================

if __name__ == "__main__":

    # Example future weather
    #
    # Replace these values with actual forecast/
    # supplied weather values.

    weather_df = pd.DataFrame({
        "PERIODE": pd.date_range(
            "2025-01-01",
            "2025-12-01",
            freq="MS"
        ),

        "temperature_moyenne": np.zeros(12),
        "temperature_max": np.zeros(12),
        "temperature_min": np.zeros(12),
        "precipitation": np.zeros(12),
        "humidite": np.zeros(12),
        "vent": np.zeros(12),
        "rayonnement_solaire": np.zeros(12),
    })

    forecast = predict_region(
        drs="DRS Dakar",
        start_date="2025-01-01",
        end_date="2025-12-01",
        weather_df=weather_df
    )

    print(
        "\nREGIONAL FORECAST"
    )

    print(
        forecast.to_string(
            index=False
        )
    )