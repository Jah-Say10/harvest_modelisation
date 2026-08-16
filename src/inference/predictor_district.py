import pandas as pd
import numpy as np

# ============================================================
# 3. LOAD MODEL
# ============================================================
model = pd.read_pickle("../models/registry/district_model.pkl")
columns = pd.read_pickle("../models/registry/feature_columns_district.pkl")
df = pd.read_pickle("../models/registry/history_df_district.pkl")

def predict_date(
    district,
    period,
    temperature_moyenne=None,
    temperature_max=None,
    temperature_min=None,
    precipitation=None,
    humidite=None,
    vent=None,
    rayonnement_solaire=None
):
    """
    Predict one month for one DISTRICT.

    Parameters
    ----------
    district : str
        Example: "Dakar"

    period : str or datetime
        Example: "2025-01"

    Weather variables are optional.
    """

    period = pd.to_datetime(period)

    # latest observation before requested date
    history = df[df["DISTRICT"] == district]

    history = history.loc[history.index < period]

    if history.empty:
        raise ValueError("No historical data available before this period.")

    sample = history.iloc[-1:].copy()

    # calendar features
    sample["month"] = period.month
    sample["quarter"] = period.quarter
    sample["year"] = period.year

    # overwrite weather if supplied
    if temperature_moyenne is not None:
        sample["temperature_moyenne"] = temperature_moyenne

    if temperature_max is not None:
        sample["temperature_max"] = temperature_max

    if temperature_min is not None:
        sample["temperature_min"] = temperature_min

    if precipitation is not None:
        sample["precipitation"] = precipitation

    if humidite is not None:
        sample["humidite"] = humidite

    if vent is not None:
        sample["vent"] = vent

    if rayonnement_solaire is not None:
        sample["rayonnement_solaire"] = rayonnement_solaire

    # encode exactly like training
    sample = pd.get_dummies(sample, columns=["DISTRICT"])

    sample = sample.reindex(columns=columns, fill_value=0)

    prediction = model.predict(sample)

    return prediction[0]

def predict_range(
    district,
    start,
    end,
    temperature_moyenne=None,
    temperature_max=None,
    temperature_min=None,
    precipitation=None,
    humidite=None,
    vent=None,
    rayonnement_solaire=None
):

    dates = pd.date_range(
        start=start,
        end=end,
        freq="MS"
    )

    results = []

    for date in dates:

        pred = predict_date(
            district=district,
            period=date,
            temperature_moyenne=temperature_moyenne,
            temperature_max=temperature_max,
            temperature_min=temperature_min,
            precipitation=precipitation,
            humidite=humidite,
            vent=vent,
            rayonnement_solaire=rayonnement_solaire
        )

        results.append(
            {
                "PERIODE": date,
                "Prediction": pred
            }
        )

    return pd.DataFrame(results)

def predict_district(district):

    history = df[df["DISTRICT"] == district]

    if history.empty:
        raise ValueError("Unknown DISTRICT.")

    last_period = history.index.max()

    next_period = last_period + pd.DateOffset(months=1)

    prediction = predict_date(
        district=district,
        period=next_period
    )

    return {
        "DISTRICT": district,
        "PERIODE": next_period,
        "Prediction": prediction
    }