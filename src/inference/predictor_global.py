import pandas as pd
import numpy as np
import json

from inference.predictor_drs import predict_date, predict_range

from utils.get_drs import drs_districts

DRS = json.loads(drs_districts)
DRS = list(DRS.keys())

print(f"DRS: {DRS}")

# Sum for all DRS to get a national prediction by single dates
def predict_national_date(
    period,
    temperature_moyenne=None,
    temperature_max=None,
    temperature_min=None,
    precipitation=None,
    humidite=None,
    vent=None,
    rayonnement_solaire=None
):
    national_prediction = 0

    for drs in DRS:
        drs_prediction = predict_date(
            drs=drs,
            period=period,
            temperature_moyenne=temperature_moyenne,
            temperature_max=temperature_max,
            temperature_min=temperature_min,
            precipitation=precipitation,
            humidite=humidite,
            vent=vent,
            rayonnement_solaire=rayonnement_solaire
        )
        national_prediction += drs_prediction

    return national_prediction

# Sum for all DRS to get a national prediction by range of dates
def predict_national_range(
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
    """
    Predict all DRS and aggregate their predictions
    to obtain the national forecast.

    Returns
    -------
    pd.DataFrame
        National prediction by period.
    """

    national_results = []

    for drs in DRS:

        drs_results = predict_range(
            drs=drs,
            start=start,
            end=end,
            temperature_moyenne=temperature_moyenne,
            temperature_max=temperature_max,
            temperature_min=temperature_min,
            precipitation=precipitation,
            humidite=humidite,
            vent=vent,
            rayonnement_solaire=rayonnement_solaire
        )

        # Keep the DRS identifier
        drs_results = drs_results.copy()
        drs_results["DRS"] = drs

        national_results.append(
            drs_results
        )

    # Combine all DRS results
    all_drs_predictions = pd.concat(
        national_results,
        ignore_index=True
    )

    # Aggregate by period
    national_prediction = (
        all_drs_predictions
        .groupby("PERIODE", as_index=False)["Prediction"]
        .sum()
        .rename(
            columns={
                "Prediction": "National_Prediction"
            }
        )
    )

    return national_prediction