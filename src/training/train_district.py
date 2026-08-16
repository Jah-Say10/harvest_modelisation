# ============================================================
# python -m training.train_district_global
# Global district malaria forecasting model
# ============================================================

# ============================
# Import libraries
# ============================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from datetime import datetime
import pickle
import json

# Machine learning imports
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import (
    TimeSeriesSplit,
    GridSearchCV
)

from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error
)

from sklearn.inspection import permutation_importance

# Project utilities
from utils.sort_year import sort_and_replace
from utils.save_metadata import save_metadata

# Read data in ../data/raw/Base_PEC_MALARIA_CS_2021_2024.xlsx
data = pd.read_excel('../data/features/Base_PEC_MALARIA_CS_2021_2024_with_weather.xlsx')

# Convert French month names to English
df = sort_and_replace(data)
# Sort dates correctly
print(df.head())

target = "R20_DSME_Femmes enceintes ayant reçu TPI 2"

# Variables to aggregate
variables = data.columns[3:]

# ============================================================
# 3. CREATE TIME SERIES FEATURES
# ============================================================

# Extract calendar information

df["month"] = df.index.month

df["quarter"] = (
    df.index
    .quarter
)

df["year"] = (
    df.index
    .year
)

# ------------------------------------------------------------
# Lag variables
# ------------------------------------------------------------

# Previous months target values
for lag in range(1,13):

    df[f"lag_{lag}"] = (
        df
        .groupby("DRS")[target]
        .shift(lag)
    )
    
# ------------------------------------------------------------
# Rolling statistics
# ------------------------------------------------------------

# Moving average of last 3 months

df["rolling_mean_3"] = (
    df
    .groupby("DRS")[target]
    .shift(1)
    .rolling(3)
    .mean()
)

# Moving average of last 6 months

df["rolling_mean_6"] = (
    df
    .groupby("DRS")[target]
    .shift(1)
    .rolling(6)
    .mean()
)

# Difference with previous month

df["diff_1"] = (
    df
    .groupby("DRS")[target]
    .diff(1)
)

# Remove rows without history

df = df.dropna()

print(
    "Dataset after feature engineering:",
    df.shape
)

# ============================================================
# 4. ENCODE DISTRICT
# ============================================================

# RandomForest cannot read strings
# Convert DISTRICT categories to numerical columns

df_model = pd.get_dummies(
    df,
    columns=["DISTRICT"],
    drop_first=False
)

# ============================================================
# 5. TRAIN TEST SPLIT (TIME BASED)
# ============================================================

# Sort again after encoding

df_model = df_model.sort_index()

# Define X and y

X = df_model.drop(
    columns=[
        target,
        "DRS"
    ]
)

y = df_model[target]

# ------------------------------------------------------------
# Use last 20% as test period
# ------------------------------------------------------------

split_index = int(
    len(df_model)*0.8
)

X_train = X.iloc[:split_index]

X_test = X.iloc[split_index:]


y_train = y.iloc[:split_index]

y_test = y.iloc[split_index:]

print(
    "Train:",
    X_train.shape
)

print(
    "Test:",
    X_test.shape
)

# ============================================================
# 6. GRID SEARCH WITH TIME SERIES CV
# ============================================================

# Time series folds
tscv = TimeSeriesSplit(
    n_splits=5
)

# Random Forest model
rf = RandomForestRegressor(
    random_state=42
)

# Parameters to test
params = {
    "n_estimators":
    [
        100,
        300,
        500
    ],
    "max_depth":
    [
        5,
        10,
        20,
        None
    ],
    "min_samples_split":
    [
        2,
        5,
        10
    ],
    "max_features":
    [
        "sqrt",
        "log2"
    ]
}
# Grid search
grid = GridSearchCV(
    estimator=rf,
    param_grid=params,
    cv=tscv,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    verbose=2
)

print("Starting GridSearch...")

grid.fit(
    X_train,
    y_train
)

# Best model
model = grid.best_estimator_

print("BEST PARAMETERS")
print(grid.best_params_)

# ============================================================
# 7. MODEL EVALUATION
# ============================================================

prediction = model.predict(X_test)
mae = mean_absolute_error(y_test, prediction)
rmse = root_mean_squared_error(y_test, prediction)
mape = mean_absolute_percentage_error(y_test, prediction)
accuracy = (1-mean_absolute_percentage_error(y_test, prediction))*100
metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape, "accuracy": accuracy}
print(metrics)

# ============================================================
# 8. VISUALIZATION
# ============================================================

# ------------------------------------------------------------
# Actual vs predicted
# ------------------------------------------------------------

plt.figure(figsize=(14,5))


plt.plot(
    y_test.values,
    label="Actual",
    linewidth=3
)


plt.plot(
    prediction,
    label="Prediction",
    linewidth=3
)

plt.title("Actual vs predicted malaria indicator")

plt.legend()
plt.grid()
plt.show()

# ------------------------------------------------------------
# Residual plot
# ------------------------------------------------------------

residuals = (y_test.values - prediction)

plt.figure(figsize=(10,5))
sns.histplot(residuals, kde=True)
plt.title("Prediction residual distribution")
plt.show()

# ------------------------------------------------------------
# Feature importance
# ------------------------------------------------------------
importance = pd.DataFrame(
    {
        "feature": X.columns,
        "importance":model.feature_importances_

    }
)

importance = (
    importance
    .sort_values(
        "importance",
        ascending=False
    )
    .head(20)
)

plt.figure(figsize=(10,6))

sns.barplot(
    data=importance,
    x="importance",
    y="feature"
)

plt.title("Top 20 important variables")
plt.show()

# ============================================================
# 9. SAVE MODEL
# ============================================================

# Save the model
with open("../models/registry/district_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save feature names
with open("../models/registry/feature_columns_district.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

# Save history dataframe
with open("../models/registry/history_df_district.pkl", "wb") as f:
    pickle.dump(df, f)
    
# Save metadata
model_name = grid.best_estimator_.__class__.__name__
# Save metadata with the best parameters and accuracy
save_metadata(
    district="DISTRICT",
    algorithm=model_name,
    accuracy=accuracy,
    training_rows=len(X_train),
    version="1.0.0",
    best_params=grid.best_params_,
    features=list(X.columns)
)

# ============================================================
# 10. TEST DAKAR WITH DIFFERENT CONDITIONS
# ============================================================


def predict_date(
    district,

    period,
    temperature=None,
    precipitation=None,
    humidity=None,
    wind=None,
    radiation=None
):
    """
    Predict one month for one DRS.

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
    if temperature is not None:
        sample["temperature_moyenne"] = temperature

    if precipitation is not None:
        sample["precipitation"] = precipitation

    if humidity is not None:
        sample["humidite"] = humidity

    if wind is not None:
        sample["vent"] = wind

    if radiation is not None:
        sample["rayonnement_solaire"] = radiation

    # encode exactly like training
    sample = pd.get_dummies(sample, columns=["DRS"])

    sample = sample.reindex(columns=X.columns, fill_value=0)

    prediction = model.predict(sample)

    return prediction[0]

def predict_range(
    district,
    start,
    end,
    temperature=None,
    precipitation=None,
    humidity=None,
    wind=None,
    radiation=None
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
            temperature=temperature,
            precipitation=precipitation,
            humidity=humidity,
            wind=wind,
            radiation=radiation
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
        raise ValueError("Unknown district.")

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
    
forecast = predict_date(
    district="Dakar Sud",
    period="2025-01",
    temperature=31,
    precipitation=120
)
print(forecast)

forecast = predict_range(
    district="Dakar Sud",
    start="2025-01",
    end="2025-12"
)
print(forecast)

forecast = predict_district("Dakar Sud")
print(forecast)

# Example scenarios
scenarios = {
    "normal":
    {
        "district": "Dakar Sud",
        "temperature":25,
        "precipitation":50,
        "humidity":70,
        "wind":20,
        "radiation":40
    },
    "hot_dry":
    {
        "district": "Dakar Sud",
        "temperature":35,
        "precipitation":0,
        "humidity":30,
        "wind":15,
        "radiation":70
    },
    "rainy":
    {
        "district": "Dakar Sud",
        "temperature":28,
        "precipitation":200,
        "humidity":90,
        "wind":25,
        "radiation":20
    }
}