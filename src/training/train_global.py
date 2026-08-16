# ============================================================
# python -m training.train_global
# Global regional malaria forecasting model
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

# ============================================================
# 1. LOAD DATA
# ============================================================

# Read feature dataset
data = pd.read_excel("../data/features/Base_PEC_MALARIA_CS_2021_2024_with_weather.xlsx")

# ============================================================
# 2. AGGREGATE DATA BY REGION AND MONTH
# ============================================================

# Target variable
target = "R20_DSME_Femmes enceintes ayant reçu TPI 2"

# Variables météorologiques
meteo_vars = [
    "temperature_moyenne",
    "temperature_max",
    "temperature_min",
    "precipitation",
    "humidite",
    "vent",
    "rayonnement_solaire"
]

# Définir l'agrégation pour chaque variable
agg_dict = {
    target: "sum",
    "temperature_moyenne": "mean",
    "temperature_max": "mean",
    "temperature_min": "mean",
    "precipitation": "sum",
    "humidite": "mean",
    "vent": "mean",
    "rayonnement_solaire": "mean"
}

# Agrégation par période
data_agg = (
    data
    .groupby("PERIODE", as_index=False)
    .agg(agg_dict)
)

# Display structure
print(data_agg.iloc)
print(data_agg.info())

# Sort dates correctly
df = sort_and_replace(data_agg)