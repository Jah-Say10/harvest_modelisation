"""
Couche de services pour l'interface Streamlit.

Tout ce qui touche au chargement, à la préparation des données météo,
aux prévisions multi-DRS, aux scénarios et à l'évaluation vit ici.
Les vues (dossier views/) ne font qu'appeler ces fonctions et afficher.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import streamlit as st

from core import predictor as P

TARGET = P.TARGET
WEATHER_VARS = P.WEATHER_VARS
LAGS = P.LAGS

# Libellés lisibles pour l'interface
WEATHER_LABELS = {
    "temperature_moyenne": "Température moyenne (°C)",
    "temperature_max": "Température max (°C)",
    "temperature_min": "Température min (°C)",
    "precipitation": "Précipitations (mm/j)",
    "humidite": "Humidité (%)",
    "vent": "Vent (m/s)",
    "rayonnement_solaire": "Rayonnement solaire (kWh/m²)",
}

MOIS_FR = [
    "Jan", "Fév", "Mar", "Avr", "Mai", "Juin",
    "Juil", "Août", "Sep", "Oct", "Nov", "Déc",
]

# Centroïdes approximatifs des régions du Sénégal (pour la carte)
DRS_COORDS = {
    "DRS Dakar":       (14.72, -17.45),
    "DRS Diourbel":    (14.66, -16.23),
    "DRS Fatick":      (14.34, -16.41),
    "DRS Kaffrine":    (14.11, -15.55),
    "DRS Kaolack":     (14.15, -16.07),
    "DRS Kedougou":    (12.55, -12.18),
    "DRS Kolda":       (12.89, -14.94),
    "DRS Louga":       (15.62, -16.22),
    "DRS Matam":       (15.66, -13.26),
    "DRS Saint-Louis": (16.03, -16.49),
    "DRS Sedhiou":     (12.71, -15.56),
    "DRS Tambacounda": (13.77, -13.67),
    "DRS Thies":       (14.79, -16.93),
    "DRS Ziguinchor":  (12.58, -16.27),
}


# ------------------------------------------------------------------
# Chargement (mis en cache par Streamlit)
# ------------------------------------------------------------------

@st.cache_resource(show_spinner="Chargement du modèle…")
def get_model():
    return P.load_model()


@st.cache_data(show_spinner="Chargement de l'historique…")
def get_history() -> pd.DataFrame:
    return P.load_history()


@st.cache_data
def get_drs_list() -> list[str]:
    return sorted(get_history()["DRS"].dropna().unique().tolist())


def short_name(drs: str) -> str:
    """'DRS Dakar' -> 'Dakar'"""
    return drs.replace("DRS ", "")


def history_range() -> tuple[pd.Timestamp, pd.Timestamp]:
    h = get_history()
    return h["PERIODE"].min(), h["PERIODE"].max()


# ------------------------------------------------------------------
# Météo : climatologie, année de référence, import CSV
# ------------------------------------------------------------------

@st.cache_data
def climatology(drs: str) -> pd.DataFrame:
    """Moyenne mensuelle historique de chaque variable météo pour une DRS.
    Index = mois (1..12)."""
    h = get_history()
    sub = h[h["DRS"] == drs]
    return sub.groupby(sub["PERIODE"].dt.month)[WEATHER_VARS].mean()


def weather_from_climatology(drs: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Construit un DataFrame météo futur à partir de la climatologie."""
    clim = climatology(drs)
    rows = [clim.loc[d.month].to_dict() | {"PERIODE": d} for d in dates]
    return pd.DataFrame(rows)[["PERIODE"] + WEATHER_VARS]


def weather_from_reference_year(drs: str, dates: pd.DatetimeIndex, year: int) -> pd.DataFrame:
    """Reprend, mois par mois, la météo observée d'une année historique."""
    h = get_history()
    sub = h[(h["DRS"] == drs) & (h["PERIODE"].dt.year == year)]
    by_month = sub.set_index(sub["PERIODE"].dt.month)[WEATHER_VARS]
    rows = [by_month.loc[d.month].to_dict() | {"PERIODE": d} for d in dates]
    return pd.DataFrame(rows)[["PERIODE"] + WEATHER_VARS]


def parse_weather_csv(uploaded, drs: str | None = None) -> pd.DataFrame:
    """Lit un CSV/XLSX météo importé. Colonnes attendues : PERIODE + WEATHER_VARS
    (et éventuellement une colonne DRS pour filtrer)."""
    name = getattr(uploaded, "name", "").lower()
    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded)
    else:
        df = pd.read_csv(uploaded, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]
    if "DRS" in df.columns and drs is not None:
        df = df[df["DRS"] == drs]
    if "PERIODE" not in df.columns:
        raise ValueError("Le fichier doit contenir une colonne 'PERIODE'.")
    df["PERIODE"] = pd.to_datetime(df["PERIODE"]).dt.to_period("M").dt.to_timestamp()
    missing = [c for c in WEATHER_VARS if c not in df.columns]
    if missing:
        raise ValueError("Colonnes météo manquantes : " + ", ".join(missing))
    return df[["PERIODE"] + WEATHER_VARS].sort_values("PERIODE").reset_index(drop=True)


@dataclass
class WeatherAdjust:
    """Ajustements de scénario appliqués à une météo de base."""
    precip_pct: float = 0.0     # % de variation des précipitations
    temp_delta: float = 0.0     # °C ajoutés aux 3 températures
    humid_delta: float = 0.0    # points d'humidité ajoutés
    wind_pct: float = 0.0
    solar_pct: float = 0.0

    def is_neutral(self) -> bool:
        return all(v == 0 for v in vars(self).values())

    def apply(self, w: pd.DataFrame) -> pd.DataFrame:
        w = w.copy()
        w["precipitation"] = (w["precipitation"] * (1 + self.precip_pct / 100)).clip(lower=0)
        for c in ["temperature_moyenne", "temperature_max", "temperature_min"]:
            w[c] = w[c] + self.temp_delta
        w["humidite"] = (w["humidite"] + self.humid_delta).clip(0, 100)
        w["vent"] = (w["vent"] * (1 + self.wind_pct / 100)).clip(lower=0)
        w["rayonnement_solaire"] = (w["rayonnement_solaire"] * (1 + self.solar_pct / 100)).clip(lower=0)
        return w


# ------------------------------------------------------------------
# Prévision
# ------------------------------------------------------------------

def month_range(start: pd.Timestamp, n_months: int) -> pd.DatetimeIndex:
    start = pd.Timestamp(start).to_period("M").to_timestamp()
    return pd.date_range(start, periods=n_months, freq="MS")


def forecast_one(drs: str, dates: pd.DatetimeIndex, weather: pd.DataFrame,
                 history: pd.DataFrame | None = None) -> pd.DataFrame:
    """Appelle le pipeline de prévision de l'équipe pour une DRS."""
    return P.predict_region(
        drs=drs,
        start_date=dates[0],
        end_date=dates[-1],
        weather_df=weather,
        model=get_model(),
        history=get_history() if history is None else history,
    )


def forecast_many(drs_list: list[str], dates: pd.DatetimeIndex,
                  weather_fn, history: pd.DataFrame | None = None) -> pd.DataFrame:
    """weather_fn(drs) -> DataFrame météo. Retourne les prévisions concaténées."""
    out = []
    for d in drs_list:
        out.append(forecast_one(d, dates, weather_fn(d), history))
    return pd.concat(out, ignore_index=True)


# ------------------------------------------------------------------
# Évaluation (rétro-prévision)
# ------------------------------------------------------------------

def _mape(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    mask = y > 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y[mask] - yhat[mask]) / y[mask])) * 100)


def metrics(y, yhat) -> dict:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    err = yhat - y
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) if len(y) > 1 else np.nan
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE (%)": _mape(y, yhat),
        "R²": 1 - ss_res / ss_tot if ss_tot and ss_tot > 0 else np.nan,
        "Biais": float(np.mean(err)),
    }


@st.cache_data(show_spinner="Rétro-prévision en cours…")
def backtest(drs: str, cutoff: pd.Timestamp, horizon: int) -> pd.DataFrame:
    """Prévoit récursivement `horizon` mois après `cutoff` en n'utilisant que
    l'historique <= cutoff, avec la météo réellement observée, puis compare
    aux cas observés. Retourne PERIODE, observé, prévu."""
    h = get_history()
    cutoff = pd.Timestamp(cutoff).to_period("M").to_timestamp()
    dates = pd.date_range(cutoff + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    dates = dates[dates <= h["PERIODE"].max()]
    if len(dates) == 0:
        return pd.DataFrame(columns=["PERIODE", "observe", "prevu"])
    sub = h[h["DRS"] == drs]
    weather = sub[sub["PERIODE"].isin(dates)][["PERIODE"] + WEATHER_VARS]
    truncated = h[h["PERIODE"] <= cutoff]
    fc = forecast_one(drs, dates, weather, history=truncated)
    obs = sub[sub["PERIODE"].isin(dates)][["PERIODE", TARGET]].rename(columns={TARGET: "observe"})
    return fc.merge(obs, on="PERIODE").rename(columns={"prediction": "prevu"})[["PERIODE", "observe", "prevu"]]


# ------------------------------------------------------------------
# Explicabilité
# ------------------------------------------------------------------

@st.cache_data
def feature_importances() -> pd.DataFrame:
    model = get_model()
    names = model.named_steps["preprocessor"].get_feature_names_out()
    imp = model.named_steps["model"].feature_importances_
    df = pd.DataFrame({"feature": names, "importance": imp})
    df["famille"] = df["feature"].map(_feature_family)
    df["feature"] = (df["feature"].str.replace("num__", "", regex=False)
                     .str.replace("cat__DRS_DRS ", "Région : ", regex=False))
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def _feature_family(name: str) -> str:
    if name.startswith("cat__"):
        return "Région (one-hot)"
    n = name.replace("num__", "")
    if n.startswith("target_lag") or n.startswith("target_roll"):
        return "Historique des cas (lags / moyennes glissantes)"
    if n in WEATHER_VARS:
        return "Météo"
    return "Calendrier"


def model_summary() -> dict:
    model = get_model()
    lgb = model.named_steps["model"]
    keys = ["n_estimators", "learning_rate", "max_depth", "num_leaves",
            "subsample", "objective", "random_state"]
    return {k: lgb.get_params()[k] for k in keys}


# ------------------------------------------------------------------
# Export
# ------------------------------------------------------------------

def to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)
    return buf.getvalue()


def label_period(ts: pd.Timestamp) -> str:
    return f"{MOIS_FR[ts.month - 1]} {ts.year}"
