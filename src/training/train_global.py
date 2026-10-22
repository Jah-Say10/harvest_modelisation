# ============================================================
# python -m training.train_global
# Global (national, monthly) forecasting model
# Target: R20_DSME_Femmes enceintes ayant reçu TPI 2
#
# Models searched: RandomForest, XGBoost (small grids)
# Compared against two naive baselines (last month, same month last year)
# ============================================================

import json
import logging
import pickle
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from tqdm.auto import tqdm
from xgboost import XGBRegressor

from utils.sort_year import sort_and_replace

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "../data/features/Base_PEC_MALARIA_CS_2021_2024_with_weather.xlsx"

REGISTRY_DIR = Path("../models/registry")
REGISTRY_DIR_META = Path("../models/metadata")
LOG_DIR = REGISTRY_DIR_META / "logs"
for d in (REGISTRY_DIR, REGISTRY_DIR_META, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

VERSION = "1.0.0"
TAG = "global_tpi2"
RANDOM_STATE = 42

target = "R20_DSME_Femmes enceintes ayant reçu TPI 2"

meteo_vars = [
    "temperature_moyenne", "temperature_max", "temperature_min",
    "precipitation", "humidite", "vent", "rayonnement_solaire",
]

# Same aggregation as your script. NOTE: precipitation is summed over rows
# (facilities), so it grows with the number of reporting rows. "mean" is
# usually safer - switch if you want.
agg_dict = {
    target: "sum",
    "temperature_moyenne": "mean",
    "temperature_max": "mean",
    "temperature_min": "mean",
    "precipitation": "sum",
    "humidite": "mean",
    "vent": "mean",
    "rayonnement_solaire": "mean",
}

# Only ~48 monthly points exist, so keep features light:
# lag 12 would cost 12 more rows; seasonality is carried by month_sin/cos.
LAGS = [1, 2, 3, 6]
TEST_MONTHS = 12        # last 12 months of the series are held out
CV_SPLITS = 3
XGB_N_JOBS = 4          # CPU is faster than GPU for ~30 rows

np.random.seed(RANDOM_STATE)

SEARCH_SPACES = {
    "RandomForest": {          # 2*3*3*3 = 54 combos
        "estimator": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        "param_grid": {
            "n_estimators": [300, 600],
            "max_depth": [None, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": [0.5, 1.0, "sqrt"],
        },
    },
    "XGBoost": {               # 2*3*2*2*2 = 48 combos
        "estimator": XGBRegressor(
            random_state=RANDOM_STATE, objective="reg:squarederror",
            tree_method="hist", device="cpu", n_jobs=XGB_N_JOBS,
        ),
        "param_grid": {
            "n_estimators": [200, 400],
            "max_depth": [2, 3, 4],
            "learning_rate": [0.03, 0.1],
            "min_child_weight": [1, 3],
            "subsample": [0.8, 1.0],
        },
    },
}

# ============================================================
# LOGGING
# ============================================================

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"train_{TAG}_{VERSION}_{run_id}.log"

logger = logging.getLogger("global")
logger.setLevel(logging.INFO)
logger.handlers.clear()
fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, encoding="utf-8")):
    h.setFormatter(fmt)
    logger.addHandler(h)


def fmt_secs(s):
    s = int(s)
    return f"{s // 3600:d}h{(s % 3600) // 60:02d}m{s % 60:02d}s"


# ============================================================
# 1. LOAD + AGGREGATE
# ============================================================

logger.info("=" * 70)
logger.info("STEP 1/5 - Load and aggregate by month")

data = pd.read_excel(DATA_PATH)
logger.info(f"Raw data: {data.shape}")

data_agg = data.groupby("PERIODE", as_index=False).agg(agg_dict)

df = sort_and_replace(data_agg)
df["PERIODE"] = pd.to_datetime(df["PERIODE"])
df = df.sort_values("PERIODE").reset_index(drop=True)

logger.info(f"Monthly series: {df.shape} | {df['PERIODE'].min():%Y-%m} -> {df['PERIODE'].max():%Y-%m}")

# Lags assume consecutive months: warn if a month is missing
months = df["PERIODE"].dt.to_period("M")
gaps = (months.diff().dropna().apply(lambda x: x.n) != 1).sum()
if gaps:
    logger.warning(f"{gaps} gap(s) in the monthly series - lag features will be misaligned there!")
if df[target].isna().any():
    logger.warning(f"{df[target].isna().sum()} missing target value(s)")

# ============================================================
# 2. FEATURES
# ============================================================

logger.info("STEP 2/5 - Feature engineering")

feat = df.copy()
for lag in LAGS:
    feat[f"target_lag_{lag}"] = feat[target].shift(lag)
feat["target_roll_mean_3"] = feat[target].shift(1).rolling(3).mean()
feat["target_roll_mean_6"] = feat[target].shift(1).rolling(6).mean()
feat["month"] = feat["PERIODE"].dt.month
feat["month_sin"] = np.sin(2 * np.pi * feat["month"] / 12)
feat["month_cos"] = np.cos(2 * np.pi * feat["month"] / 12)

# Naive baselines (not model inputs)
feat["naive_last_month"] = feat[target].shift(1)
feat["naive_same_month_last_year"] = feat[target].shift(12)

feature_cols = (
    [c for c in meteo_vars if c in feat.columns]
    + [f"target_lag_{l}" for l in LAGS]
    + ["target_roll_mean_3", "target_roll_mean_6", "month", "month_sin", "month_cos"]
)

ml_df = feat.dropna(subset=feature_cols + [target]).reset_index(drop=True)

# ============================================================
# 3. TRAIN / TEST SPLIT (last TEST_MONTHS months held out)
# ============================================================

ml_train = ml_df.iloc[:-TEST_MONTHS].copy()
ml_test = ml_df.iloc[-TEST_MONTHS:].copy()
X_train, y_train = ml_train[feature_cols], ml_train[target]
X_test, y_test = ml_test[feature_cols], ml_test[target]

logger.info(f"STEP 3/5 - Split | train {X_train.shape} ({ml_train['PERIODE'].min():%Y-%m} -> {ml_train['PERIODE'].max():%Y-%m}) "
            f"| test {X_test.shape} ({ml_test['PERIODE'].min():%Y-%m} -> {ml_test['PERIODE'].max():%Y-%m})")
if len(X_train) < 20:
    logger.warning(f"Only {len(X_train)} training rows: expect noisy CV and results. Compare against the baselines below.")


# ============================================================
# METRICS
# ============================================================

def smape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    den = np.abs(y_true) + np.abs(y_pred)
    m = den != 0
    return np.nan if not m.any() else 100 * np.mean(2 * np.abs(y_pred[m] - y_true[m]) / den[m])


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    mask = y_true != 0
    mape = 100 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) if mask.any() else np.nan
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "sMAPE": float(smape(y_true, y_pred)),
        "MAPE": float(mape),
        "R2": float(r2_score(y_true, y_pred)),
    }


def log_metrics(label, m):
    logger.info(f"{label:<28} RMSE {m['RMSE']:.2f} | MAE {m['MAE']:.2f} | "
                f"sMAPE {m['sMAPE']:.1f}% | MAPE {m['MAPE']:.1f}% | R2 {m['R2']:.3f}")


# ============================================================
# 4. GRID SEARCH
# ============================================================

def grid_search(name, estimator, param_grid, X, y, cv):
    combos = list(ParameterGrid(param_grid))
    n = len(combos)
    splits = list(cv.split(X))
    logger.info(f"{name}: {n} combos x {len(splits)} folds = {n * len(splits)} fits")

    best_rmse, best_params = np.inf, None
    t0 = time.time()
    log_every = max(1, n // 10)

    with tqdm(total=n, desc=name, unit="combo", dynamic_ncols=True) as pbar:
        for i, params in enumerate(combos, 1):
            fold_rmse = []
            for tr, va in splits:
                m = clone(estimator).set_params(**params)
                m.fit(X.iloc[tr], y.iloc[tr])
                pred = np.maximum(m.predict(X.iloc[va]), 0)
                fold_rmse.append(np.sqrt(mean_squared_error(y.iloc[va], pred)))
            cv_rmse = float(np.mean(fold_rmse))
            if cv_rmse < best_rmse:
                best_rmse, best_params = cv_rmse, params
                logger.info(f"{name} new best CV RMSE {best_rmse:.2f} (combo {i}/{n})")

            elapsed = time.time() - t0
            eta = elapsed / i * (n - i)
            pbar.set_postfix(best=f"{best_rmse:.1f}", eta=fmt_secs(eta))
            pbar.update(1)
            if i % log_every == 0 or i == n:
                logger.info(f"{name} {i}/{n} ({100 * i / n:.0f}%) | best CV RMSE {best_rmse:.2f} "
                            f"| elapsed {fmt_secs(elapsed)} | ETA {fmt_secs(eta)}")
    return best_params, best_rmse


logger.info("=" * 70)
logger.info("STEP 4/5 - Grid search")

cv = TimeSeriesSplit(n_splits=CV_SPLITS)
t_global = time.time()
results = {}

for idx, (name, spec) in enumerate(SEARCH_SPACES.items(), 1):
    logger.info(f"--- Model {idx}/{len(SEARCH_SPACES)}: {name} ---")
    best_params, cv_rmse = grid_search(name, spec["estimator"], spec["param_grid"], X_train, y_train, cv)
    logger.info(f"{name} best params: {best_params}")

    model = clone(spec["estimator"]).set_params(**best_params).fit(X_train, y_train)
    preds = np.maximum(model.predict(X_test), 0)
    m = metrics(y_test, preds)
    log_metrics(f"{name} TEST", m)
    results[name] = {"model": model, "preds": preds, "metrics": m, "cv_rmse": cv_rmse, "best_params": best_params}

# Baselines
baselines = {}
for label, col in [("Naive: last month", "naive_last_month"),
                   ("Naive: same month last year", "naive_same_month_last_year")]:
    if ml_test[col].notna().all():
        baselines[label] = metrics(y_test, ml_test[col])
        log_metrics(label, baselines[label])

best_name = min(results, key=lambda k: results[k]["metrics"]["RMSE"])
best = results[best_name]
logger.info(f"BEST MODEL: {best_name} (test RMSE {best['metrics']['RMSE']:.2f})")
for label, bm in baselines.items():
    if best["metrics"]["RMSE"] >= bm["RMSE"]:
        logger.warning(f"The model does NOT beat '{label}' (RMSE {bm['RMSE']:.2f}). Treat results with caution.")

# ============================================================
# 5. PLOTS + SAVE
# ============================================================

logger.info("=" * 70)
logger.info("STEP 5/5 - Plots, importance, saving")

plot_dir = REGISTRY_DIR / f"Plot_{TAG}_{datetime.now():%Y-%m-%d}"
plot_dir.mkdir(parents=True, exist_ok=True)

# Predictions plot
plt.figure(figsize=(14, 7))
plt.plot(df["PERIODE"], df[target], color="black", lw=2, label="Historical")
plt.plot(ml_test["PERIODE"], y_test, color="royalblue", lw=2.5, marker="o", label="Test actual")
plt.plot(ml_test["PERIODE"], best["preds"], color="red", lw=2.5, ls="--", marker="o", label=f"{best_name} prediction")
if "naive_same_month_last_year" in ml_test and ml_test["naive_same_month_last_year"].notna().all():
    plt.plot(ml_test["PERIODE"], ml_test["naive_same_month_last_year"], color="green", lw=1.5, ls=":", label="Same month last year")
plt.axvline(ml_test["PERIODE"].min(), color="gray", ls=":", lw=2)
plt.title(f"Global model - {best_name}", fontsize=16, fontweight="bold")
plt.xlabel("PERIODE")
plt.ylabel(target)
plt.grid(True, alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig(plot_dir / f"{TAG}_{best_name}_predictions.png", dpi=200, bbox_inches="tight")
plt.close()

# Permutation importance (on test set)
logger.info("Computing permutation importance")
imp = permutation_importance(best["model"], X_test, y_test, n_repeats=20, random_state=RANDOM_STATE,
                             scoring="neg_root_mean_squared_error")
imp_df = (pd.DataFrame({"feature": feature_cols, "importance": imp.importances_mean, "std": imp.importances_std})
          .sort_values("importance", ascending=False).reset_index(drop=True))
logger.info("Top features:\n" + imp_df.head(8).to_string(index=False))

top = imp_df.head(12).iloc[::-1]
plt.figure(figsize=(10, 6))
plt.barh(top["feature"], top["importance"], xerr=top["std"])
plt.title("Permutation importance (RMSE increase when shuffled)")
plt.tight_layout()
plt.savefig(plot_dir / f"{TAG}_{best_name}_importance.png", dpi=200, bbox_inches="tight")
plt.close()

# Save model
model_path = REGISTRY_DIR / f"best_ml_model_{TAG}_{VERSION}.pkl"
with open(model_path, "wb") as f:
    pickle.dump(best["model"], f)
logger.info(f"Model saved: {model_path}")

# Refit on all available data for deployment
X_all = pd.concat([X_train, X_test])
y_all = pd.concat([y_train, y_test])
full_model = clone(best["model"]).fit(X_all, y_all)
full_model_path = REGISTRY_DIR / f"best_ml_model_{TAG}_{VERSION}_full.pkl"
with open(full_model_path, "wb") as f:
    pickle.dump(full_model, f)
logger.info(f"Full-data model saved: {full_model_path}")

# Metadata
summary = {
    "version": VERSION,
    "tag": TAG,
    "trained_at": datetime.now().isoformat(),
    "duration_seconds": round(time.time() - t_global, 1),
    "target": target,
    "best_ml_model": best_name,
    "best_params": best["best_params"],
    "model_path": str(model_path),
    "full_model_path": str(full_model_path),
    "feature_columns": feature_cols,
    "weather_variables": meteo_vars,
    "aggregation": agg_dict,
    "lags": LAGS,
    "train_period": {"start": str(ml_train["PERIODE"].min()), "end": str(ml_train["PERIODE"].max()), "rows": len(ml_train)},
    "test_period": {"start": str(ml_test["PERIODE"].min()), "end": str(ml_test["PERIODE"].max()), "rows": len(ml_test)},
    "best_ml_metrics": best["metrics"],
    "baselines": baselines,
    "all_ml_results": [
        {"model": n, "cv_rmse": r["cv_rmse"], **r["metrics"], "best_params": r["best_params"]}
        for n, r in results.items()
    ],
    "feature_importance": imp_df.to_dict(orient="records"),
}
meta_path = REGISTRY_DIR_META / f"summary_{TAG}_{VERSION}.json"
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, default=str)
logger.info(f"Metadata saved: {meta_path}")

logger.info("=" * 70)
logger.info(f"GLOBAL TRAINING COMPLETE in {fmt_secs(time.time() - t_global)} | log: {log_file}")