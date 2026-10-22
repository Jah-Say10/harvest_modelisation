# ============================================================
# python -m training.train_region_gpu
#
# REGIONAL MALARIA FORECASTING - 3 MODELS (ML only)
#   1. confirmed  : confirmed cases (TDR, outpatient)
#   2. screening  : DSDOM positive TDR (female + male)
#   3. stock      : ACT stock received
#
# Models searched: RandomForest, XGBoost (reduced grid)
# Logs: console + file in ../models/metadata/logs/
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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from tqdm.auto import tqdm
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
matplotlib.use("Agg")  # plots are saved to disk, never block the run

# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "../data/features/Base_MALARIA_CS_CLEAN_2021_2024_with_weather.xlsx"
DATA_PATH_TEST = "../data/features/Base_MALARIA_CS_CLEAN_2025_with_weather.xlsx"

REGISTRY_DIR = Path("../models/registry")
REGISTRY_DIR_META = Path("../models/metadata")
LOG_DIR = REGISTRY_DIR_META / "logs"
for d in (REGISTRY_DIR, REGISTRY_DIR_META, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

VERSION = "1.0.0"
RANDOM_STATE = 42
LAGS = [1, 2, 3, 6, 12]
CV_SPLITS = 3

# With only a few hundred rows (DRS x month) the GPU is often SLOWER than
# the CPU for XGBoost (transfer overhead). Try "cpu" and compare timings.
XGB_DEVICE = "cuda"          # "cuda" or "cpu"
XGB_N_JOBS = 4

# Also save a copy of the best model refit on train + 2025 (for deployment)
REFIT_ON_ALL = True

np.random.seed(RANDOM_STATE)

# Targets: key -> column used (target2 is built as female + male)
COL_T1 = "Cas Confirmés (par TDR) palu consultations externes"
COL_T2_F = "TDR positifs DSDOM Feminin"
COL_T2_H = "TDR positifs DSDOM Masculin"
COL_T2 = "TDR positifs DSDOM Total"          # created below
COL_T3 = "Palu:ACT NRSS-Stock reçu"

TARGETS = {
    "confirmed": COL_T1,
    "screening": COL_T2,
    "stock": COL_T3,
}

METEO_VARS = [
    "temperature_moyenne", "temperature_max", "temperature_min",
    "precipitation", "humidite", "vent", "rayonnement_solaire",
]

# ============================================================
# REDUCED SEARCH SPACES
#   RF  : 2*3*3*3       =  54 combos x 3 folds = 162 fits
#   XGB : 2*2*2*2*2*2   =  64 combos x 3 folds = 192 fits
# ============================================================

SEARCH_SPACES = {
    "RandomForest": {
        "estimator": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        "param_grid": {
            "model__n_estimators": [300, 600],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": [0.5, 1.0, "sqrt"],
        },
    },
    "XGBoost": {
        "estimator": XGBRegressor(
            random_state=RANDOM_STATE,
            objective="reg:squarederror",
            tree_method="hist",
            device=XGB_DEVICE,
            n_jobs=XGB_N_JOBS,
        ),
        "param_grid": {
            "model__n_estimators": [300, 600],
            "model__max_depth": [3, 5],
            "model__learning_rate": [0.03, 0.1],
            "model__min_child_weight": [1, 5],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0],
        },
    },
}

# ============================================================
# LOGGING
# ============================================================

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"train_{VERSION}_{run_id}.log"

logger = logging.getLogger("malaria")
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
# GPU CHECK
# ============================================================

try:
    import torch
    if torch.cuda.is_available():
        logger.info(f"GPU detected: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})")
    else:
        logger.warning("No CUDA GPU detected -> set XGB_DEVICE = 'cpu'")
except ImportError:
    logger.info("PyTorch not installed; GPU detection left to XGBoost.")

# ============================================================
# 1. LOAD DATA
# ============================================================

logger.info("=" * 70)
logger.info("STEP 1/4 - Loading data")
train_raw = pd.read_excel(DATA_PATH)
test_raw = pd.read_excel(DATA_PATH_TEST)
for raw in (train_raw, test_raw):
    raw[COL_T2] = raw[COL_T2_F].fillna(0) + raw[COL_T2_H].fillna(0)
logger.info(f"Train raw: {train_raw.shape} | Test raw: {test_raw.shape}")


# ============================================================
# HELPERS
# ============================================================

def aggregate(raw, target):
    agg = {target: "sum", **{c: "mean" for c in METEO_VARS if c in raw.columns}}
    df = raw.groupby(["DRS", "PERIODE"]).agg(agg).reset_index()
    df["PERIODE"] = pd.to_datetime(df["PERIODE"])
    return df.sort_values(["DRS", "PERIODE"]).reset_index(drop=True)


def create_ml_features(df, target):
    data = df.sort_values(["DRS", "PERIODE"]).reset_index(drop=True)
    g = data.groupby("DRS")[target]
    for lag in LAGS:
        data[f"target_lag_{lag}"] = g.shift(lag)
    data["target_roll_mean_3"] = g.transform(lambda x: x.shift(1).rolling(3).mean())
    data["target_roll_mean_6"] = g.transform(lambda x: x.shift(1).rolling(6).mean())
    data["month"] = data["PERIODE"].dt.month
    data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12)
    data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12)
    return data


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
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "sMAPE": smape(y_true, y_pred),
        "MAPE": mape,
        "R2": r2_score(y_true, y_pred),
    }


def period_splits(df, n_splits):
    """
    Time-based CV splits on UNIQUE PERIODS (all DRS of a month stay together).
    A plain TimeSeriesSplit on rows sorted by DRS would split by region, not by time.
    """
    periods = np.sort(df["PERIODE"].unique())
    for tr_p, va_p in TimeSeriesSplit(n_splits=n_splits).split(periods):
        tr_idx = np.where(df["PERIODE"].isin(periods[tr_p]))[0]
        va_idx = np.where(df["PERIODE"].isin(periods[va_p]))[0]
        yield tr_idx, va_idx


def grid_search(name, pipeline, param_grid, X, y, splits, tag):
    """Manual grid search so we can log real progress + ETA."""
    combos = list(ParameterGrid(param_grid))
    n = len(combos)
    logger.info(f"[{tag}] {name}: {n} combos x {len(splits)} folds = {n * len(splits)} fits")

    best_rmse, best_params = np.inf, None
    t0 = time.time()
    log_every = max(1, n // 10)

    with tqdm(total=n, desc=f"{tag} | {name}", unit="combo", dynamic_ncols=True) as pbar:
        for i, params in enumerate(combos, 1):
            fold_rmse = []
            for tr_idx, va_idx in splits:
                model = clone(pipeline).set_params(**params)
                model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
                pred = np.maximum(model.predict(X.iloc[va_idx]), 0)
                fold_rmse.append(np.sqrt(mean_squared_error(y.iloc[va_idx], pred)))
            cv_rmse = float(np.mean(fold_rmse))
            if cv_rmse < best_rmse:
                best_rmse, best_params = cv_rmse, params
                logger.info(f"[{tag}] {name} new best CV RMSE {best_rmse:.3f} (combo {i}/{n})")

            elapsed = time.time() - t0
            eta = elapsed / i * (n - i)
            pbar.set_postfix(best=f"{best_rmse:.2f}", eta=fmt_secs(eta))
            pbar.update(1)
            if i % log_every == 0 or i == n:
                logger.info(
                    f"[{tag}] {name} {i}/{n} ({100 * i / n:.0f}%) | "
                    f"best CV RMSE {best_rmse:.3f} | elapsed {fmt_secs(elapsed)} | ETA {fmt_secs(eta)}"
                )
    return best_params, best_rmse


def plot_predictions(full_df, test_df, preds, model_name, target, tag, test_start, out_dir):
    plot_test = test_df[["DRS", "PERIODE", target]].copy()
    plot_test["prediction"] = np.asarray(preds)
    for region in plot_test["DRS"].dropna().unique():
        hist = full_df[full_df["DRS"] == region].sort_values("PERIODE")
        tr = plot_test[plot_test["DRS"] == region].sort_values("PERIODE")
        plt.figure(figsize=(14, 7))
        plt.plot(hist["PERIODE"], hist[target], color="black", lw=2, label="Historical")
        plt.plot(tr["PERIODE"], tr[target], color="royalblue", lw=2.5, marker="o", ms=5, label="Test actual")
        plt.plot(tr["PERIODE"], tr["prediction"], color="red", lw=2.5, ls="--", marker="o", ms=5,
                 label=f"{model_name} prediction")
        plt.axvline(test_start, color="gray", ls=":", lw=2, label="Test start")
        plt.axvspan(test_start, hist["PERIODE"].max(), color="orange", alpha=0.08)
        plt.title(f"[{tag}] {model_name} — {region}", fontsize=16, fontweight="bold")
        plt.xlabel("PERIODE")
        plt.ylabel(target)
        plt.grid(True, alpha=0.25)
        plt.legend(loc="best")
        plt.tight_layout()
        safe = str(region).replace("/", "_").replace("\\", "_").replace(" ", "_")
        plt.savefig(out_dir / f"{model_name}_{safe}.png", dpi=200, bbox_inches="tight")
        plt.close()


# ============================================================
# 2. TRAIN ONE TARGET
# ============================================================

def run_target(tag, target):
    t_start = time.time()
    logger.info("=" * 70)
    logger.info(f"TARGET '{tag}' -> {target}")
    logger.info("=" * 70)

    # --- aggregate + features ---------------------------------
    logger.info(f"[{tag}] Aggregating by DRS/month and building features")
    train_df = aggregate(train_raw, target)
    test_df = aggregate(test_raw, target)
    train_end = train_df["PERIODE"].max()
    test_start, test_end = test_df["PERIODE"].min(), test_df["PERIODE"].max()

    # Concatenate so early-2025 lags can look back into 2024
    all_df = pd.concat([train_df, test_df], ignore_index=True).sort_values(["DRS", "PERIODE"]).reset_index(drop=True)
    ml_df = create_ml_features(all_df, target)

    # "year" removed: trees cannot extrapolate to an unseen year (2025)
    feature_cols = (
        ["DRS"]
        + [c for c in METEO_VARS if c in ml_df.columns]
        + [f"target_lag_{l}" for l in LAGS]
        + ["target_roll_mean_3", "target_roll_mean_6", "month", "month_sin", "month_cos"]
    )
    ml_df = ml_df.dropna(subset=feature_cols + [target]).copy()

    ml_train = ml_df[ml_df["PERIODE"] <= train_end].sort_values(["PERIODE", "DRS"]).reset_index(drop=True)
    ml_test = ml_df[(ml_df["PERIODE"] >= test_start) & (ml_df["PERIODE"] <= test_end)].reset_index(drop=True)
    X_train, y_train = ml_train[feature_cols], ml_train[target]
    X_test, y_test = ml_test[feature_cols], ml_test[target]
    logger.info(f"[{tag}] Train {X_train.shape} ({ml_train['PERIODE'].min():%Y-%m} -> {ml_train['PERIODE'].max():%Y-%m}) | "
                f"Test {X_test.shape} ({ml_test['PERIODE'].min():%Y-%m} -> {ml_test['PERIODE'].max():%Y-%m})")

    categorical = ["DRS"]
    numeric = [c for c in feature_cols if c not in categorical]
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", "passthrough", numeric),
    ])

    splits = list(period_splits(ml_train, CV_SPLITS))

    # --- search both models -----------------------------------
    results = {}
    for name, spec in SEARCH_SPACES.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", clone(spec["estimator"]))])
        best_params, cv_rmse = grid_search(name, pipe, spec["param_grid"], X_train, y_train, splits, tag)

        logger.info(f"[{tag}] {name} best params: {best_params}")
        logger.info(f"[{tag}] {name} refitting best model on full train + evaluating on test")
        best = clone(pipe).set_params(**best_params).fit(X_train, y_train)
        preds = np.maximum(best.predict(X_test), 0)
        m = metrics(y_test, preds)
        logger.info(f"[{tag}] {name} TEST | RMSE {m['RMSE']:.3f} | MAE {m['MAE']:.3f} | "
                    f"sMAPE {m['sMAPE']:.2f}% | MAPE {m['MAPE']:.2f}% | R2 {m['R2']:.3f}")
        results[name] = {"estimator": best, "preds": preds, "metrics": m,
                         "cv_rmse": cv_rmse, "best_params": best_params}

    # --- select best by test RMSE -----------------------------
    best_name = min(results, key=lambda k: results[k]["metrics"]["RMSE"])
    best = results[best_name]
    logger.info(f"[{tag}] BEST MODEL: {best_name} (test RMSE {best['metrics']['RMSE']:.3f})")

    # --- plots -------------------------------------------------
    plot_dir = REGISTRY_DIR / f"Plot_{tag}_{datetime.now():%Y-%m-%d}"
    plot_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[{tag}] Saving plots to {plot_dir}")
    plot_predictions(all_df, ml_test, best["preds"], best_name, target, tag, test_start, plot_dir)

    # --- save model(s) ----------------------------------------
    model_path = REGISTRY_DIR / f"best_ml_model_{tag}_{VERSION}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(best["estimator"], f)
    logger.info(f"[{tag}] Model saved: {model_path}")

    full_model_path = None
    if REFIT_ON_ALL:
        X_all = pd.concat([X_train, X_test], ignore_index=True)
        y_all = pd.concat([y_train, y_test], ignore_index=True)
        full_model = clone(best["estimator"]).fit(X_all, y_all)
        full_model_path = REGISTRY_DIR / f"best_ml_model_{tag}_{VERSION}_full.pkl"
        with open(full_model_path, "wb") as f:
            pickle.dump(full_model, f)
        logger.info(f"[{tag}] Full-data model saved: {full_model_path}")

    # --- save metadata ----------------------------------------
    summary = {
        "version": VERSION,
        "tag": tag,
        "trained_at": datetime.now().isoformat(),
        "duration_seconds": round(time.time() - t_start, 1),
        "target": target,
        "best_ml_model": best_name,
        "best_params": best["best_params"],
        "model_path": str(model_path),
        "full_model_path": str(full_model_path) if full_model_path else None,
        "feature_columns": feature_cols,
        "categorical_features": categorical,
        "numeric_features": numeric,
        "weather_variables": METEO_VARS,
        "lags": LAGS,
        "xgb_device": XGB_DEVICE,
        "train_period": {"start": str(ml_train["PERIODE"].min()), "end": str(ml_train["PERIODE"].max())},
        "test_period": {"start": str(ml_test["PERIODE"].min()), "end": str(ml_test["PERIODE"].max())},
        "best_ml_metrics": {k: float(v) for k, v in best["metrics"].items()},
        "all_ml_results": [
            {"model": n, "cv_rmse": float(r["cv_rmse"]), **{k: float(v) for k, v in r["metrics"].items()},
             "best_params": r["best_params"]}
            for n, r in results.items()
        ],
    }
    meta_path = REGISTRY_DIR_META / f"summary_{tag}_{VERSION}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"[{tag}] Metadata saved: {meta_path}")
    logger.info(f"[{tag}] Done in {fmt_secs(time.time() - t_start)}")
    return summary


# ============================================================
# 3. RUN ALL THREE MODELS
# ============================================================

logger.info("=" * 70)
logger.info(f"STEP 2/4 - Training {len(TARGETS)} models | log file: {log_file}")

global_t0 = time.time()
all_summaries = []
for i, (tag, target) in enumerate(TARGETS.items(), 1):
    logger.info(f"########## MODEL {i}/{len(TARGETS)}: {tag} ##########")
    all_summaries.append(run_target(tag, target))
    elapsed = time.time() - global_t0
    logger.info(f"Overall progress: {i}/{len(TARGETS)} models | elapsed {fmt_secs(elapsed)} | "
                f"ETA {fmt_secs(elapsed / i * (len(TARGETS) - i))}")

# ============================================================
# 4. GLOBAL SUMMARY
# ============================================================

logger.info("=" * 70)
logger.info("STEP 4/4 - Final summary")
rows = [{"model": s["tag"], "best": s["best_ml_model"], **s["best_ml_metrics"]} for s in all_summaries]
logger.info("\n" + pd.DataFrame(rows).to_string(index=False))

with open(REGISTRY_DIR_META / f"summary_all_{VERSION}.json", "w", encoding="utf-8") as f:
    json.dump({"version": VERSION, "trained_at": datetime.now().isoformat(),
               "total_duration_seconds": round(time.time() - global_t0, 1),
               "models": all_summaries}, f, indent=2, default=str)

logger.info(f"ALL TRAINING COMPLETE in {fmt_secs(time.time() - global_t0)}")