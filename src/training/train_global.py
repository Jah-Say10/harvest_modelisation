# ============================================================
# python -m training.train_global
#
# GLOBAL (NATIONAL, MONTHLY) MALARIA FORECASTING - 3 MODELS
#   1. confirmed  : confirmed cases (TDR, outpatient)
#   2. screening  : DSDOM positive TDR (female + male)
#   3. stock      : ACT stock received
#
# Same modelisation as train_region_gpu.py, but all DRS are
# summed into ONE national series per month (no DRS feature).
# Models searched: RandomForest, XGBoost (reduced grid)
# Baselines: last month, same month last year
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

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

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
LAGS = [1, 2, 3, 6, 12]     # drop 12 if you want more training rows (only ~36 remain)
CV_SPLITS = 3

# ~36 training rows: CPU is faster than GPU for XGBoost at this size
XGB_DEVICE = "cpu"          # "cuda" or "cpu"
XGB_N_JOBS = 4

REFIT_ON_ALL = True

np.random.seed(RANDOM_STATE)

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

SEARCH_SPACES = {
    "RandomForest": {
        "estimator": RandomForestRegressor(
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "param_grid": {
            "model__n_estimators": [300, 600, 900],
            "model__max_depth": [None, 10, 20, 30],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": [0.3, 0.5, 1.0],
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
            "model__n_estimators": [300, 600, 900],
            "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.01, 0.03, 0.1],
            "model__min_child_weight": [1, 3, 5],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__colsample_bytree": [0.7, 0.85, 1.0],
            "model__reg_lambda": [1, 5],
        },
    },
}

# ============================================================
# LOGGING
# ============================================================

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"train_global_{VERSION}_{run_id}.log"

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
# 1. LOAD DATA
# ============================================================

logger.info("=" * 70)
logger.info("STEP 1/3 - Loading data")
train_raw = pd.read_excel(DATA_PATH)
test_raw = pd.read_excel(DATA_PATH_TEST)
for raw in (train_raw, test_raw):
    raw[COL_T2] = raw[COL_T2_F].fillna(0) + raw[COL_T2_H].fillna(0)
logger.info(f"Train raw: {train_raw.shape} | Test raw: {test_raw.shape}")


# ============================================================
# HELPERS
# ============================================================

def aggregate_national(raw, target):
    """One row per month: target summed over all DRS, weather averaged."""
    agg = {target: "sum", **{c: "mean" for c in METEO_VARS if c in raw.columns}}
    df = raw.groupby("PERIODE").agg(agg).reset_index()
    df["PERIODE"] = pd.to_datetime(df["PERIODE"])
    return df.sort_values("PERIODE").reset_index(drop=True)


def create_ml_features(df, target):
    data = df.sort_values("PERIODE").reset_index(drop=True)
    s = data[target]
    for lag in LAGS:
        data[f"target_lag_{lag}"] = s.shift(lag)
    data["target_roll_mean_3"] = s.shift(1).rolling(3).mean()
    data["target_roll_mean_6"] = s.shift(1).rolling(6).mean()
    data["month"] = data["PERIODE"].dt.month
    data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12)
    data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12)
    # baselines (not model inputs)
    data["naive_last_month"] = s.shift(1)
    data["naive_same_month_last_year"] = s.shift(12)
    return data


def smape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    den = np.abs(y_true) + np.abs(y_pred)
    m = den != 0
    return np.nan if not m.any() else 100 * np.mean(2 * np.abs(y_pred[m] - y_true[m]) / den[m])


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mask = y_true != 0
    mape = 100 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) if mask.any() else np.nan
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "sMAPE": float(smape(y_true, y_pred)),
        "MAPE": float(mape),
        "R2": float(r2_score(y_true, y_pred)),
    }


def log_metrics(tag, label, m):
    logger.info(f"[{tag}] {label:<28} RMSE {m['RMSE']:.2f} | MAE {m['MAE']:.2f} | "
                f"sMAPE {m['sMAPE']:.1f}% | MAPE {m['MAPE']:.1f}% | R2 {m['R2']:.3f}")


def grid_search(name, estimator, param_grid, X, y, cv, tag):
    combos = list(ParameterGrid(param_grid))
    n = len(combos)
    splits = list(cv.split(X))
    logger.info(f"[{tag}] {name}: {n} combos x {len(splits)} folds = {n * len(splits)} fits")

    best_rmse, best_params = np.inf, None
    t0 = time.time()
    log_every = max(1, n // 10)

    with tqdm(total=n, desc=f"{tag} | {name}", unit="combo", dynamic_ncols=True) as pbar:
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
                logger.info(f"[{tag}] {name} new best CV RMSE {best_rmse:.2f} (combo {i}/{n})")

            elapsed = time.time() - t0
            eta = elapsed / i * (n - i)
            pbar.set_postfix(best=f"{best_rmse:.1f}", eta=fmt_secs(eta))
            pbar.update(1)
            if i % log_every == 0 or i == n:
                logger.info(f"[{tag}] {name} {i}/{n} ({100 * i / n:.0f}%) | best CV RMSE {best_rmse:.2f} "
                            f"| elapsed {fmt_secs(elapsed)} | ETA {fmt_secs(eta)}")
    return best_params, best_rmse


# ============================================================
# 2. TRAIN ONE TARGET
# ============================================================

def run_target(tag, target):
    t_start = time.time()
    logger.info("=" * 70)
    logger.info(f"TARGET '{tag}' (national) -> {target}")
    logger.info("=" * 70)

    train_df = aggregate_national(train_raw, target)
    test_df = aggregate_national(test_raw, target)
    train_end = train_df["PERIODE"].max()
    test_start, test_end = test_df["PERIODE"].min(), test_df["PERIODE"].max()
    logger.info(f"[{tag}] National series: train {len(train_df)} months, test {len(test_df)} months")

    # Concatenate so early-2025 lags can look back into 2024
    all_df = pd.concat([train_df, test_df], ignore_index=True).sort_values("PERIODE").reset_index(drop=True)
    months = all_df["PERIODE"].dt.to_period("M")
    if (months.diff().dropna().apply(lambda x: x.n) != 1).any():
        logger.warning(f"[{tag}] Gap(s) in the monthly series - lag features will be misaligned!")

    ml_df = create_ml_features(all_df, target)
    feature_cols = (
        [c for c in METEO_VARS if c in ml_df.columns]
        + [f"target_lag_{l}" for l in LAGS]
        + ["target_roll_mean_3", "target_roll_mean_6", "month", "month_sin", "month_cos"]
    )
    ml_df = ml_df.dropna(subset=feature_cols + [target]).reset_index(drop=True)

    ml_train = ml_df[ml_df["PERIODE"] <= train_end].copy()
    ml_test = ml_df[(ml_df["PERIODE"] >= test_start) & (ml_df["PERIODE"] <= test_end)].copy()
    X_train, y_train = ml_train[feature_cols], ml_train[target]
    X_test, y_test = ml_test[feature_cols], ml_test[target]
    logger.info(f"[{tag}] Train {X_train.shape} ({ml_train['PERIODE'].min():%Y-%m} -> {ml_train['PERIODE'].max():%Y-%m}) | "
                f"Test {X_test.shape} ({ml_test['PERIODE'].min():%Y-%m} -> {ml_test['PERIODE'].max():%Y-%m})")

    cv = TimeSeriesSplit(n_splits=CV_SPLITS)

    # --- search both models -----------------------------------
    results = {}
    for name, spec in SEARCH_SPACES.items():
        best_params, cv_rmse = grid_search(name, spec["estimator"], spec["param_grid"], X_train, y_train, cv, tag)
        logger.info(f"[{tag}] {name} best params: {best_params}")
        model = clone(spec["estimator"]).set_params(**best_params).fit(X_train, y_train)
        preds = np.maximum(model.predict(X_test), 0)
        m = metrics(y_test, preds)
        log_metrics(tag, f"{name} TEST", m)
        results[name] = {"model": model, "preds": preds, "metrics": m, "cv_rmse": cv_rmse, "best_params": best_params}

    # --- baselines --------------------------------------------
    baselines = {}
    for label, col in [("Naive: last month", "naive_last_month"),
                       ("Naive: same month last year", "naive_same_month_last_year")]:
        if ml_test[col].notna().all():
            baselines[label] = metrics(y_test, ml_test[col])
            log_metrics(tag, label, baselines[label])

    best_name = min(results, key=lambda k: results[k]["metrics"]["RMSE"])
    best = results[best_name]
    logger.info(f"[{tag}] BEST MODEL: {best_name} (test RMSE {best['metrics']['RMSE']:.2f})")
    for label, bm in baselines.items():
        if best["metrics"]["RMSE"] >= bm["RMSE"]:
            logger.warning(f"[{tag}] Model does NOT beat '{label}' (RMSE {bm['RMSE']:.2f})")

    # --- plots -------------------------------------------------
    plot_dir = REGISTRY_DIR / f"Plot_global_{tag}_{datetime.now():%Y-%m-%d}"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(14, 7))
    plt.plot(all_df["PERIODE"], all_df[target], color="black", lw=2, label="Historical")
    plt.plot(ml_test["PERIODE"], y_test, color="royalblue", lw=2.5, marker="o", label="Test actual")
    plt.plot(ml_test["PERIODE"], best["preds"], color="red", lw=2.5, ls="--", marker="o", label=f"{best_name} prediction")
    if ml_test["naive_same_month_last_year"].notna().all():
        plt.plot(ml_test["PERIODE"], ml_test["naive_same_month_last_year"], color="green", lw=1.5, ls=":",
                 label="Same month last year")
    plt.axvline(test_start, color="gray", ls=":", lw=2)
    plt.axvspan(test_start, all_df["PERIODE"].max(), color="orange", alpha=0.08)
    plt.title(f"[{tag}] Global model - {best_name}", fontsize=16, fontweight="bold")
    plt.xlabel("PERIODE")
    plt.ylabel(target)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / f"{tag}_{best_name}_predictions.png", dpi=200, bbox_inches="tight")
    plt.close()

    logger.info(f"[{tag}] Computing permutation importance")
    imp = permutation_importance(best["model"], X_test, y_test, n_repeats=20, random_state=RANDOM_STATE,
                                 scoring="neg_root_mean_squared_error")
    imp_df = (pd.DataFrame({"feature": feature_cols, "importance": imp.importances_mean, "std": imp.importances_std})
              .sort_values("importance", ascending=False).reset_index(drop=True))
    logger.info(f"[{tag}] Top features:\n" + imp_df.head(6).to_string(index=False))
    top = imp_df.head(12).iloc[::-1]
    plt.figure(figsize=(10, 6))
    plt.barh(top["feature"], top["importance"], xerr=top["std"])
    plt.title(f"[{tag}] Permutation importance (RMSE increase when shuffled)")
    plt.tight_layout()
    plt.savefig(plot_dir / f"{tag}_{best_name}_importance.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"[{tag}] Plots saved to {plot_dir}")

    # --- save models ------------------------------------------
    model_path = REGISTRY_DIR / f"best_ml_model_global_{tag}_{VERSION}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(best["model"], f)
    logger.info(f"[{tag}] Model saved: {model_path}")

    full_model_path = None
    if REFIT_ON_ALL:
        full_model = clone(best["model"]).fit(pd.concat([X_train, X_test]), pd.concat([y_train, y_test]))
        full_model_path = REGISTRY_DIR / f"best_ml_model_global_{tag}_{VERSION}_full.pkl"
        with open(full_model_path, "wb") as f:
            pickle.dump(full_model, f)
        logger.info(f"[{tag}] Full-data model saved: {full_model_path}")

    # --- metadata ---------------------------------------------
    summary = {
        "version": VERSION,
        "scope": "global",
        "tag": tag,
        "trained_at": datetime.now().isoformat(),
        "duration_seconds": round(time.time() - t_start, 1),
        "target": target,
        "best_ml_model": best_name,
        "best_params": best["best_params"],
        "model_path": str(model_path),
        "full_model_path": str(full_model_path) if full_model_path else None,
        "feature_columns": feature_cols,
        "weather_variables": METEO_VARS,
        "lags": LAGS,
        "xgb_device": XGB_DEVICE,
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
    meta_path = REGISTRY_DIR_META / f"summary_global_{tag}_{VERSION}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"[{tag}] Metadata saved: {meta_path}")
    logger.info(f"[{tag}] Done in {fmt_secs(time.time() - t_start)}")
    return summary


# ============================================================
# 3. RUN ALL THREE MODELS
# ============================================================

logger.info("=" * 70)
logger.info(f"STEP 2/3 - Training {len(TARGETS)} global models | log file: {log_file}")

global_t0 = time.time()
all_summaries = []
for i, (tag, target) in enumerate(TARGETS.items(), 1):
    logger.info(f"########## GLOBAL MODEL {i}/{len(TARGETS)}: {tag} ##########")
    all_summaries.append(run_target(tag, target))
    elapsed = time.time() - global_t0
    logger.info(f"Overall progress: {i}/{len(TARGETS)} models | elapsed {fmt_secs(elapsed)} | "
                f"ETA {fmt_secs(elapsed / i * (len(TARGETS) - i))}")

logger.info("=" * 70)
logger.info("STEP 3/3 - Final summary")
rows = [{"model": s["tag"], "best": s["best_ml_model"], **s["best_ml_metrics"]} for s in all_summaries]
logger.info("\n" + pd.DataFrame(rows).to_string(index=False))

with open(REGISTRY_DIR_META / f"summary_global_all_{VERSION}.json", "w", encoding="utf-8") as f:
    json.dump({"version": VERSION, "scope": "global", "trained_at": datetime.now().isoformat(),
               "total_duration_seconds": round(time.time() - global_t0, 1),
               "models": all_summaries}, f, indent=2, default=str)

logger.info(f"ALL GLOBAL TRAINING COMPLETE in {fmt_secs(time.time() - global_t0)}")