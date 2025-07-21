"""
final_project_ML_model.py
────────────────────────────────────────────────────────────────────────────
Builds two models on Data/model/model_data_full.csv:

1. Ridge regression baseline (predict next‑day % move magnitude).
2. XGBoost classifier (predict next‑day direction) with
   hyper‑parameter search, early stopping, threshold optimisation,
   evaluation plots, learning‑curve scatter, and feature importance.

Plots are styled to match the sample scatter & loss curves provided.

Updated: 2025‑07‑20
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from xgboost import XGBClassifier

# ═════════════════════ Paths / Config ═════════════════════════════
ROOT = Path(__file__).resolve().parents[1]
DATA_MODEL = ROOT / "Data" / "model" / "model_data_full.csv"
MODEL_OUT = ROOT / "model" / "xgb_tsla_final.json"
THRESH_OUT = ROOT / "model" / "xgb_threshold.json"

CONFIG: Dict[str, Any] = {
    "RANDOM_STATE": 42,
    "MAX_JOBS": 8,
    "BASE_XGB": dict(
        n_estimators=3000,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
    ),
    "HPARAM_SPACE": {
        "learning_rate": uniform(0.001, 0.5),
        "max_depth": randint(3, 10),
        "min_child_weight": randint(3, 25),
        "subsample": uniform(0.7, 0.3),
        "colsample_bytree": uniform(0.7, 0.3),
        "gamma": uniform(0.0, 5.0),
        "reg_lambda": uniform(0.0, 5.0),
    },
    "CV_SPLITS": 4,
    "CV_TEST_SIZE": 0.15,
    "EARLY_STOP": 100,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)5s │ %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def banner(msg: str) -> None:
    line = "═" * 100
    log.info(line)
    log.info(msg)
    log.info(line)


# ════════════════════ Feature engineering ════════════════════════
NUMERIC_EXCLUDE = {"next_day_pct_change"}


def add_alpha_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds simple lag/momentum/RSI style features."""
    ret = df["next_day_pct_change"]

    for k in (1, 2, 3, 5, 10):
        df[f"ret_lag_{k}d"] = ret.shift(k)

    df["vol_10d"] = ret.rolling(10).std()
    df["ma_5_20"] = ret.rolling(5).mean() / ret.rolling(20).mean()

    if "tweet_sentiment" in df.columns:
        ts = df["tweet_sentiment"]
        df["sent_diff"] = ts - ts.rolling(3).mean()

    df["mom_5d"] = ret.rolling(5).sum()

    delta = ret
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-12)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    return df


def load_full() -> pd.DataFrame:
    df = pd.read_csv(DATA_MODEL, parse_dates=["date"])
    df = add_alpha_features(df)
    return df.dropna(subset=["next_day_pct_change"]).reset_index(drop=True)


def feature_target(
        df: pd.DataFrame, *, direction: bool
) -> Tuple[pd.DataFrame, pd.Series]:
    """Returns X, y with highly‑correlated numeric cols (>0.95) dropped."""
    y = (
        (df["next_day_pct_change"] > 0).astype(int)
        if direction
        else df["next_day_pct_change"]
    )
    num_cols = [
        c
        for c in df.select_dtypes("number").columns
        if c not in NUMERIC_EXCLUDE
    ]
    corr = df[num_cols].corr().abs()
    to_drop = [
        c
        for c in num_cols
        if any(corr[c] > 0.95) and not c.endswith("ret_lag_1d")
    ]
    return df[num_cols].drop(columns=to_drop), y


# ═══════════════ Ridge baseline (regression) ══════════════════════
def ridge_regression(df: pd.DataFrame, *, split: float) -> None:
    banner("📊 Ridge baseline – predicting % change")
    X, y = feature_target(df, direction=False)
    X = X.dropna()
    y = y.loc[X.index]

    idx = int(split * len(X))
    X_tr, y_tr = X.iloc[:idx], y.iloc[:idx]
    X_te, y_te = X.iloc[idx:], y.iloc[idx:]

    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

    grid = RandomizedSearchCV(
        Ridge(random_state=CONFIG["RANDOM_STATE"]),
        {"alpha": np.logspace(-4, 2, 25)},
        cv=TimeSeriesSplit(CONFIG["CV_SPLITS"]),
        n_iter=20,
        scoring="neg_root_mean_squared_error",
        n_jobs=CONFIG["MAX_JOBS"],
        random_state=CONFIG["RANDOM_STATE"],
    ).fit(X_tr_s, y_tr)

    best_alpha = grid.best_params_["alpha"]
    preds = grid.predict(X_te_s)
    rmse = np.sqrt(mean_squared_error(y_te, preds))
    log.info("Best α = %g   RMSE = %.5f   R² = %.4f",
             best_alpha, rmse, r2_score(y_te, preds))

    # ── Styled scatter plot (matches sample) ──────────────────────
    plt.figure(figsize=(6, 5))
    plt.scatter(y_te, preds,
                color="#1f77b4", alpha=0.65, s=25,
                label="Predicted vs Actual", edgecolors="none")
    lims = [min(y_te.min(), preds.min()), max(y_te.max(), preds.max())]
    plt.plot(lims, lims, "r--", linewidth=2, label="Perfect Fit")
    plt.title("Ridge Regression – Actual vs Predicted", fontsize=13)
    plt.xlabel("Actual Values");
    plt.ylabel("Predicted Values")
    plt.grid(True, linestyle="--", alpha=.4)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ═══════════ XGBoost classifier (direction) ═══════════════════════
def xgb_direction(df: pd.DataFrame, *, split: float, n_iter: int) -> None:
    banner("🚀 XGBoost – predicting next‑day direction")
    X, y = feature_target(df, direction=True)

    idx_test = int(split * len(df))
    idx_val = int(idx_test * 0.9)
    X_train, y_train = X.iloc[:idx_val], y.iloc[:idx_val]
    X_val, y_val = X.iloc[idx_val:idx_test], y.iloc[idx_val:idx_test]
    X_test, y_test = X.iloc[idx_test:], y.iloc[idx_test:]

    base_params = dict(CONFIG["BASE_XGB"],
                       random_state=CONFIG["RANDOM_STATE"],
                       n_jobs=CONFIG["MAX_JOBS"])

    cv = TimeSeriesSplit(
        n_splits=CONFIG["CV_SPLITS"],
        test_size=int(CONFIG["CV_TEST_SIZE"] * len(X_train)),
    )

    search = RandomizedSearchCV(
        XGBClassifier(**base_params),
        CONFIG["HPARAM_SPACE"],
        n_iter=n_iter,
        cv=cv,
        scoring="f1",
        n_jobs=CONFIG["MAX_JOBS"],
        random_state=CONFIG["RANDOM_STATE"],
        verbose=0,
    ).fit(X_train, y_train)

    best_params = {k: (v.item() if isinstance(v, np.generic) else v)
                   for k, v in search.best_params_.items()}
    log.info("Best hyper‑parameters: %s", best_params)

    model = XGBClassifier(**base_params | best_params)
    fit_kwargs = dict(
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )

    try:  # modern callback API
        fit_kwargs["callbacks"] = [
            xgb.callback.EarlyStopping(rounds=CONFIG["EARLY_STOP"])
        ]
        model.fit(X_train, y_train, **fit_kwargs)
    except TypeError:
        try:  # legacy early_stopping_rounds arg
            fit_kwargs.pop("callbacks", None)
            fit_kwargs["early_stopping_rounds"] = CONFIG["EARLY_STOP"]
            model.fit(X_train, y_train, **fit_kwargs)
        except TypeError:  # very old wrapper
            fit_kwargs.pop("early_stopping_rounds", None)
            model.fit(X_train, y_train, **fit_kwargs)

    # ── Scatter learning curve (matches sample) ───────────────────
    evals = model.evals_result()
    if evals and "validation_0" in evals:
        train_loss = evals["validation_0"]["logloss"]
        val_loss = evals["validation_1"]["logloss"]
        epochs = range(len(train_loss))

        plt.figure(figsize=(7, 5))
        plt.scatter(epochs, train_loss, s=30, label="Training Loss")
        plt.scatter(epochs, val_loss, s=30, label="Validation Loss")
        plt.xlabel("Epoch");
        plt.ylabel("Log‑loss")
        plt.title("XGBoost Training and Validation Loss Over Epochs")
        plt.grid(True, linestyle="--", alpha=.4)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # ── Threshold optimisation ────────────────────────────────────
    prob_val = model.predict_proba(X_val)[:, 1]
    p, r, t = precision_recall_curve(y_val, prob_val)
    best_tau = float(t[int((2 * p * r / (p + r + 1e-12)).argmax())])

    # ── Evaluation on hold‑out test ───────────────────────────────
    prob = model.predict_proba(X_test)[:, 1]
    y_hat = (prob >= best_tau).astype(int)

    log.info("τ ≈ %.3f   F1 = %.3f   Acc = %.3f   AUC = %.3f",
             best_tau,
             f1_score(y_test, y_hat),
             accuracy_score(y_test, y_hat),
             roc_auc_score(y_test, prob))

    RocCurveDisplay.from_predictions(y_test, prob)
    plt.title("ROC — XGBoost");
    plt.tight_layout();
    plt.show()

    PrecisionRecallDisplay.from_predictions(y_test, prob)
    plt.title("PR — XGBoost");
    plt.tight_layout();
    plt.show()

    ConfusionMatrixDisplay.from_predictions(
        y_test, y_hat, cmap="viridis", colorbar=False)
    plt.title("Confusion Matrix — XGBoost");
    plt.tight_layout();
    plt.show()

    # ── Feature importance ────────────────────────────────────────
    importance = model.get_booster().get_score(importance_type="gain")
    if importance:
        (pd.Series(importance)
         .sort_values(ascending=False)
         .head(20)[::-1]
         .plot.barh(color="#1f77b4"))
        plt.title("Top‑20 Feature Importance (Gain)")
        plt.tight_layout()
        plt.show()

    # ── Persist model & threshold ─────────────────────────────────
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_OUT)
    with open(THRESH_OUT, "w") as f:
        json.dump({"tau": best_tau}, f)

    log.info("✅ model saved → %s", MODEL_OUT)
    log.info("✅ threshold saved → %s", THRESH_OUT)


# ═════════════════════ CLI / main ═════════════════════════════════
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=float, default=0.7,
                    help="train/test split ratio (chronological)")
    ap.add_argument("--search-iter", type=int, default=400,
                    help="randomised‑search iterations for XGB")
    return ap.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    args = parse_args()

    banner("🏁 PIPELINE START — reading model_data_full.csv")
    df = load_full()
    banner(f"Dataset loaded → {len(df):,} daily rows")

    ridge_regression(df, split=args.split)
    xgb_direction(df, split=args.split, n_iter=args.search_iter)

    banner("✅ PIPELINE FINISHED")


if __name__ == "__main__":
    main()
