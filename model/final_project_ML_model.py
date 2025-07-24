"""
model/final_project_ML_model.py
────────────────────────────────────────────────────────────────────────────
Pipeline:

1. Ridge regression baseline.
2. XGBoost classifier (early‑stopping grid search, **no SMOTE**).
3. LightGBM classifier + Conditional‑SMOTE + GridSearchCV.

Conditional‑SMOTE is applied only in the LightGBM branch, and only for
days with social activity.
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
from imblearn.base import BaseSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
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
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from xgboost import XGBClassifier

import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="X does not have valid feature names, but LGBMClassifier was fitted"
)

# ═════════════════════ Paths / Config ═════════════════════════════
ROOT = Path(__file__).resolve().parents[1]
DATA_MODEL = ROOT / "Data" / "model" / "model_data_full.csv"
MODEL_OUT = ROOT / "model" / "xgb_tsla_final.json"
LGB_OUT = ROOT / "model" / "lgb_tsla_final.txt"
THRESH_OUT = ROOT / "model" / "xgb_threshold.json"

CONFIG: Dict[str, Any] = {
    "RANDOM_STATE": 42,
    "MAX_JOBS": 8,
    # XGB discrete grid
    "XGB_PARAM_GRID": {
        "clf__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "clf__max_depth": [3, 5, 7],
        "clf__min_child_weight": [3, 7, 15],
        "clf__subsample": [0.8, 1.0],
        "clf__colsample_bytree": [0.8, 1.0],
        "clf__gamma": [0, 1, 3],
        "clf__reg_lambda": [0.5, 1.0, 2.0],
    },
    # LightGBM grid
    "LGB_PARAM_GRID": {
        "clf__learning_rate": [0.01, 0.05, 0.1],
        "clf__num_leaves": [31, 63, 127],
        "clf__max_depth": [-1, 6, 10],
        "clf__subsample": [0.8, 1.0],
        "clf__colsample_bytree": [0.8, 1.0],
    },
    "CV_SPLITS": 4,
    "CV_TEST_SIZE": 0.15,
    "EARLY_STOP": 50,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)5s │ %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def banner(msg: str) -> None:
    bar = "═" * 100
    log.info(bar)
    log.info(msg)
    log.info(bar)


def log_smote_stats(name: str, y_before: pd.Series, y_after: np.ndarray) -> None:
    """Print class counts before / after SMOTE and how many rows were added."""
    before = np.bincount(y_before, minlength=2)
    after = np.bincount(y_after, minlength=2)
    added = len(y_after) - len(y_before)
    log.info(
        "%s SMOTE ⇒ rows %d → %d  (+%d synthetic)  class0 %d→%d  class1 %d→%d",
        name, len(y_before), len(y_after), added, before[0], after[0], before[1], after[1],
    )


# ════════════════════ Conditional‑SMOTE ══════════════════════════
class ConditionalSMOTE(BaseSampler):
    """
    SMOTE applied only to rows where post_count_tweet + post_count_retweet > 0.
    Provides both mandatory attributes:
        • _parameter_constraints  (sklearn ≥1.2)
        • sampling_strategy       (imblearn ≥0.12)
    """
    _sampling_type = "over-sampling"
    _parameter_constraints: dict = {}  # satisfies sklearn param‑checker

    def __init__(self, sampling_strategy: str | float | dict = "auto",
                 **smote_kwargs):
        # required by BaseSampler
        self.sampling_strategy = sampling_strategy
        self.smote_kwargs = smote_kwargs

    # imblearn calls _fit_resample
    def _fit_resample(self, X, y):
        X_df = pd.DataFrame(X).reset_index(drop=True)
        y_s = pd.Series(y).reset_index(drop=True)

        if {"post_count_tweet", "post_count_retweet"}.issubset(X_df.columns):
            mask = (X_df["post_count_tweet"] + X_df["post_count_retweet"]) > 0
        else:
            mask = pd.Series(True, index=X_df.index)

        sm = SMOTE(
            sampling_strategy=self.sampling_strategy,
            random_state=CONFIG["RANDOM_STATE"],
            **self.smote_kwargs,
        )
        X_res, y_res = sm.fit_resample(X_df[mask], y_s[mask])

        X_final = pd.concat([pd.DataFrame(X_res), X_df[~mask]], ignore_index=True)
        y_final = pd.concat([pd.Series(y_res), y_s[~mask]], ignore_index=True)
        return X_final.values, y_final.values


# ═════════════════ Feature engineering & load ════════════════════
NUMERIC_EXCLUDE = {"next_day_pct_change"}


def add_alpha_features(df: pd.DataFrame) -> pd.DataFrame:
    ret = df["next_day_pct_change"]
    for k in (1, 2, 3, 5, 10):
        df[f"ret_lag_{k}d"] = ret.shift(k)

    df["vol_10d"] = ret.rolling(10).std()
    df["ma_5_20"] = ret.rolling(5).mean() / ret.rolling(20).mean()
    df["mom_5d"] = ret.rolling(5).sum()

    delta = ret
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi_14"] = 100 - (100 / (1 + gain / (loss + 1e-12)))
    return df


def load_full() -> pd.DataFrame:
    df = pd.read_csv(DATA_MODEL, parse_dates=["date"])
    df = add_alpha_features(df)
    return df.dropna(subset=["next_day_pct_change"]).reset_index(drop=True)


def feature_target(
        df: pd.DataFrame, *, direction: bool
) -> Tuple[pd.DataFrame, pd.Series]:
    """Return X, y with:
       · y = next‑day % or direction;
       · X = top‑variance numeric columns (max 150) excluding the target."""
    y = (
        (df["next_day_pct_change"] > 0).astype(int)
        if direction
        else df["next_day_pct_change"]
    )

    # All numeric predictors except the target itself
    num_cols = [
        c for c in df.select_dtypes("number").columns
        if c not in NUMERIC_EXCLUDE
    ]

    # Remove highly collinear columns (>0.95) to stabilise trees
    corr = df[num_cols].corr().abs()
    to_drop = [c for c in num_cols if any(corr[c] > 0.95) and not c.endswith("ret_lag_1d")]
    num_cols = [c for c in num_cols if c not in to_drop]

    # Keep top‑variance 150 (or fewer) – helps with small sample size
    top_var = (
        df[num_cols].var()
        .sort_values(ascending=False)
        .head(150)
        .index
    )
    return df[top_var], y


# ═══════════════ Ridge baseline (regression) ══════════════════════
def ridge_regression(df: pd.DataFrame, *, split: float) -> None:
    """
    Fit a Ridge model that predicts the *magnitude* of next‑day % change
    and show a square scatter plot with equal x/y limits so deviations
    from the 45‑degree line are visually meaningful.
    """
    banner("📊 Ridge baseline – predicting % change")
    X, y = feature_target(df, direction=False)

    # drop rows that still contain NaNs
    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask], y[mask]

    # chronological split
    idx = int(split * len(X))
    X_tr, y_tr = X.iloc[:idx], y.iloc[:idx]
    X_te, y_te = X.iloc[idx:], y.iloc[idx:]

    # scale → Ridge(α) grid search
    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

    grid = GridSearchCV(
        Ridge(random_state=CONFIG["RANDOM_STATE"]),
        {"alpha": np.logspace(-4, 2, 25)},
        cv=TimeSeriesSplit(CONFIG["CV_SPLITS"]),
        scoring="neg_root_mean_squared_error",
        n_jobs=CONFIG["MAX_JOBS"],
    ).fit(X_tr_s, y_tr)

    best_alpha = grid.best_params_["alpha"]
    preds = grid.predict(X_te_s)
    rmse = np.sqrt(mean_squared_error(y_te, preds))
    r2 = r2_score(y_te, preds)
    log.info("Best α = %g   RMSE = %.5f   R² = %.4f", best_alpha, rmse, r2)

    # ── Improved scatter plot ─────────────────────────────────────
    plt.figure(figsize=(6, 6))
    lim_lo = min(y_te.min(), preds.min())
    lim_hi = max(y_te.max(), preds.max())
    lims = (np.floor(lim_lo) - 1, np.ceil(lim_hi) + 1)

    plt.scatter(
        y_te, preds,
        alpha=0.35, s=20, color="#1f77b4", edgecolors="k",
        label="Predicted vs Actual",
    )
    plt.plot(lims, lims, "r--", lw=1.2, label="Perfect Fit")

    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel("Actual % change")
    plt.ylabel("Predicted % change")
    plt.title(f"Ridge α={best_alpha}   RMSE={rmse:.2f}   R²={r2:.3f}")
    plt.grid(ls="--", alpha=.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ══════════════ Common evaluator for classifiers ═════════════════
def eval_clf(model, X_val, y_val, X_test, y_test,
             name: str, save_path: Path | None = None) -> float:
    prob_val = model.predict_proba(X_val)[:, 1]
    p, r, t = precision_recall_curve(y_val, prob_val)
    tau = float(t[int((2 * p * r / (p + r + 1e-12)).argmax())])

    prob = model.predict_proba(X_test)[:, 1]
    y_hat = (prob >= tau).astype(int)

    f1 = f1_score(y_test, y_hat)
    log.info("%s τ=%.3f  F1=%.3f  Acc=%.3f  AUC=%.3f",
             name, tau, f1,
             accuracy_score(y_test, y_hat),
             roc_auc_score(y_test, prob))

    RocCurveDisplay.from_predictions(y_test, prob)
    plt.title(f"ROC — {name}")
    plt.tight_layout()
    plt.show()

    PrecisionRecallDisplay.from_predictions(y_test, prob)
    plt.title(f"PR — {name}")
    plt.tight_layout()
    plt.show()

    ConfusionMatrixDisplay.from_predictions(
        y_test, y_hat, cmap="viridis", colorbar=False)
    plt.title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    plt.show()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(model.named_steps["clf"], XGBClassifier):
            model.named_steps["clf"].save_model(str(save_path))
        else:
            model.named_steps["clf"].booster_.save_model(str(save_path))
        with open(THRESH_OUT, "w") as f:
            json.dump({"tau": tau}, f)
        log.info("✅ model saved → %s", save_path)
    return f1


# ═══════════════════ XGB ═══════════════════
def xgb_direction(df: pd.DataFrame, *, split: float) -> Tuple[Pipeline, float]:
    """
    XGBoost direction classifier (no SMOTE).

    • hyper‑parameter grid on learning‑rate & tree params
    • model is *refitted* with an eval_set **after** GridSearchCV, so we don’t
      hit incompatibilities during cross‑validation.
    • learning‑curve plotted if eval metrics are available.
    """
    banner("🚀 XGBoost – grid search (learning‑rate tuned, no SMOTE)")

    # ── chronological split --------------------------------------------------
    X, y = feature_target(df, direction=True)
    idx_test, idx_val = int(split * len(df)), int(split * len(df) * 0.9)
    X_train, y_train = X.iloc[:idx_val], y.iloc[:idx_val]
    X_val, y_val = X.iloc[idx_val:idx_test], y.iloc[idx_val:idx_test]
    X_test, y_test = X.iloc[idx_test:], y.iloc[idx_test:]

    base_est = XGBClassifier(
        n_estimators=2000,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=CONFIG["RANDOM_STATE"],
        n_jobs=CONFIG["MAX_JOBS"],
        scale_pos_weight=1.0,  # dataset already ~balanced
    )

    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", base_est),
    ])

    cv = TimeSeriesSplit(
        n_splits=CONFIG["CV_SPLITS"],
        test_size=int(CONFIG["CV_TEST_SIZE"] * len(X_train)),
    )

    # ── pure grid‑search (NO early‑stop) -------------------------------------
    grid = GridSearchCV(
        pipe, CONFIG["XGB_PARAM_GRID"],
        cv=cv, scoring="f1", n_jobs=CONFIG["MAX_JOBS"]
    ).fit(X_train, y_train)

    model: Pipeline = grid.best_estimator_
    booster: XGBClassifier = model.named_steps["clf"]

    # ── refit booster with (train,val) eval_set ------------------------------
    import inspect
    supports_callbacks = "callbacks" in inspect.signature(booster.fit).parameters

    booster.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
        **({"callbacks": [xgb.callback.EarlyStopping(
            rounds=CONFIG["EARLY_STOP"], save_best=True, min_delta=0)]}
           if supports_callbacks else {})
    )

    # learning‑curve
    ev = booster.evals_result()
    if ev:
        tr, vl = ev["validation_0"]["logloss"], ev["validation_1"]["logloss"]
        plt.figure(figsize=(7, 5))
        plt.plot(tr, label="Training")
        plt.plot(vl, label="Validation")
        plt.ylabel("Log‑loss")
        plt.xlabel("Trees")  # ← clearer axis label
        plt.title("XGB Training & Validation Loss")
        plt.grid(ls="--", alpha=.4)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # replace pipeline’s estimator with the freshly‑refitted booster
    model.named_steps["clf"] = booster

    # ── evaluate & save ------------------------------------------------------
    f1 = eval_clf(
        model, X_val, y_val, X_test, y_test,
        "XGBoost", save_path=MODEL_OUT
    )
    return model, f1


# ════════════════ LightGBM + SMOTE Pipeline ═══════════════════════
def lgb_direction(df: pd.DataFrame, *, split: float) -> Tuple[Pipeline, float]:
    banner("⚡ LightGBM – grid search + Conditional‑SMOTE")
    X, y = feature_target(df, direction=True)

    # chronological split
    idx_test, idx_val = int(split * len(df)), int(split * len(df) * 0.9)
    X_train, y_train = X.iloc[:idx_val], y.iloc[:idx_val]
    X_val, y_val = X.iloc[idx_val:idx_test], y.iloc[idx_val:idx_test]
    X_test, y_test = X.iloc[idx_test:], y.iloc[idx_test:]

    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("smote", ConditionalSMOTE()),
        ("clf", LGBMClassifier(
            n_estimators=2000,
            objective="binary",
            random_state=CONFIG["RANDOM_STATE"],
            verbose=-1,
        )),
    ])

    cv = TimeSeriesSplit(
        n_splits=CONFIG["CV_SPLITS"],
        test_size=int(CONFIG["CV_TEST_SIZE"] * len(X_train)),
    )

    # one‑off SMOTE preview
    X_imp = SimpleImputer(strategy="median").fit_transform(X_train)
    X_sm, y_sm = ConditionalSMOTE().fit_resample(X_imp, y_train)
    log_smote_stats("LGB", y_train, y_sm)

    grid = GridSearchCV(
        pipe, CONFIG["LGB_PARAM_GRID"],
        cv=cv, scoring="f1", n_jobs=CONFIG["MAX_JOBS"]
    ).fit(X_train, y_train)

    model: Pipeline = grid.best_estimator_
    f1 = eval_clf(
        model, X_val, y_val, X_test, y_test,
        "LightGBM", save_path=LGB_OUT
    )
    return model, f1


# ═════════════════════ CLI / main ═════════════════════════════════
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=float, default=0.7,
                    help="train/test split ratio (chronological)")
    return ap.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    args = parse_args()

    banner("🏁 PIPELINE START — loading dataset")
    df = load_full()
    banner(f"Dataset loaded → {len(df):,} rows")

    ridge_regression(df, split=args.split)
    xgb_model, f1_xgb = xgb_direction(df, split=args.split)
    # lgb_model, f1_lgb = lgb_direction(df, split=args.split)

    banner("✅ PIPELINE FINISHED")
    # log.info("F1 scores — XGB: %.3f   LightGBM: %.3f", f1_xgb, f1_lgb)
    log.info("F1 scores — XGB: %.3f   LightGBM: %.3f", f1_xgb)


if __name__ == "__main__":
    main()