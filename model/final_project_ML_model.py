# # ─────────────────────────────────────────────────────────────────────────────
# # final_project_ML_model.py            (updated for model_data_full.csv)
# # ----------------------------------------------------------------------------
# # What this script does
# # ---------------------
# # 1.  Loads the daily‑level, **already‑cleaned** dataset produced by
# #     build_full_model_data.py   →  data/model/model_data_full.csv
# # 2.  Builds two models:
# #     • A *baseline* **Ridge regression** that predicts the *magnitude*
# #       of the next‑day % price move (regression).
# #     • A production **XGBoost classifier** that predicts only the
# #       *direction* (up / down) of the next‑day move.
# # 3.  Uses **chronological splitting** (time‑series safe) so no look‑ahead.
# # 4.  Hyper‑parameter search for XGB with early‑stopping and evaluation
# #     plots (ROC / PR / Confusion‑matrix).
# # 5.  Saves the trained XGBoost model to   model/xgb_tsla_final.json
# # ----------------------------------------------------------------------------
# # NOTE:  We no longer rebuild tweets or TF‑IDF here; the dataset already
# #        contains:
# #          - tweet aggregates / sentiment
# #          - daily_top5_tfidf_weight_sum (+ lag / lead)
# #          - stock technicals & next_day_pct_change
# #        That keeps this script fast and simple.
# # ─────────────────────────────────────────────────────────────────────────────
# from __future__ import annotations
# import argparse, logging, sys, warnings
# from pathlib import Path
# from typing import Dict, Any
#
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.linear_model import Ridge
# from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, RandomizedSearchCV
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import (
#     mean_squared_error, r2_score,
#     precision_recall_curve, accuracy_score, roc_auc_score,
#     ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
# )
# from scipy.stats import uniform, randint
#
# import xgboost as xgb
# from xgboost import XGBClassifier
#
# # ────────────────────────── Paths & Config ────────────────────────
# ROOT = Path(__file__).resolve().parents[1]  # project root
# DATA_MODEL = ROOT / "data" / "model" / "model_data_full.csv"
# MODEL_OUT = ROOT / "model" / "xgb_tsla_final.json"
#
# CONFIG: Dict[str, Any] = {
#     "RANDOM_STATE": 42,
#     "MAX_JOBS": 8,
#     # Base params for XGB before H‑param search
#     "BASE_XGB": dict(
#         n_estimators=1000,
#         objective="binary:logistic",
#         eval_metric="logloss",
#         tree_method="hist",
#     ),
#     # Hyper‑parameter search space for RandomizedSearchCV
#     "HPARAM_SPACE": {
#         "learning_rate": uniform(0.01, 0.19),
#         "max_depth": randint(3, 9),
#         "subsample": uniform(0.6, 0.4),
#         "colsample_bytree": uniform(0.6, 0.4),
#         "gamma": uniform(0.0, 4.0),
#         "reg_lambda": uniform(0.5, 4.5),
#     },
# }
#
# # ─────────────────────── Logging helper ───────────────────────────
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)6s] %(message)s",
#     handlers=[logging.StreamHandler(sys.stdout)]
# )
# log = logging.getLogger(__name__)
#
#
# def banner(msg: str) -> None:
#     log.info("=" * 100)
#     log.info(msg)
#     log.info("=" * 100)
#
#
# # ───────────────── Load dataset & basic prep ──────────────────────
# # Numeric columns we must NOT feed as features (target or text cols)
# NUMERIC_EXCLUDE = {
#     "next_day_pct_change",  # ← regression target
# }
#
#
# def load_full() -> pd.DataFrame:
#     """Read model_data_full.csv, add a lagged return feature, drop initial NaNs."""
#     df = pd.read_csv(DATA_MODEL, parse_dates=["date"])
#
#     # 1‑day lag of next_day_pct_change (helps XGB)
#     df["ret_lag_1d"] = df["next_day_pct_change"].shift(1)
#
#     # Drop rows where the regression target is NaN
#     return df.dropna(subset=["next_day_pct_change"]).reset_index(drop=True)
#
#
# def feature_target(df: pd.DataFrame, *, direction: bool):
#     """
#     Build X and y.
#     If direction=True → binary (up=1 / down=0) target.
#     Otherwise use raw % change.
#     """
#     y = (df["next_day_pct_change"] > 0).astype(int) if direction else df["next_day_pct_change"]
#     # Any numeric column that is *not* excluded is a feature
#     num_cols = [c for c in df.select_dtypes("number").columns
#                 if c not in NUMERIC_EXCLUDE]
#     X = df[num_cols]
#     return X, y
#
#
# # ───────────────────── Ridge baseline (regression) ───────────────
# def ridge_regression(df: pd.DataFrame, *, split: float) -> None:
#     """
#     Simple baseline: predict *magnitude* of next‑day return with Ridge.
#     We drop rows that still contain NaNs (mostly first few days due to MAs).
#     """
#     banner("📊  Ridge baseline (predicting % change)")
#     X, y = feature_target(df, direction=False)
#
#     # -- Drop any remaining NaNs so Ridge won't crash --
#     X = X.dropna()
#     y = y.loc[X.index]  # keep y aligned with X
#
#     # Chronological split (no shuffle)
#     idx = int(split * len(X))
#     X_tr, y_tr = X.iloc[:idx], y.iloc[:idx]
#     X_te, y_te = X.iloc[idx:], y.iloc[idx:]
#
#     # Scale → Ridge (with grid search over α)
#     scaler = StandardScaler().fit(X_tr)
#     X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)
#
#     grid = GridSearchCV(
#         Ridge(random_state=CONFIG["RANDOM_STATE"]),
#         {"alpha": [10 ** i for i in range(-4, 3)]},
#         cv=TimeSeriesSplit(5),
#         scoring="neg_root_mean_squared_error",
#         n_jobs=CONFIG["MAX_JOBS"],
#     ).fit(X_tr_s, y_tr)
#
#     best_alpha = grid.best_params_["alpha"]
#     preds = grid.predict(X_te_s)
#     rmse = np.sqrt(mean_squared_error(y_te, preds))
#     log.info("Best α = %g   RMSE = %.5f   R² = %.4f",
#              best_alpha, rmse, r2_score(y_te, preds))
#
#     plt.figure(figsize=(4, 4))
#     plt.scatter(y_te, preds, alpha=.5)
#     plt.axline((0, 0), slope=1, color="k", linestyle="--")
#     plt.title(f"Ridge α={best_alpha}")
#     plt.tight_layout()
#     plt.show()
#
#
# # ─────────────────────── XGBoost classifier ──────────────────────
# def xgb_direction(df: pd.DataFrame, *, split: float, n_iter: int) -> None:
#     """
#     XGBoost to predict *direction* (up / down).
#     Uses randomized search for hyper‑parameters with early stopping.
#     """
#     banner("🚀  XGBoost (direction)")
#     X, y = feature_target(df, direction=True)
#
#     # Chronological split
#     idx_test = int(split * len(df))
#     idx_val = int(idx_test * 0.9)  # last 10% of train acts as validation
#
#     X_train, y_train = X.iloc[:idx_val], y.iloc[:idx_val]
#     X_val, y_val = X.iloc[idx_val:idx_test], y.iloc[idx_val:idx_test]
#     X_test, y_test = X.iloc[idx_test:], y.iloc[idx_test:]
#
#     # Base params + class imbalance scaling
#     base_params = dict(CONFIG["BASE_XGB"],
#                        random_state=CONFIG["RANDOM_STATE"],
#                        n_jobs=CONFIG["MAX_JOBS"],
#                        scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1)
#                        )
#
#     # ── Randomized hyper‑parameter search ────────────────────────
#     search = RandomizedSearchCV(
#         XGBClassifier(**base_params),
#         CONFIG["HPARAM_SPACE"], n_iter=n_iter,
#         cv=TimeSeriesSplit(4), scoring="roc_auc",
#         n_jobs=CONFIG["MAX_JOBS"],
#         random_state=CONFIG["RANDOM_STATE"],
#         verbose=0
#     ).fit(X_train, y_train)
#
#     best_params = search.best_params_
#     log.info("Best hyper‑parameters: %s", best_params)
#
#     # ── Train final model with graceful fallback for early stopping ──
#     model = XGBClassifier(**base_params | best_params)
#
#     try:
#         # Modern API (>= 1.6) – preferred
#         model.fit(
#             X_train, y_train,
#             eval_set=[(X_train, y_train), (X_val, y_val)],
#             callbacks=[xgb.callback.EarlyStopping(rounds=50)],
#             verbose=False,
#         )
#     except TypeError:
#         try:
#             # Legacy API (0.90 – 1.5)
#             model.fit(
#                 X_train, y_train,
#                 eval_set=[(X_train, y_train), (X_val, y_val)],
#                 early_stopping_rounds=50,
#                 verbose=False,
#             )
#         except TypeError:
#             # Very old wrapper – no early stopping available
#             model.fit(
#                 X_train, y_train,
#                 eval_set=[(X_train, y_train), (X_val, y_val)],
#                 verbose=False,
#             )
#
#     # ── Evaluation on hold‑out test set ──────────────────────────
#     prob = model.predict_proba(X_test)[:, 1]
#     p, r, t = precision_recall_curve(y_test, prob)
#     f1 = 2 * p * r / (p + r + 1e-12)
#     tau = t[int(f1.argmax())]
#     y_hat = (prob >= tau).astype(int)
#
#     log.info("τ ≈ %.3f   F1 = %.3f   Acc = %.3f   AUC = %.3f",
#              tau, f1.max(), accuracy_score(y_test, y_hat),
#              roc_auc_score(y_test, prob))
#
#     # ── Plots ────────────────────────────────────────────────────
#     RocCurveDisplay.from_predictions(y_test, prob)
#     plt.title("ROC — XGB")
#     plt.tight_layout()
#     plt.show()
#
#     PrecisionRecallDisplay.from_predictions(y_test, prob)
#     plt.title("PR — XGB")
#     plt.tight_layout()
#     plt.show()
#
#     ConfusionMatrixDisplay.from_predictions(
#         y_test, y_hat, cmap="viridis", colorbar=False)
#     plt.title("Confusion Matrix — XGB")
#     plt.tight_layout()
#     plt.show()
#
#     # ── Save model ───────────────────────────────────────────────
#     MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
#     model.save_model(MODEL_OUT)
#     log.info("✅  model saved → %s", MODEL_OUT)
#
#
# # ───────────────────────────── main ──────────────────────────────
#
# def main() -> None:
#     warnings.filterwarnings("ignore", category=FutureWarning)
#
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--split", type=float, default=.7,
#                     help="train/test split ratio (chronological)")
#     ap.add_argument("--search-iter", type=int, default=40,
#                     help="randomized‑search iterations for XGB")
#     args = ap.parse_args()
#
#     banner("🏁  PIPELINE START — loading model_data_full.csv")
#     df = load_full()
#     banner(f"Dataset loaded → {len(df):,} daily rows")
#
#     ridge_regression(df, split=args.split)
#     xgb_direction(df, split=args.split, n_iter=args.search_iter)
#
#     banner("✅  PIPELINE FINISHED")
#
#
# if __name__ == "__main__":
#     main()


"""
final_project_ML_model.py
────────────────────────────────────────────────────────────────────────────
Build two models on model_data_full.csv:

1. Ridge regression (magnitude baseline).
2. XGBoost classifier (direction) with 400‑draw RandomisedSearchCV,
   early‑stopping fallback, threshold optimisation, full evaluation
   plots, feature‑importance bar chart, and a new TRAIN / VAL log‑loss
   learning‑curve plot.

Updated: 2025‑07‑16
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
DATA_MODEL = ROOT / "data" / "model" / "model_data_full.csv"
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
        "scale_pos_weight": uniform(0.8, 0.6),
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

    plt.figure(figsize=(4, 4))
    plt.scatter(y_te, preds, alpha=.5)
    plt.axline((0, 0), slope=1, color="k", linestyle="--")
    plt.title(f"Ridge α={best_alpha}")
    plt.tight_layout()
    plt.show()


# ═══════════ XGBoost classifier (direction) ═══════════════════════
def xgb_direction(df: pd.DataFrame, *, split: float, n_iter: int) -> None:
    banner("🚀 XGBoost – predicting direction")
    X, y = feature_target(df, direction=True)

    idx_test = int(split * len(df))
    idx_val = int(idx_test * 0.9)
    X_train, y_train = X.iloc[:idx_val], y.iloc[:idx_val]
    X_val, y_val     = X.iloc[idx_val:idx_test], y.iloc[idx_val:idx_test]
    X_test, y_test   = X.iloc[idx_test:], y.iloc[idx_test:]

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

    try:
        fit_kwargs["callbacks"] = [
            xgb.callback.EarlyStopping(rounds=CONFIG["EARLY_STOP"])
        ]
        model.fit(X_train, y_train, **fit_kwargs)
    except TypeError:
        try:
            fit_kwargs.pop("callbacks", None)
            fit_kwargs["early_stopping_rounds"] = CONFIG["EARLY_STOP"]
            model.fit(X_train, y_train, **fit_kwargs)
        except TypeError:
            fit_kwargs.pop("early_stopping_rounds", None)
            model.fit(X_train, y_train, **fit_kwargs)

    # ── NEW: log‑loss learning curve ───────────────────────────────
    evals = model.evals_result()
    if evals and "validation_0" in evals:
        train_loss = evals["validation_0"]["logloss"]
        val_loss   = evals["validation_1"]["logloss"]
        plt.figure(figsize=(6, 4))
        plt.plot(train_loss, label="train logloss")
        plt.plot(val_loss,   label="val logloss")
        plt.axvline(len(val_loss) - 1, color="k", ls="--",
                    label="early stop")
        plt.legend()
        plt.xlabel("Trees")
        plt.ylabel("Logloss")
        plt.title("Learning curve — XGB")
        plt.tight_layout()
        plt.show()

    # ── Threshold optimisation ────────────────────────────────────
    prob_val = model.predict_proba(X_val)[:, 1]
    p, r, t = precision_recall_curve(y_val, prob_val)
    best_tau = float(t[int((2 * p * r / (p + r + 1e-12)).argmax())])

    # ── Evaluation on test ────────────────────────────────────────
    prob = model.predict_proba(X_test)[:, 1]
    y_hat = (prob >= best_tau).astype(int)

    log.info(
        "τ ≈ %.3f   F1 = %.3f   Acc = %.3f   AUC = %.3f",
        best_tau,
        f1_score(y_test, y_hat),
        accuracy_score(y_test, y_hat),
        roc_auc_score(y_test, prob),
    )

    RocCurveDisplay.from_predictions(y_test, prob)
    plt.title("ROC — XGB"); plt.tight_layout(); plt.show()

    PrecisionRecallDisplay.from_predictions(y_test, prob)
    plt.title("PR — XGB"); plt.tight_layout(); plt.show()

    ConfusionMatrixDisplay.from_predictions(
        y_test, y_hat, cmap="viridis", colorbar=False)
    plt.title("Confusion Matrix — XGB"); plt.tight_layout(); plt.show()

    importance = model.get_booster().get_score(importance_type="gain")
    if importance:
        pd.Series(importance).sort_values(ascending=False).head(20)[::-1].plot.barh()
        plt.title("Top‑20 feature importance (Gain)")
        plt.tight_layout()
        plt.show()

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_OUT)
    with open(THRESH_OUT, "w") as f:
        json.dump({"tau": best_tau}, f)

    log.info("✅ model saved → %s", MODEL_OUT)
    log.info("✅ Threshold saved → %s", THRESH_OUT)


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
