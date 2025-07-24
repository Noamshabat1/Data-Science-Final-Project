"""
final_project_ML_model.py
---------------------------------
Train & evaluate:
  • Ridge regression (regression on next_day_pct_change)
  • XGBClassifier (direction up/down)
  • RandomForestClassifier (direction up/down)

Key features:
  - Chronological split: 80% train, 10% val, 10% test
  - Correlation filter + (optional) VIF filter
  - Fast manual grid for XGB using sklearn API + early stopping
  - Plots saved under model/output/<model_name>/plots (cleaned each run)
  - Metrics & models saved to model/output/<model_name>

Requires the dataset produced by build_full_model_data.py:
  Data/model/model_data_full.csv
"""

from __future__ import annotations

import json
import shutil
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Optional statsmodels for VIF
try:
    import statsmodels.api as sm  # type: ignore
except ImportError:  # pragma: no cover
    sm = None

# ────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "Data" / "model" / "model_data_full.csv"
OUT_ROOT = ROOT / "model" / "output"

CONFIG: Dict = {
    "RANDOM_STATE": 42,
    "MAX_JOBS": -1,
    "CV_SPLITS": 5,
    "VAL_FRACTION_OF_TRAIN": 0.1,  # 10% of train chunk becomes validation
    "CORR_THRESH": 0.95,
    "USE_VIF": True,  # will skip automatically if statsmodels not available
    "VIF_THRESH": 10.0,

    # Ridge grid
    "RIDGE_GRID": {"model__alpha": [0.1, 1.0, 10.0, 100.0]},

    # XGB manual grid (keep small!)
    "XGB_GRID": {
        "max_depth": [3, 4, 5],
        "learning_rate": [0.01, 0.03, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "reg_lambda": [1.0, 3.0],
        "reg_alpha": [0.0],
        "n_estimators": [400, 800],  # large, but ES will trim
    },
    "XGB_ES_ROUNDS": 30,
    "XGB_MAX_TRIALS": 20,  # slice grid combos if huge

    # RF params
    "RF_PARAMS": {
        "n_estimators": 400,
        "max_depth": None,
        "min_samples_leaf": 3,
        "n_jobs": -1,
        "random_state": 42,
    },
}


# ────────────────────────────────────────────────────────────────
# Logging helper
# ────────────────────────────────────────────────────────────────
def log(msg: str, *args, level: str = "INFO") -> None:
    if args:  # allow printf-style usage
        msg = msg % args
    print(f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S} {level:>6} │ {msg}")


# ────────────────────────────────────────────────────────────────
# IO helpers
# ────────────────────────────────────────────────────────────────
def prepare_model_dirs() -> Dict[str, Path]:
    """
    Create (and clean) per-model directories.
    Returns dict: {"ridge": Path, "xgb": Path, "rf": Path}
    """
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    dirs = {}
    for name in ("ridge", "xgb", "rf"):
        mdir = OUT_ROOT / name
        plots = mdir / "plots"
        mdir.mkdir(parents=True, exist_ok=True)
        if plots.exists():
            shutil.rmtree(plots)
        plots.mkdir(parents=True, exist_ok=True)
        dirs[name] = mdir
    return dirs


def plot_and_save(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ────────────────────────────────────────────────────────────────
# Data & feature selection
# ────────────────────────────────────────────────────────────────
def load_dataset() -> pd.DataFrame:
    log("🏁 PIPELINE START — loading %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    # Ensure target present
    df = df.dropna(subset=["next_day_pct_change"]).reset_index(drop=True)
    log("Dataset → %d rows × %d cols", len(df), df.shape[1])
    return df


def train_val_test_split(df: pd.DataFrame, target_col: str, split: float = 0.8
                         ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
pd.Series, pd.Series, pd.Series]:
    """
    Chronological split:
      train = first 80%
      val   = last 10% of train
      test  = last 10% total
    """
    N = len(df)
    idx_test = int(split * N)  # start of test
    idx_val = int(idx_test * (1 - CONFIG["VAL_FRACTION_OF_TRAIN"]))  # end of train

    X = df.drop(columns=[target_col, "date"])
    y = df[target_col]

    Xtr, ytr = X.iloc[:idx_val], y.iloc[:idx_val]
    Xval, yval = X.iloc[idx_val:idx_test], y.iloc[idx_val:idx_test]
    Xte, yte = X.iloc[idx_test:], y.iloc[idx_test:]
    return Xtr, Xval, Xte, ytr, yval, yte


def corr_filter(X: pd.DataFrame, thr: float) -> Tuple[pd.DataFrame, List[str]]:
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = [col for col in upper.columns if any(upper[col] > thr)]
    X2 = X.drop(columns=drop_cols)
    return X2, drop_cols


def vif_filter(X: pd.DataFrame, thr: float) -> Tuple[pd.DataFrame, List[str]]:
    if sm is None:
        log("statsmodels not installed — skipping VIF filter.", level="WARNING")
        return X, []
    cols = list(X.columns)
    removed = []
    while True:
        vif_vals = []
        for i, c in enumerate(cols):
            try:
                vif = sm.stats.outliers_influence.variance_inflation_factor(
                    X[cols].values, i
                )
            except Exception:
                vif = 0.0
            vif_vals.append(vif)
        max_vif = max(vif_vals)
        if max_vif > thr:
            idx = vif_vals.index(max_vif)
            removed_col = cols.pop(idx)
            removed.append(removed_col)
        else:
            break
    return X[cols], removed


def select_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    # Numeric only
    num_cols = df.drop(columns=["date", "next_day_pct_change"]).select_dtypes(include=np.number).columns
    X = df[num_cols].copy()

    # Step 1: correlation filter
    Xc, dropped_corr = corr_filter(X, CONFIG["CORR_THRESH"])

    # Step 2: VIF (optional)
    if CONFIG["USE_VIF"]:
        Xf, dropped_vif = vif_filter(Xc, CONFIG["VIF_THRESH"])
    else:
        dropped_vif = []
        Xf = Xc

    log("After var-select: %d features (dropped corr:%d, vif:%d)",
        Xf.shape[1], len(dropped_corr), len(dropped_vif))
    return Xf, dropped_corr, dropped_vif


# ────────────────────────────────────────────────────────────────
# Regression (Ridge)
# ────────────────────────────────────────────────────────────────
def train_ridge(Xtr, ytr, Xval, yval, Xte, yte, out_dir: Path) -> None:
    log("Training Ridge regression (grid CV)…")
    plots_dir = out_dir / "plots"

    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", Ridge(random_state=CONFIG["RANDOM_STATE"]))
    ])

    cv = TimeSeriesSplit(
        n_splits=CONFIG["CV_SPLITS"],
        test_size=int(len(Xtr) * CONFIG["VAL_FRACTION_OF_TRAIN"])
    )

    grid = GridSearchCV(
        pipe,
        CONFIG["RIDGE_GRID"],
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=CONFIG["MAX_JOBS"]
    ).fit(Xtr, ytr)

    best = grid.best_estimator_
    log("Ridge best params: %s", grid.best_params_)

    pred = best.predict(Xte)
    rmse = float(np.sqrt(mean_squared_error(yte, pred)))
    r2 = r2_score(yte, pred)
    log("Ridge RMSE = %.4f   R2 = %.4f", rmse, r2)

    # Actual vs Pred
    plt.figure(figsize=(6, 4))
    plt.scatter(yte, pred, alpha=0.5)
    lims = [float(np.min([yte.min(), pred.min()])), float(np.max([yte.max(), pred.max()]))]
    plt.plot(lims, lims, "r--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Ridge: Actual vs Pred")
    plot_and_save(plots_dir / "ridge_actual_vs_pred.png")

    # Residuals
    residuals = yte - pred
    plt.figure(figsize=(6, 4))
    plt.scatter(pred, residuals, alpha=0.5)
    plt.axhline(0, color="r", linestyle="--")
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.title("Ridge: Residual plot")
    plot_and_save(plots_dir / "ridge_residuals.png")

    joblib.dump(best, out_dir / "ridge_model.joblib")
    with open(out_dir / "ridge_metrics.json", "w") as f:
        json.dump({"rmse": rmse, "r2": r2, "best_params": grid.best_params_}, f, indent=2)


# ────────────────────────────────────────────────────────────────
# Classification helpers
# ────────────────────────────────────────────────────────────────
def optimal_threshold(y_true, prob, beta: float = 1.0) -> Tuple[float, Dict[str, float]]:
    """
    Choose threshold that maximizes F1 (or Fbeta) on validation.
    Returns threshold and dict of metrics.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    f = (1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall + 1e-12)
    idx = int(np.nanargmax(f))
    thr = thresholds[idx] if idx < len(thresholds) else 0.5
    return float(thr), {
        "f1": float(f[idx]),
        "precision": float(precision[idx]),
        "recall": float(recall[idx]),
    }


def eval_classifier(model_name: str,
                    y_true,
                    prob,
                    plots_dir: Path,
                    out_dir: Path) -> float:
    """
    Evaluate a probabilistic binary classifier:
    - picks best F1 threshold on provided set
    - saves PR/ROC curves, confusion matrix, report & metrics json
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    prob = np.asarray(prob).ravel()

    thr, _ = optimal_threshold(y_true, prob)
    pred = (prob >= thr).astype(int)

    ap = average_precision_score(y_true, prob)
    roc = roc_auc_score(y_true, prob)
    f1_pos = f1_score(y_true, pred, pos_label=1)
    f1_neg = f1_score(y_true, pred, pos_label=0)
    macro_f1 = f1_score(y_true, pred, average="macro")
    acc = (pred == y_true).mean()

    log(f"{model_name} τ={thr:.3f}  MacroF1={macro_f1:.3f}  F1_pos={f1_pos:.3f}  "
        f"F1_neg={f1_neg:.3f}  Acc={acc:.3f}  AP={ap:.3f}  ROC={roc:.3f}")
    cm = confusion_matrix(y_true, pred)
    log(f"{model_name} Confusion matrix:\n{cm}")
    rep = classification_report(y_true, pred)
    log(f"{model_name} report:\n{rep}")

    # PR curve
    prec, rec, _ = precision_recall_curve(y_true, prob)
    plt.figure(figsize=(6, 4))
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{model_name} Precision-Recall Curve (AP={ap:.3f})")
    plot_and_save(plots_dir / f"{model_name.lower()}_pr_curve.png")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, prob)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"{model_name} ROC Curve (AUC={roc:.3f})")
    plot_and_save(plots_dir / f"{model_name.lower()}_roc_curve.png")

    # Save metrics
    with open(out_dir / f"{model_name.lower()}_metrics.json", "w") as f:
        json.dump({
            "threshold": thr,
            "macro_f1": macro_f1,
            "f1_pos": f1_pos,
            "f1_neg": f1_neg,
            "accuracy": acc,
            "ap": ap,
            "roc_auc": roc,
            "confusion_matrix": cm.tolist(),
            "report": rep
        }, f, indent=2)

    return thr


# ────────────────────────────────────────────────────────────────
# XGB block
# ────────────────────────────────────────────────────────────────
def _param_product(grid: Dict[str, List]) -> List[Dict]:
    keys = list(grid.keys())
    for values in product(*[grid[k] for k in keys]):
        yield dict(zip(keys, values))


def xgb_grid_search_fast(
        Xtr, ytr, Xval, yval,
        spw: float,
        base_params: Dict,
        grid: Dict[str, List],
        max_trials: int,
        es_rounds: int
) -> Dict:
    combos = list(_param_product(grid))
    if max_trials and max_trials < len(combos):
        combos = combos[:max_trials]

    best_params = None
    best_score = -1.0

    for i, hp in enumerate(combos, 1):
        params = {
            **base_params,
            **hp,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": CONFIG["RANDOM_STATE"],
        }

        model = XGBClassifier(**params)

        fit_kwargs = {}
        if "early_stopping_rounds" in model.fit.__code__.co_varnames:
            fit_kwargs = {
                "eval_set": [(Xtr, ytr), (Xval, yval)],
                "early_stopping_rounds": es_rounds,
                "verbose": False,
            }

        model.fit(
            Xtr, ytr,
            sample_weight=np.where(ytr == 1, spw, 1.0),
            **fit_kwargs
        )

        prob_val = model.predict_proba(Xval)[:, 1]
        ap = average_precision_score(yval, prob_val)

        if ap > best_score:
            best_score = ap
            best_params = params

    log("Best val PR-AUC = %.3f  with params: %s", best_score, best_params)
    return best_params


def train_xgb_classifier(Xtr, ytr, Xval, yval, Xte, yte, out_dir: Path) -> None:
    log("Training XGBClassifier (grid search)…")
    plots_dir = out_dir / "plots"

    pos_w = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    base_params = {}

    best_params = xgb_grid_search_fast(
        Xtr, ytr, Xval, yval,
        pos_w,
        base_params,
        CONFIG["XGB_GRID"],
        CONFIG["XGB_MAX_TRIALS"],
        CONFIG["XGB_ES_ROUNDS"]
    )

    # Final fit on train+val
    Xfull = pd.concat([Xtr, Xval])
    yfull = np.concatenate([ytr, yval])

    final_xgb = XGBClassifier(**best_params)
    fit_kwargs = {}
    if "early_stopping_rounds" in final_xgb.fit.__code__.co_varnames:
        fit_kwargs = {
            "eval_set": [(Xtr, ytr), (Xval, yval)],
            "early_stopping_rounds": CONFIG["XGB_ES_ROUNDS"],
            "verbose": False,
        }

    final_xgb.fit(
        Xfull, yfull,
        sample_weight=np.where(yfull == 1, pos_w, 1.0),
        **fit_kwargs
    )

    # Learning curve
    try:
        ev = final_xgb.evals_result()
        if ev:
            plt.figure(figsize=(6, 4))
            if "validation_0" in ev:
                plt.plot(ev["validation_0"]["logloss"], label="train")
            if "validation_1" in ev:
                plt.plot(ev["validation_1"]["logloss"], label="val")
            plt.xlabel("Iteration")
            plt.ylabel("Logloss")
            plt.title("XGB Learning Curve")
            plt.legend()
            plot_and_save(plots_dir / "xgb_learning_curve.png")
    except Exception:
        pass

    # Test eval
    prob_te = final_xgb.predict_proba(Xte)[:, 1]
    thr = eval_classifier("XGB", yte, prob_te, plots_dir, out_dir)

    # Feature importance
    fi = final_xgb.feature_importances_
    order = np.argsort(fi)[::-1][:30]
    plt.figure(figsize=(6, 8))
    plt.barh(np.array(Xtr.columns)[order][::-1], fi[order][::-1])
    plt.title("XGB Top-30 Feature Importances")
    plt.tight_layout()
    plot_and_save(plots_dir / "xgb_feature_importance.png")

    # Save model & params/threshold
    joblib.dump(final_xgb, out_dir / "xgb_model.joblib")
    with open(out_dir / "xgb_threshold.json", "w") as f:
        json.dump({"threshold": thr, "best_params": best_params}, f, indent=2)


# ────────────────────────────────────────────────────────────────
# RF block
# ────────────────────────────────────────────────────────────────
def train_rf_classifier(Xtr, ytr, Xval, yval, Xte, yte, out_dir: Path) -> None:
    from sklearn.ensemble import RandomForestClassifier

    log("Training RandomForestClassifier…")
    plots_dir = out_dir / "plots"

    rf = RandomForestClassifier(**CONFIG["RF_PARAMS"])
    rf.fit(Xtr, ytr)

    # threshold from validation
    prob_val = rf.predict_proba(Xval)[:, 1]
    thr, _ = optimal_threshold(yval, prob_val)

    # test probs
    prob_te = rf.predict_proba(Xte)[:, 1]

    # Evaluate on test using thr
    yte_arr = np.asarray(yte)
    pred_te = (prob_te >= thr).astype(int)

    ap = average_precision_score(yte_arr, prob_te)
    roc = roc_auc_score(yte_arr, prob_te)
    f1p = f1_score(yte_arr, pred_te, pos_label=1)
    f1n = f1_score(yte_arr, pred_te, pos_label=0)
    macro_f1 = f1_score(yte_arr, pred_te, average="macro")
    acc = (pred_te == yte_arr).mean()

    log(f"RF  τ={thr:.3f}  MacroF1={macro_f1:.3f}  F1_pos={f1p:.3f}  F1_neg={f1n:.3f}  "
        f"Acc={acc:.3f}  AP={ap:.3f}  ROC={roc:.3f}")
    log(f"RF Confusion matrix:\n{confusion_matrix(yte_arr, pred_te)}")
    log(f"RF report:\n{classification_report(yte_arr, pred_te)}")

    # PR curve
    p, r, _ = precision_recall_curve(yte_arr, prob_te)
    plt.figure(figsize=(6, 4))
    plt.plot(r, p)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"RF Precision-Recall Curve (AP={ap:.3f})")
    plot_and_save(plots_dir / "rf_pr_curve.png")

    # ROC curve
    fpr, tpr, _ = roc_curve(yte_arr, prob_te)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"RF ROC Curve (AUC={roc:.3f})")
    plot_and_save(plots_dir / "rf_roc_curve.png")

    # Feature importance
    fi = rf.feature_importances_
    order = np.argsort(fi)[::-1][:30]
    plt.figure(figsize=(6, 8))
    plt.barh(np.array(Xtr.columns)[order][::-1], fi[order][::-1])
    plt.title("RF Top-30 Feature Importances")
    plt.tight_layout()
    plot_and_save(plots_dir / "rf_feature_importance.png")

    # Save model & metrics
    joblib.dump(rf, out_dir / "rf_model.joblib")
    with open(out_dir / "rf_metrics.json", "w") as f:
        json.dump(
            {"threshold": thr, "ap": ap, "roc_auc": roc, "macro_f1": macro_f1,
             "f1_pos": f1p, "f1_neg": f1n, "acc": acc},
            f, indent=2
        )


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────
def main() -> None:
    log("═════════════════════════════════════════════════════════════════════════════════════════")
    df0 = load_dataset()

    # Feature selection
    Xf, _, _ = select_features(df0)
    df = pd.concat([df0[["date", "next_day_pct_change"]], Xf], axis=1)

    # Splits
    Xtr, Xval, Xte, ytr_reg, yval_reg, yte_reg = train_val_test_split(df, "next_day_pct_change")
    # Classification target: up/down
    ytr_cls = (ytr_reg > 0).astype(int).values
    yval_cls = (yval_reg > 0).astype(int).values
    yte_cls = (yte_reg > 0).astype(int).values

    out_dirs = prepare_model_dirs()
    log("Outputs → %s", OUT_ROOT)

    # Ridge
    train_ridge(Xtr, ytr_reg, Xval, yval_reg, Xte, yte_reg, out_dirs["ridge"])

    # RF
    train_rf_classifier(Xtr, ytr_cls, Xval, yval_cls, Xte, yte_cls, out_dirs["rf"])

    # XGB
    train_xgb_classifier(Xtr, ytr_cls, Xval, yval_cls, Xte, yte_cls, out_dirs["xgb"])

    log("✅ Done.")
    log("═════════════════════════════════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
