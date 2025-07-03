#!/usr/bin/env python
# ─────────────────────────────────────────────────────────────────────────────
# FinalProject_ML_pipeline_v5.py
# Like v4, plus extra lag/vol, rolling tweet‐features, and day‐of‐week cyclic.
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
import argparse, logging, sys, warnings
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_squared_error, r2_score, precision_recall_curve,
    accuracy_score, roc_auc_score,
    ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
)
from sklearn.preprocessing import StandardScaler
from scipy.stats import uniform, randint

import xgboost as xgb
from xgboost import XGBClassifier

# ─────────── optional SHAP ───────────
try:
    import shap

    SHAP_OK = True
except ModuleNotFoundError:
    SHAP_OK = False

# ─────────── paths / config ───────────
ROOT = Path(__file__).resolve().parents[1]
DATA_CLEAN = ROOT / "Data" / "clean"
DATA_ORIG = ROOT / "Data" / "original"
MODEL_OUT = ROOT / "Model" / "xgb_tsla.json"

US_TZ = "America/New_York"
MKT_CLOSE = pd.Timestamp("16:00").time()  # 4 pm Eastern

CONFIG: Dict[str, Any] = {
    "RANDOM_STATE": 42,
    "MAX_JOBS": 8,
    "STOCK_FILE": "tesla_stock_data_2000_2025.csv",
    "STOCK_DATE_COL": "Price",
    "TWEET_FILES": [
        "clean_musk_tweets.csv",
        "clean_musk_retweets.csv",
        "clean_musk_replies.csv",
    ],
    "KEYWORDS": [
        "model 3", "model y", "cybertruck", "ai",
        "robot", "teslabot", "fsd", "tesla energy", "spacex",
    ],
    "TFIDF_MAX": 400,
    "BASE_XGB": dict(
        n_estimators=800,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
    ),
    "HPARAM_SPACE": {
        "learning_rate": uniform(0.01, 0.19),
        "max_depth": randint(3, 9),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "gamma": uniform(0.0, 4.0),
        "reg_lambda": uniform(0.5, 4.5),
    },
}

# ─────────── logging helper ───────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)6s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)


def banner(msg: str) -> None:
    log.info("=" * 100);
    log.info(msg);
    log.info("=" * 100)


def _req(p: Path) -> None:
    if not p.exists():
        raise FileNotFoundError(p)


# ───────────────────── Tweets ─────────────────────
def _std_post(df: pd.DataFrame, src: str) -> pd.DataFrame:
    txt = ("cleanedReplyContent" if "cleanedReplyContent" in df.columns
           else "fullText" if "fullText" in df.columns
    else "originalContent")
    ts = "createdAt" if "createdAt" in df.columns else "created_at"
    dt = pd.to_datetime(df[ts], errors="coerce", utc=True).dt.tz_convert(US_TZ)

    after_close = dt.dt.time > MKT_CLOSE
    mkt = np.where(
        after_close,
        (dt + pd.Timedelta(days=1)).dt.date,
        dt.dt.date
    )

    return pd.DataFrame({
        "mkt_date": mkt,
        "text": df[txt].fillna(""),
        "sentiment": df.get("sentiment", 0.0),
        "source": src,
    })


def load_posts() -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for fn in CONFIG["TWEET_FILES"]:
        p = DATA_CLEAN / fn;
        _req(p)
        frames.append(_std_post(
            pd.read_csv(p),
            fn.split("_")[2].split(".")[0]
        ))
    return pd.concat(frames, ignore_index=True)


def agg_daily(posts: pd.DataFrame) -> pd.DataFrame:
    for kw in CONFIG["KEYWORDS"]:
        col = f"kw_{kw.replace(' ', '_')}"
        posts[col] = posts["text"].str.contains(
            kw, case=False, regex=False, na=False
        ).astype(int)

    agg = {
              "text_cat": ("text", " ".join),
              "tweet_count": ("text", "size"),
              "sentiment_mean": ("sentiment", "mean"),
          } | {
              f"kw_{k.replace(' ', '_')}": (f"kw_{k.replace(' ', '_')}", "sum")
              for k in CONFIG["KEYWORDS"]
          }

    out = posts.groupby("mkt_date").agg(**agg).reset_index()
    out["mkt_date"] = pd.to_datetime(out["mkt_date"]).dt.date
    return out


# ───────────────────── Market ─────────────────────
def load_market() -> pd.DataFrame:
    p = DATA_ORIG / CONFIG["STOCK_FILE"];
    _req(p)
    raw = pd.read_csv(p)

    dt_col = CONFIG["STOCK_DATE_COL"]
    raw = raw[~raw[dt_col].isin(["Ticker", "Date"])].copy()

    close_col = next(
        (c for c in ["Close", "Adj Close", "close", "close_price"] if c in raw.columns),
        None
    )
    if close_col is None:
        raise ValueError("No close-price column found.")

    # parse date & close
    raw[dt_col] = pd.to_datetime(raw[dt_col], format="%Y-%m-%d").dt.date
    raw[close_col] = pd.to_numeric(raw[close_col], errors="coerce")

    # last close each day
    close = (
        raw[[dt_col, close_col]]
        .dropna()
        .groupby(dt_col)[close_col]
        .last()
    )

    # build features
    df = pd.DataFrame({
        "close": close,
        "ret_fwd_1d": close.pct_change().shift(-1),
        "ret_lag_1d": close.pct_change(),
        "ret_lag_5d": close.pct_change(5),
        "vol_5d": close.pct_change().rolling(5).std(),
        "vol_10d": close.pct_change().rolling(10).std(),
    }).dropna()
    df.index.name = "mkt_date"
    return df


def merge_daily(tw: pd.DataFrame, mk: pd.DataFrame) -> pd.DataFrame:
    df = pd.merge(
        tw, mk,
        left_on="mkt_date",
        right_index=True,
        how="inner"
    ).sort_values("mkt_date")

    # rolling tweet‐count features
    df["tc_3d"] = df["tweet_count"].rolling(3).mean().bfill()
    df["tc_7d"] = df["tweet_count"].rolling(7).mean().bfill()

    # day‐of‐week cyclic
    dow = pd.to_datetime(df["mkt_date"]).dt.weekday
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    # drop any leftover NaNs
    return df.dropna().reset_index(drop=True)


# ───────────────── TF-IDF helper ───────────────────
class TrainOnlyTFIDF:
    def __init__(self, *, max_feats: int, keep: int):
        self.vec = TfidfVectorizer(
            max_features=max_feats,
            ngram_range=(1, 2),
            min_df=5, max_df=0.8
        )
        self.keep = keep

    def fit(self, text_train: pd.Series, y_train: pd.Series):
        tf = self.vec.fit_transform(text_train)
        arr = tf.toarray()  # densify
        mi = mutual_info_classif(
            arr, y_train,
            discrete_features=False,
            random_state=CONFIG["RANDOM_STATE"]
        )
        self.sel_ = np.argsort(mi)[-self.keep:]

    def transform(self, text: pd.Series) -> pd.DataFrame:
        tf = self.vec.transform(text)[:, self.sel_]
        cols = [f"tfidf_{t}" for t in self.vec.get_feature_names_out()[self.sel_]]
        return pd.DataFrame(tf.toarray(), columns=cols, index=text.index)


# ─────────────────── Ridge baseline ──────────────────────────────
def ridge_baseline(df: pd.DataFrame, split: float) -> None:
    banner("📊  Ridge baseline (numeric return)")
    kws = [f"kw_{k.replace(' ', '_')}" for k in CONFIG["KEYWORDS"]]
    feats = ["sentiment_mean", "tweet_count", "tc_3d", "tc_7d",
             "ret_lag_1d", "ret_lag_5d", "vol_5d", "vol_10d", "dow_sin", "dow_cos"] + kws
    X = df[feats].to_numpy()
    y = df["ret_fwd_1d"].to_numpy()

    idx = int(split * len(df))
    X_tr, X_te = X[:idx], X[idx:]
    y_tr, y_te = y[:idx], y[idx:]

    sc = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = sc.transform(X_tr), sc.transform(X_te)

    gcv = GridSearchCV(
        Ridge(random_state=CONFIG["RANDOM_STATE"]),
        {"alpha": [10 ** i for i in range(-4, 3)]},
        cv=TimeSeriesSplit(5),
        scoring="neg_root_mean_squared_error"
    ).fit(X_tr_s, y_tr)

    best = gcv.best_params_["alpha"]
    preds = gcv.predict(X_te_s)
    rmse = np.sqrt(mean_squared_error(y_te, preds))
    log.info("α=%g  RMSE=%.5f  R²=%.4f", best, rmse, r2_score(y_te, preds))

    plt.figure(figsize=(4, 4))
    plt.scatter(y_te, preds, alpha=.5)
    plt.axline((0, 0), slope=1, linestyle="--", color="k")
    plt.title(f"Ridge α={best}")
    plt.tight_layout();
    plt.show()


# ─────────────────── XGB model ───────────────────────────────────
def xgb_model(df: pd.DataFrame, *, split: float, tfidf_max: int, n_iter: int) -> None:
    banner("🚀  XGBoost (direction)")

    idx_test = int(split * len(df))
    idx_val = int(idx_test * 0.9)
    y_full = (df["ret_fwd_1d"] > 0).astype(int)

    banner(f"📝  TF-IDF selection (max={tfidf_max})")
    tfidf = TrainOnlyTFIDF(max_feats=6000, keep=tfidf_max)
    tfidf.fit(df["text_cat"][:idx_val], y_full[:idx_val])
    tf_df = tfidf.transform(df["text_cat"])

    base_feats = ["sentiment_mean", "tweet_count", "tc_3d", "tc_7d",
                  "ret_lag_1d", "ret_lag_5d", "vol_5d", "vol_10d", "dow_sin", "dow_cos"] + \
                 [f"kw_{k.replace(' ', '_')}" for k in CONFIG["KEYWORDS"]]
    X_full = pd.concat([df[base_feats], tf_df], axis=1).astype("float32")

    X_tr, y_tr = X_full.iloc[:idx_val], y_full.iloc[:idx_val]
    X_val, y_val = X_full.iloc[idx_val:idx_test], y_full.iloc[idx_val:idx_test]
    X_te, y_te = X_full.iloc[idx_test:], y_full.iloc[idx_test:]

    neg, pos = (y_tr == 0).sum(), (y_tr == 1).sum()
    base_params = dict(
        CONFIG["BASE_XGB"],
        random_state=CONFIG["RANDOM_STATE"],
        n_jobs=CONFIG["MAX_JOBS"],
        scale_pos_weight=np.sqrt(neg / pos)
    )

    search = RandomizedSearchCV(
        XGBClassifier(**base_params),
        CONFIG["HPARAM_SPACE"], n_iter=n_iter,
        cv=TimeSeriesSplit(4), scoring="roc_auc",
        n_jobs=CONFIG["MAX_JOBS"], random_state=CONFIG["RANDOM_STATE"]
    ).fit(X_tr, y_tr)

    best = search.best_params_;
    log.info("Best params: %s", best)
    model = XGBClassifier(**base_params | best)

    # robust early-stop
    try:
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_tr, y_tr), (X_val, y_val)],
            callbacks=[xgb.callback.EarlyStopping(rounds=50)],
            verbose=False
        )
    except TypeError:
        model = XGBClassifier(**base_params | best, early_stopping_round=50)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_tr, y_tr), (X_val, y_val)],
            verbose=False
        )

    # diagnostics
    evals = model.evals_result()
    plt.figure(figsize=(6, 3))
    plt.plot(evals["validation_0"]["logloss"], label="train")
    plt.plot(evals["validation_1"]["logloss"], label="val")
    plt.legend();
    plt.ylabel("log-loss");
    plt.xlabel("round")
    plt.title("XGB log-loss");
    plt.tight_layout();
    plt.show()

    prob = model.predict_proba(X_te)[:, 1]
    p, r, t = precision_recall_curve(y_te, prob)
    f1 = 2 * p * r / (p + r + 1e-12);
    tau = t[int(f1.argmax())]
    y_hat = (prob >= tau).astype(int)

    log.info(
        "τ≈%.3f  F1=%.3f  Acc=%.3f  AUC=%.3f",
        tau, f1.max(), accuracy_score(y_te, y_hat),
        roc_auc_score(y_te, prob)
    )

    RocCurveDisplay.from_predictions(y_te, prob);
    plt.title("ROC — XGB");
    plt.tight_layout();
    plt.show()
    PrecisionRecallDisplay.from_predictions(y_te, prob);
    plt.title("PR Curve — XGB");
    plt.tight_layout();
    plt.show()
    ConfusionMatrixDisplay.from_predictions(y_te, y_hat, cmap="viridis", colorbar=False)
    plt.title("Confusion Matrix — XGB");
    plt.tight_layout();
    plt.show()

    if SHAP_OK:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_te, check_additivity=False)
        mean_abs = np.abs(shap_vals).mean(axis=0)
        k = 20;
        idx = np.argsort(mean_abs)[-k:][::-1]
        plt.figure(figsize=(6, 5))
        plt.barh(range(k), mean_abs[idx][::-1])
        plt.yticks(range(k), X_te.columns[idx][::-1])
        plt.xlabel("mean |SHAP|");
        plt.title("Top 20 features");
        plt.tight_layout();
        plt.show()
    else:
        log.warning("SHAP not installed – skipping SHAP plot.")

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_OUT)
    log.info("✅  Model saved → %s", MODEL_OUT)


# ─────────────────── main ─────────────────────────────────────────
def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=float, default=.7)
    ap.add_argument("--tfidf-max", type=int, default=CONFIG["TFIDF_MAX"])
    ap.add_argument("--search-iter", type=int, default=40)
    args = ap.parse_args()

    banner("🏁  PIPELINE START")
    posts = load_posts()
    tweets = agg_daily(posts)
    market = load_market()
    data = merge_daily(tweets, market)
    banner(f"Merged dataset: {len(data):,} trading days")

    ridge_baseline(data, split=args.split)
    xgb_model(data, split=args.split,
              tfidf_max=args.tfidf_max,
              n_iter=args.search_iter)
    banner("✅  PIPELINE FINISHED")


if __name__ == "__main__":
    main()
