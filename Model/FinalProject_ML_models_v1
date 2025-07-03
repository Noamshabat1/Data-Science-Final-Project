#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FinalProject_ML_pipeline.py
===========================

Predict next-day TSLA close-to-close return using Elon-Musk Twitter data
and basic market metrics.

Key features
------------
• RandomisedSearchCV on 6 XGB hyper-parameters
• Market features: lag-1 return and 5-day realised volatility
• Optional FinBERT sentiment (flag --finbert)
• TimeSeriesSplit CV for Ridge + XGB
• Training log-loss curve for XGB
• SHAP explainability (skips if shap missing)
• Saves best model → Model/xgb_tsla.json
"""
from __future__ import annotations
import argparse, logging, sys, warnings
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, r2_score, precision_recall_curve,
    accuracy_score, roc_auc_score, ConfusionMatrixDisplay,
    RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import uniform, randint
from xgboost import XGBClassifier

# optional SHAP
try:
    import shap
    SHAP_OK = True
except ModuleNotFoundError:
    SHAP_OK = False

# ─────────────────── paths / config ──────────────────────────────
ROOT        = Path(__file__).resolve().parents[1]
DATA_CLEAN  = ROOT / "Data" / "clean"
DATA_ORIG   = ROOT / "Data" / "original"
MODEL_OUT   = ROOT / "Model" / "xgb_tsla.json"

CONFIG: Dict[str, object] = {
    "RANDOM_STATE": 42,
    "MAX_JOBS": 8,
    "TWEET_FILES": [
        "clean_musk_tweets.csv",
        "clean_musk_retweets.csv",
        "clean_musk_replies.csv",
    ],
    "STOCK_FILE": "tesla_stock_data_2000_2025.csv",
    "KEYWORDS": [
        "model 3", "model y", "cybertruck", "ai",
        "robot", "teslabot", "fsd", "tesla energy", "spacex",
    ],
    "TFIDF_MAX": 400,
    "BASE_XGB": dict(
        n_estimators=600, objective="binary:logistic",
        eval_metric="logloss", tree_method="hist",
    ),
    "HPARAM_SPACE": {
        "learning_rate":    uniform(0.01, 0.19),
        "max_depth":        randint(3, 9),
        "subsample":        uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "gamma":            uniform(0.0, 4.0),
        "reg_lambda":       uniform(0.5, 4.5),
    },
}

# ─────────────────── logging helper ──────────────────────────────
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)6s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)
def banner(msg:str)->None:
    log.info("="*90); log.info(msg); log.info("="*90)

def _req(p:Path)->None:
    if not p.exists():
        raise FileNotFoundError(p)

# ─────────────────── Tweet utilities ─────────────────────────────
def _finbert_sentiment(texts:pd.Series)->pd.Series:
    """Return FinBERT polarity or neutral 0.0 if transformers missing."""
    try:
        from transformers import pipeline
        pipe = pipeline("sentiment-analysis",
                        model="ProsusAI/finbert",
                        tokenizer="ProsusAI/finbert",
                        device=-1)
        out   = pipe(texts.tolist(), truncation=True, batch_size=32)
        score = {"positive": 1, "neutral": 0, "negative": -1}
        return pd.Series([score[o["label"].lower()] * o["score"] for o in out],
                         index=texts.index)
    except Exception as e:
        log.warning("FinBERT unavailable (%s) – neutral sentiment applied.", e)
        return pd.Series(np.zeros(len(texts)), index=texts.index)

def _std_post(df:pd.DataFrame, src:str)->pd.DataFrame:
    txt = ("cleanedReplyContent" if "cleanedReplyContent" in df.columns
           else "fullText" if "fullText" in df.columns
           else "originalContent")
    df["text"]      = df[txt].fillna("")
    df["sentiment"] = df.get("sentiment", 0.0)
    ts = "createdAt" if "createdAt" in df.columns else "created_at"
    df["datetime"]  = pd.to_datetime(df[ts], utc=True).dt.tz_convert("Asia/Jerusalem")
    df["date"]      = df["datetime"].dt.date
    df["source"]    = src
    return df[["date", "text", "sentiment"]]

def load_posts(finbert:bool=False)->pd.DataFrame:
    frames = []
    for fn in CONFIG["TWEET_FILES"]:
        p   = DATA_CLEAN / fn; _req(p)
        src = fn.split("_")[2].split(".")[0]
        frames.append(_std_post(pd.read_csv(p), src))
    posts = pd.concat(frames, ignore_index=True)
    if finbert:
        log.info("Running FinBERT sentiment …")
        posts["sentiment"] = _finbert_sentiment(posts["text"])
    return posts

def agg_daily(posts:pd.DataFrame)->pd.DataFrame:
    # keyword flags
    for kw in CONFIG["KEYWORDS"]:
        posts[f"kw_{kw.replace(' ','_')}"] = posts["text"].str.contains(
            kw, case=False, regex=False, na=False).astype(int)
    agg = {
        "text_cat": ("text", " ".join),
        "tweet_count": ("text", "size"),
        "sentiment_mean": ("sentiment", "mean"),
    } | {f"kw_{k.replace(' ','_')}": (f"kw_{k.replace(' ','_')}", "sum")
         for k in CONFIG["KEYWORDS"]}
    return posts.groupby("date").agg(**agg).reset_index()

# ─────────────────── Market utilities ────────────────────────────
def _detect(raw:pd.DataFrame, opts:list[str])->str:
    for c in opts:
        if c in raw.columns:
            return c
    for col in raw.columns:
        if pd.to_datetime(raw[col].head(10), errors="coerce").notna().sum() >= 8:
            return col
    raise ValueError

def load_market()->pd.DataFrame:
    p = DATA_ORIG / CONFIG["STOCK_FILE"]; _req(p)
    raw = pd.read_csv(p)
    dt  = _detect(raw, ["datetime","timestamp","date","Datetime","Date"])
    cc  = _detect(raw, ["Close","Adj Close","close"])
    raw[dt] = pd.to_datetime(raw[dt], errors="coerce")
    raw[cc] = pd.to_numeric(raw[cc], errors="coerce")
    close = raw[[dt, cc]].dropna().groupby(raw[dt].dt.date)[cc].last()
    return pd.DataFrame({
        "close": close,
        "ret_fwd_1d": close.pct_change().shift(-1),
        "ret_lag_1d": close.pct_change(),
        "vol_5d": close.pct_change().rolling(5).std(),
    }).dropna()

def merge_daily(p:pd.DataFrame, m:pd.DataFrame)->pd.DataFrame:
    return pd.merge(p, m, left_on="date", right_index=True, how="inner")

# ─────────────────── TF-IDF helper ───────────────────────────────
def tfidf_select(text:pd.Series, y:pd.Series,*,max_feats:int,keep:int)->pd.DataFrame:
    vec = TfidfVectorizer(max_features=max_feats, ngram_range=(1,2), min_df=3)
    tf  = vec.fit_transform(text)
    corr = np.array([np.corrcoef(tf[:,i].toarray().ravel(), y)[0,1] for i in range(tf.shape[1])])
    sel  = np.argsort(np.nan_to_num(np.abs(corr)))[-keep:]
    return pd.DataFrame(tf[:,sel].toarray(),
        columns=[f"tfidf_{t}" for t in vec.get_feature_names_out()[sel]],
        index=text.index)

# ─────────────────── Ridge baseline ───────────────────────────────
def ridge_baseline(df:pd.DataFrame, split:float)->None:
    banner("📊  Ridge baseline")
    kw = [f"kw_{k.replace(' ','_')}" for k in CONFIG["KEYWORDS"]]
    X  = df[["sentiment_mean","tweet_count","ret_lag_1d","vol_5d",*kw]].to_numpy()
    y  = df["ret_fwd_1d"].to_numpy()
    idx = int(split * len(df)); Xtr,Xte = X[:idx], X[idx:]; ytr,yte = y[:idx], y[idx:]
    sc  = StandardScaler().fit(Xtr); Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    gcv = GridSearchCV(Ridge(random_state=CONFIG["RANDOM_STATE"]),
                       {"alpha":[10**i for i in range(-4,3)]},
                       cv=TimeSeriesSplit(5),
                       scoring="neg_root_mean_squared_error").fit(Xtr_s,ytr)
    best = gcv.best_params_["alpha"]
    preds = gcv.predict(Xte_s)
    rmse  = np.sqrt(mean_squared_error(yte, preds))
    log.info("α=%g  RMSE=%.5f  R²=%.4f", best, rmse, r2_score(yte,preds))
    plt.figure(figsize=(4,4))
    plt.scatter(yte, preds, alpha=.4)
    plt.axline((0,0), slope=1, color="k")
    plt.title(f"Ridge α={best}"); plt.tight_layout(); plt.show()

# ─────────────────── XGB model ────────────────────────────────────
def xgb_model(df:pd.DataFrame, *, split:float, tfidf_max:int, n_iter:int)->None:
    banner("🚀  XGBoost – RandomisedSearchCV")
    tf_df = tfidf_select(df["text_cat"], df["ret_fwd_1d"], max_feats=6000, keep=tfidf_max)
    base  = ["sentiment_mean","tweet_count","ret_lag_1d","vol_5d"] + \
            [f"kw_{k.replace(' ','_')}" for k in CONFIG["KEYWORDS"]]
    X = pd.concat([df[base], tf_df], axis=1).astype("float32")
    y = (df["ret_fwd_1d"] > 0).astype(int)

    idx = int(split * len(X)); Xtr,Xte = X.iloc[:idx], X.iloc[idx:]; ytr,yte = y.iloc[:idx], y.iloc[idx:]

    base_params = dict(CONFIG["BASE_XGB"],
        random_state=CONFIG["RANDOM_STATE"], n_jobs=CONFIG["MAX_JOBS"],
        scale_pos_weight=(ytr==0).sum()/max((ytr==1).sum(),1))

    search = RandomizedSearchCV(
        XGBClassifier(**base_params),
        CONFIG["HPARAM_SPACE"],
        n_iter=n_iter, cv=TimeSeriesSplit(4),
        scoring="roc_auc", n_jobs=CONFIG["MAX_JOBS"],
        random_state=CONFIG["RANDOM_STATE"], verbose=1).fit(Xtr,ytr)

    best = search.best_params_
    log.info("Best params: %s", best)
    model = XGBClassifier(**base_params | best)
    model.fit(Xtr, ytr, eval_set=[(Xtr, ytr)], verbose=False)

    # --- evaluation on hold-out ---
    prob = model.predict_proba(Xte)[:,1]
    p,r,t = precision_recall_curve(yte, prob); f1 = 2*p*r/(p+r+1e-9)
    tau = t[int(f1.argmax())]; yhat = (prob >= tau).astype(int)
    log.info("τ≈%.3f  F1=%.3f  Acc=%.3f  AUC=%.3f",
             tau, f1.max(), accuracy_score(yte,yhat), roc_auc_score(yte,prob))

    # training loss
    plt.figure(figsize=(6,3))
    plt.plot(model.evals_result()["validation_0"]["logloss"])
    plt.title("XGB training log-loss"); plt.xlabel("Boost round"); plt.ylabel("log-loss")
    plt.tight_layout(); plt.show()

    RocCurveDisplay.from_predictions(yte, prob); plt.title("ROC — XGB"); plt.show()
    PrecisionRecallDisplay.from_predictions(yte, prob); plt.title("PR Curve — XGB"); plt.show()
    ConfusionMatrixDisplay.from_predictions(yte, yhat, cmap="viridis"); plt.title("Confusion Matrix — XGB"); plt.show()

    if SHAP_OK:
        explainer = shap.TreeExplainer(model)
        shap.summary_plot(explainer.shap_values(Xte), Xte, max_display=20, plot_size=(9,5), show=False)
        plt.title("SHAP – top 20 features"); plt.tight_layout(); plt.show()
    else:
        log.warning("SHAP not installed – skipping SHAP plot.")

    model.save_model(MODEL_OUT)
    log.info("Saved model → %s", MODEL_OUT)

# ─────────────────── main ─────────────────────────────────────────
def main()->None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=float, default=.7, help="train/test split fraction")
    ap.add_argument("--tfidf-max", type=int, default=CONFIG["TFIDF_MAX"])
    ap.add_argument("--finbert", action="store_true", help="recompute FinBERT sentiment")
    ap.add_argument("--search-iter", type=int, default=40, help="random-search iterations")
    args = ap.parse_args()

    banner("🏁  PIPELINE START")
    posts  = load_posts(finbert=args.finbert)
    tweets = agg_daily(posts)
    market = load_market()
    data   = merge_daily(tweets, market)
    banner(f"Merged dataset: {len(data):,} days")

    ridge_baseline(data, split=args.split)
    xgb_model(data, split=args.split, tfidf_max=args.tfidf_max, n_iter=args.search_iter)
    banner("✅  PIPELINE FINISHED")

if __name__ == "__main__":
    main()
