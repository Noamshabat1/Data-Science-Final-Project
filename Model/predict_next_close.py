#!/usr/bin/env python3
"""
predict_next_close.py — load saved XGB model and today’s tweets,
output probability that tomorrow’s TSLA close is higher.
"""
from pathlib import Path
import pandas as pd, numpy as np
from xgboost import XGBClassifier
from FinalProject_ML_pipeline import (
    DATA_CLEAN, CONFIG, _standardise_post, tfidf_select
)

MODEL_PATH = Path(__file__).with_name("xgb_tsla.json")

def load_today_posts() -> pd.DataFrame:
    frames=[]
    for fn in CONFIG["TWEET_FILES"]:
        p=DATA_CLEAN/fn
        if p.exists():
            df=pd.read_csv(p)
            df=_standardise_post(df, "src")  # src value unused
            df=df[df["date"]==pd.Timestamp("now", tz="Asia/Jerusalem").date()]
            frames.append(df)
    return pd.concat(frames, ignore_index=True)

def main():
    posts=load_today_posts()
    if posts.empty:
        print("No posts for today yet.")
        return
    posts_day=posts.assign(text_cat=lambda d:d["text"]).groupby("date").agg(
        text_cat=("text_cat"," ".join),
        tweet_count=("text","size"),
        sentiment_mean=("sentiment","mean")
    )

    text_feats=tfidf_select(
        posts_day["text_cat"], np.zeros(len(posts_day)),
        max_feats=6000, keep=CONFIG["TFIDF_MAX"]
    )
    base=pd.DataFrame({
        "sentiment_mean":posts_day["sentiment_mean"],
        "tweet_count":posts_day["tweet_count"],
    })
    X=pd.concat([base, text_feats], axis=1).astype("float32")

    model=XGBClassifier(); model.load_model(MODEL_PATH)
    prob=model.predict_proba(X)[:,1][0]
    print(f"Probability TSLA closes UP tomorrow: {prob:.2%}")

if __name__=="__main__":
    main()
