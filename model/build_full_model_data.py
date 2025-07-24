"""
build_full_model_data.py
────────────────────────
Builds a daily modelling dataset that joins:

  • Tesla stock prices  (Data/clean/clean_stock.csv)
  • Elon Musk tweets    (Data/clean/clean_musk_tweets.csv)
  • Elon Musk retweets  (Data/clean/clean_musk_retweets.csv)

Features added:
  • Technical indicators (MAs, volatility, next‑day return)
  • Sentiment + engagement aggregates for tweets & retweets
  • TF‑IDF summary (top‑5 weight sum, keywords, ±1‑day lag/lead)

Config flags let you:
  · DROP_MISSING_ROWS    – drop rows missing look‑backs or target
  · DROP_NO_SOCIAL_ROWS  – drop days without tweets & retweets
  · DROP_OHLCV_COLUMNS   – drop raw OHLCV columns after feature calc

Usage (from project root):
    $ python model/build_full_model_data.py
"""
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG FLAGS – tweak as required
# ──────────────────────────────────────────────────────────────────────────────
DROP_MISSING_ROWS = True  # drop rows lacking price_ma_5, price_ma_10, or next_day_pct_change
DROP_NO_SOCIAL_ROWS = True  # drop rows with zero tweets AND zero retweets
DROP_OHLCV_COLUMNS = True  # drop Open, High, Low, Price, Volume from output

# ──────────────────────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────────────────────
import re
from pathlib import Path
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer


# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Strip URLs, @mentions, '#', literal 'nan', collapse whitespace."""
    if not isinstance(text, str) or text.lower() == "nan":
        return ""
    text = re.sub(r"http\S+", "", text)  # URLs
    text = re.sub(r"@\w+", "", text)  # mentions
    text = re.sub(r"#", "", text)  # keep word, drop '#'
    text = re.sub(r"\bnan\b", "", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


def _sentimentize(df: pd.DataFrame, col: str) -> pd.Series:
    """Vectorised VADER compound sentiment score."""
    nltk.download("vader_lexicon", quiet=True)
    sia = SentimentIntensityAnalyzer()
    return df[col].apply(lambda t: sia.polarity_scores(t)["compound"])


def _aggregate_posts(df: pd.DataFrame, date_col: str = "timestamp") -> pd.DataFrame:
    """Aggregate tweets/retweets at daily granularity."""
    df["date"] = df[date_col].dt.date
    return df.groupby("date").agg(
        post_count=(date_col, "size"),
        sentiment_mean=("sentiment", "mean"),
        sentiment_std=("sentiment", "std"),
        median_len=("text_len", "median"),
        std_len=("text_len", "std"),
        sum_retweets=("retweetCount", "sum"),
        sum_replies=("replyCount", "sum"),
        sum_likes=("likeCount", "sum"),
        sum_quotes=("quoteCount", "sum"),
        sum_views=("viewCount", "sum"),
        sum_bookmarks=("bookmarkCount", "sum"),
        all_posts=("clean_text", lambda texts: " <SEP> ".join(texts)),
    ).reset_index()


# ──────────────────────────────────────────────────────────────────────────────
# Stock loader / engineering
# ──────────────────────────────────────────────────────────────────────────────
def load_and_engineer_stock(project_root: Path) -> pd.DataFrame:
    """
    Load the clean Tesla price file and add:
        - next‑day % return (target)
        - 5‑ & 10‑day moving averages
        - 5‑day return volatility
        - 0‑day return (today's % change)
        - 3‑ and 7‑day momentum (% change over window)
    """
    csv = project_root / "Data" / "clean" / "clean_stock.csv"
    df = pd.read_csv(csv, parse_dates=["Date"])

    # Harmonise column names
    if "Close" in df.columns and "Price" not in df.columns:
        df = df.rename(columns={"Close": "Price"})
    if "Price" not in df.columns:
        raise KeyError(f"{csv} must contain a 'Price' column.")

    df["date"] = df["Date"].dt.date

    # Targets & technicals
    df["next_day_pct_change"] = df["Price"].pct_change().shift(-1) * 100
    df["price_ma_5d"] = df["Price"].rolling(5).mean()
    df["price_ma_10d"] = df["Price"].rolling(10).mean()
    df["pct_vol_5d"] = df["Price"].pct_change().rolling(5).std()

    # price‑action features
    df["pct_return_1d"] = df["Price"].pct_change() * 100  # today's return
    df["pct_return_3d"] = df["Price"].pct_change(3) * 100  # 3‑day momentum
    df["pct_return_7d"] = df["Price"].pct_change(7) * 100  # 7‑day momentum

    return df[[
        "date", "Open", "High", "Low", "Price", "Volume",
        "next_day_pct_change",
        "pct_return_1d", "pct_return_3d", "pct_return_7d",
        "price_ma_5d", "price_ma_10d", "pct_vol_5d",
    ]].copy()


# ──────────────────────────────────────────────────────────────────────────────
# Tweets & Retweets processors
# ──────────────────────────────────────────────────────────────────────────────
def process_and_aggregate_tweets(project_root: Path) -> pd.DataFrame:
    csv = project_root / "Data" / "clean" / "clean_musk_tweets.csv"
    tw = pd.read_csv(csv, parse_dates=["timestamp"])

    tw["clean_text"] = tw["text"].fillna("").astype(str).apply(clean_text)
    tw["text_len"] = tw["clean_text"].str.len()
    tw["sentiment"] = _sentimentize(tw, "clean_text")
    return _aggregate_posts(tw)


def process_and_aggregate_retweets(project_root: Path) -> pd.DataFrame:
    csv = project_root / "Data" / "clean" / "clean_musk_retweets.csv"
    rt = pd.read_csv(csv, parse_dates=["timestamp"])

    # Determine RT content column
    if "tweet" in rt.columns:
        body_col = "tweet"
    elif "original_content" in rt.columns:
        body_col = "original_content"
    elif "text" in rt.columns:
        body_col = "text"
    else:
        raise KeyError(f"No retweet text column found in {csv}")

    rt["clean_text"] = rt[body_col].fillna("").astype(str).apply(clean_text)
    rt["text_len"] = rt["clean_text"].str.len()
    rt["sentiment"] = _sentimentize(rt, "clean_text")
    return _aggregate_posts(rt)


# ──────────────────────────────────────────────────────────────────────────────
# TF‑IDF summary
# ──────────────────────────────────────────────────────────────────────────────
def add_tfidf_summary(df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    docs = df["all_posts"].fillna("").astype(str).str.replace("nan", "").tolist()
    vec = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), stop_words="english")
    mat = vec.fit_transform(docs).toarray()
    feats = vec.get_feature_names_out()

    weight_sums, kws = [], []
    for row, txt in zip(mat, docs):
        if not txt.strip():
            weight_sums.append(0.0);
            kws.append("")
        else:
            idx = row.argsort()[-top_k:][::-1]
            weight_sums.append(row[idx].sum())
            kws.append(",".join(feats[i] for i in idx))

    out = df.copy()
    out["tfidf_top5_weight_sum"] = weight_sums
    out["tfidf_top_keywords"] = kws
    out = out.sort_values("date").reset_index(drop=True)
    out["tfidf_weight_sum_lag1"] = out["tfidf_top5_weight_sum"].shift(1)
    out["tfidf_weight_sum_lead1"] = out["tfidf_top5_weight_sum"].shift(-1)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    project_root = Path(__file__).resolve().parent.parent  # model/ → project root

    # Stock
    stock_df = load_and_engineer_stock(project_root)

    # Social aggregates
    tweets_df = process_and_aggregate_tweets(project_root)
    retweets_df = process_and_aggregate_retweets(project_root)

    # Combine social metrics
    daily = tweets_df.merge(
        retweets_df,
        on="date",
        how="outer",
        suffixes=("_tweet", "_retweet")
    ).fillna(0)

    daily["all_posts"] = (
            daily["all_posts_tweet"].astype(str)
            + " <SEP> "
            + daily["all_posts_retweet"].astype(str)
    ).str.strip(" <SEP> ")
    daily.drop(columns=["all_posts_tweet", "all_posts_retweet"], inplace=True)

    # Merge with stock & add TF‑IDF
    merged = stock_df.merge(daily, on="date", how="left")
    final_df = add_tfidf_summary(merged, top_k=5)

    # Fill values that represent "no social activity"
    zero_fill_cols = [
        c for c in final_df.columns
        if c.startswith(("post_count",  # post_count_tweet / _retweet
                         "sentiment_mean",  # sentiment_mean_tweet / _retweet
                         "median_len", "std_len",
                         "sum_retweets", "sum_replies",
                         "sum_likes", "sum_quotes",
                         "sum_views", "sum_bookmarks"))
    ]
    final_df[zero_fill_cols] = final_df[zero_fill_cols].fillna(0)

    # Optional row/column filtering
    if DROP_MISSING_ROWS:
        final_df = final_df.dropna(subset=["price_ma_5d", "price_ma_10d", "next_day_pct_change"])
    if DROP_NO_SOCIAL_ROWS:
        final_df = final_df[(final_df["post_count_tweet"] > 0) | (final_df["post_count_retweet"] > 0)].copy()
    if DROP_OHLCV_COLUMNS:
        final_df = final_df.drop(columns=["Open", "High", "Low", "Price", "Volume"])

    # Save
    out_dir = project_root / "Data" / "model"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "model_data_full.csv"
    final_df.to_csv(out_csv, index=False)

    print(
        f"✅ Saved full modelling dataset to {out_csv} "
        f"({len(final_df):,} rows, {len(final_df.columns)} columns)"
    )


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
