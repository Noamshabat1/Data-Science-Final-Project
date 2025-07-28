"""
build_data.py
=============

Merge Elon‑Musk Twitter data with Tesla OHLC and engineer
sentiment, TF‑IDF and MiniLM‑embedding features.
Outputs
-------
• data/model/model_data.csv
• data/model/data_overview.png   (Figure1 for the report)
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm.auto import tqdm

# ───────────────────── configuration ────────────────────────────────
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
CLEAN_DIR: Final[Path] = PROJECT_ROOT / "data" / "clean"
MODEL_DIR: Final[Path] = PROJECT_ROOT / "data" / "model"

NUMERIC_COLS: Final[List[str]] = [
    "retweetCount", "replyCount", "likeCount",
    "quoteCount", "viewCount", "bookmarkCount",
]

TFIDF_MAX_FEAT: Final[int] = 1_000
TFIDF_SVD_DIM: Final[int] = 10
EMBED_DIM: Final[int] = 8
EMBED_BATCH: Final[int] = 256
EMBED_MODEL: Final[str] = "all-MiniLM-L6-v2"


# ───────────────────── helper: social loaders ───────────────────────
def _load_social(fname: str, *, source: str, text_col: str = "text") -> pd.DataFrame:
    """
    Load one cleaned Twitter CSV and harmonise column names.

    Parameters
    ----------
    fname : str         Filename in *data/clean/*.
    source : str        Tag: 'tweets', 'retweets', or 'replies'.
    text_col : str      Column containing the post text.

    Returns
    -------
    pd.DataFrame with *NUMERIC_COLS + timestamp + text + source*.
    """
    df = pd.read_csv(CLEAN_DIR / fname, usecols=NUMERIC_COLS + ["timestamp", text_col])
    df = df.rename(columns={text_col: "text"})
    df["source"] = source
    return df


def _aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse raw posts into one row per calendar day.

    * Sums engagement metrics
    * Concatenates all post texts (chronological, '<SEP>' delimiter)
    """
    df[NUMERIC_COLS] = df[NUMERIC_COLS].fillna(0)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date

    text_blob = (
        df.groupby("date")["text"]
        .apply(lambda s: " <SEP> ".join(s.fillna("").astype(str)))
        .rename("all_posts")
    )
    metrics = df.groupby("date")[NUMERIC_COLS].sum()

    out = metrics.join(text_blob).reset_index()
    out["timestamp"] = pd.to_datetime(out["date"])
    return out.drop(columns="date")


def _merge_stock(social: pd.DataFrame) -> pd.DataFrame:
    """
    Align social features with Tesla close price using a backward as‑of merge.
    """
    stock = (
        pd.read_csv(CLEAN_DIR / "clean_tesla_stock.csv", usecols=["Date", "Close"])
        .rename(columns={"Date": "timestamp", "Close": "tesla_close"})
    )
    stock["timestamp"] = pd.to_datetime(stock["timestamp"], errors="coerce")
    stock = stock.dropna(subset=["timestamp"]).sort_values("timestamp")

    merged = pd.merge_asof(social.sort_values("timestamp"), stock, on="timestamp", direction="backward")
    merged["tesla_close"] = merged["tesla_close"].ffill().bfill()
    return merged


# ───────────────────── feature generators ───────────────────────────
def add_sentiment_features(df: pd.DataFrame) -> pd.DataFrame:
    """VADER compound score (−1 … +1)."""
    print("· Generating sentiment feature")
    analyser = SentimentIntensityAnalyzer()
    df["sentiment_compound"] = df["all_posts"].apply(lambda t: analyser.polarity_scores(str(t))["compound"])
    return df


def add_tfidf_features(df: pd.DataFrame) -> pd.DataFrame:
    """10‑dim TF‑IDF topic vectors (tfidf_0 … tfidf_9)."""
    print("· Generating TF‑IDF features")
    vec = TfidfVectorizer(
        max_features=TFIDF_MAX_FEAT,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=2,
        max_df=0.95,
    )
    matrix = vec.fit_transform(df["all_posts"])
    svd = TruncatedSVD(TFIDF_SVD_DIM, random_state=42)
    reduced = svd.fit_transform(matrix)
    for i in range(TFIDF_SVD_DIM):
        df[f"tfidf_{i}"] = reduced[:, i]
    return df


def add_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    """8 principal components of MiniLM sentence embeddings."""
    print("· Generating MiniLM embeddings")
    model = SentenceTransformer(EMBED_MODEL)
    embeds = np.empty((len(df), model.get_sentence_embedding_dimension()))

    for start in tqdm(range(0, len(df), EMBED_BATCH), desc="Embedding"):
        end = start + EMBED_BATCH
        embeds[start:end] = model.encode(df["all_posts"].iloc[start:end].tolist(), convert_to_numpy=True)

    for i in range(EMBED_DIM):
        df[f"embed_{i}"] = embeds[:, i]
    return df


# ───────────────────── reporting figure helper ───────────────────────
def create_data_overview_plot(df: pd.DataFrame, save_to: Path) -> None:
    """
    4‑panel Figure1– Data‑processing overview.

    Panels: (A) engagement mix – log‑scaled,
            (B) sentiment histogram,
            (C) Tesla close price (log‑$),
            (D) mean |TF‑IDF component|.
    """
    import matplotlib.pyplot as plt

    mean_counts = df[NUMERIC_COLS].mean()
    tfidf_cols = [c for c in df.columns if c.startswith("tfidf_")]
    tfidf_mean = df[tfidf_cols].abs().mean()

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    (ax0, ax1), (ax2, ax3) = axes

    # (A) Engagement mix  –– log‑scaled
    ax0.bar(mean_counts.index, mean_counts.values, color="tab:orange", alpha=0.75)
    ax0.set_yscale("log")
    ax0.set_title("(A) Avg. daily engagement mix (log scale)")
    ax0.set_ylabel("Mean count (log)")
    ax0.tick_params(axis="x", rotation=35)

    # (B) Sentiment histogram
    ax1.hist(df["sentiment_compound"], bins=30, color="tab:green", alpha=0.75)
    ax1.set_title("(B) VADER compound distribution")
    ax1.set_xlabel("Compound score")
    ax1.set_ylabel("Frequency")

    # (C) Tesla close price
    ax2.plot(df["timestamp"], df["tesla_close"], color="tab:blue")
    ax2.set_yscale("log")
    ax2.set_title("(C) Tesla close price (log $)")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Close price ($, log)")

    # (D) TF‑IDF magnitudes
    ax3.bar(tfidf_mean.index, tfidf_mean.values, color="tab:purple", alpha=0.8)
    ax3.set_title("(D) Mean |TF‑IDF component|")
    ax3.set_ylabel("Mean abs value")
    ax3.tick_params(axis="x", rotation=35)

    fig.suptitle("Figure 1 – Data‑processing overview", fontsize=14, y=0.98)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    fig.savefig(save_to, dpi=300)
    plt.close()
    print(f"✔ Saved Figure 1 to {save_to}")


# ───────────────────── main pipeline ────────────────────────────────
def load_and_merge_data() -> pd.DataFrame:
    """
    Build feature frame + overview figure, write to disk, return DataFrame.
    """
    print("\n=== BUILDING DATASET ===")

    tweets = _load_social("clean_musk_tweets.csv", source="tweets")
    retweets = _load_social("clean_musk_retweets.csv", source="retweets", text_col="tweet")
    replies = _load_social("clean_musk_replies.csv", source="replies")

    social = _aggregate_daily(pd.concat([tweets, retweets, replies], ignore_index=True))
    merged = _merge_stock(social)

    merged = add_sentiment_features(merged)
    merged = add_tfidf_features(merged)
    merged = add_embeddings(merged)

    # report figure
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    create_data_overview_plot(merged, MODEL_DIR / "data_overview.png")

    # final clean‑up
    merged.drop(columns="all_posts", inplace=True)
    assert merged.isna().sum().sum() == 0, "NaNs present after feature creation"

    out_path = MODEL_DIR / "model_data.csv"
    merged.to_csv(out_path, index=False)
    print(f"✔ Saved dataset to {out_path}  shape={merged.shape}\n")
    return merged


# ───────────────────── CLI entry point ───────────────────────────────
if __name__ == "__main__":
    df_final = load_and_merge_data()
    print(df_final.head())
