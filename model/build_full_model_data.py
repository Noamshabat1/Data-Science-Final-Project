"""
build_full_model_data.py — 2025-07-24

Builds a daily modelling dataset by merging:
  • Tesla OHLCV + engineered technical features
  • S&P 500 index OHLCV + parallel technical features (prefixed sp500_)
  • Relative TSLA vs S&P features (return/vol spreads)
  • Elon Musk tweets & retweets (emoji-aware, VADER sentiment)
  • TF‑IDF summary of posts (lag only → no leakage)

All VADER-based columns end with `_VADER`.

Edit the FLAGS section below to control what’s dropped.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Tuple, List

import nltk
import numpy as np
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer

# ────────────────────────────────────────────────────────────────
# ===== FLAGS & PARAMS (edit here, no CLI) =======================
# ────────────────────────────────────────────────────────────────
DROP_MISSING_ROWS: bool      = True   # drop rows where essential feats/target are NaN
DROP_NO_SOCIAL_ROWS: bool    = True   # drop days with 0 tweets & 0 retweets
DROP_OHLCV_COLS: bool        = False  # drop raw TSLA Open/High/Low/Price/Volume
DROP_SP500_OHLCV_COLS: bool  = False  # drop raw S&P500 OHLCV
DROP_ALL_POSTS_TEXT: bool    = True   # drop the giant 'all_posts' text blob after TF-IDF
DROP_CONSTANT_COLS: bool     = True   # drop columns with only one unique value

# Columns to always drop if present (safety)
ALWAYS_DROP: List[str] = [
    "tfidf_weight_sum_lead1",           # leakage from older versions
    "sp500_next_day_pct_change",        # future market move relative to same-day target
]

# TF‑IDF configuration
TFIDF_TOP_K: int        = 5
TFIDF_MAX_FEATURES: int = 1000
TFIDF_NGRAM_LOW: int    = 1
TFIDF_NGRAM_HIGH: int   = 2

# Output path
OUT_FULL = "Data/model/model_data_full.csv"

# ────────────────────────────────────────────────────────────────
# Paths / Globals
# ────────────────────────────────────────────────────────────────
PROJ_ROOT      = Path(__file__).resolve().parents[1]
EMOJI_LEX_PATH = PROJ_ROOT / "Data" / "original" / "emoji_utf8_lexicon.txt"

_SIA_TEXT: SentimentIntensityAnalyzer | None = None

# ────────────────────────────────────────────────────────────────
# Utils
# ────────────────────────────────────────────────────────────────
def drop_tz(s: pd.Series) -> pd.Series:
    """Force tz-naive datetime64[ns] (works for tz-aware or naive input)."""
    return pd.to_datetime(s, utc=True).dt.tz_localize(None)


def clean_text(text: str) -> str:
    """Remove URLs/@/#/literal 'nan', collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"\bnan\b", "", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


def ensure_sia_text() -> SentimentIntensityAnalyzer:
    global _SIA_TEXT
    if _SIA_TEXT is None:
        nltk.download("vader_lexicon", quiet=True)
        _SIA_TEXT = SentimentIntensityAnalyzer()
    return _SIA_TEXT


# ────────────────────────────────────────────────────────────────
# Emoji machinery
# ────────────────────────────────────────────────────────────────
def load_emoji_lexicon(path: Path) -> dict[str, str]:
    df = pd.read_csv(path, sep="\t", header=None, names=["emoji", "desc"], engine="python")
    df = df.dropna(subset=["desc"])
    return dict(zip(df["emoji"], df["desc"]))


EMOJI_MAP = load_emoji_lexicon(EMOJI_LEX_PATH)
EMOJI_PATTERN = re.compile("|".join(map(re.escape, EMOJI_MAP.keys())))

_SIA_EMOJI = SentimentIntensityAnalyzer()
EMOJI_SENTIMENT = {e: _SIA_EMOJI.polarity_scores(d)["compound"] for e, d in EMOJI_MAP.items()}


def replace_emojis(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return EMOJI_PATTERN.sub(lambda m: " " + EMOJI_MAP[m.group(0)].replace(" ", "_") + " ", text)


def count_emojis(text: str) -> Tuple[int, int]:
    hits = EMOJI_PATTERN.findall(text or "")
    return len(hits), len(set(hits))


def emoji_sent_mean(text: str) -> float:
    vals = [EMOJI_SENTIMENT[e] for e in EMOJI_PATTERN.findall(text or "")]
    return float(np.mean(vals)) if vals else 0.0


# ────────────────────────────────────────────────────────────────
# Loaders & feature builders
# ────────────────────────────────────────────────────────────────
def load_and_engineer_stock() -> pd.DataFrame:
    """
    Load Tesla cleaned OHLCV and engineer technical features.
    File expected: Data/clean/clean_tesla_stock.csv
    Target = next_day_pct_change (TSLA).
    """
    csv = PROJ_ROOT / "Data" / "clean" / "clean_tesla_stock.csv"
    if not csv.exists():
        raise FileNotFoundError(f"Missing Tesla file: {csv}")

    df = pd.read_csv(csv)

    # detect/parse date
    date_col = "date" if "date" in df.columns else "Date"
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)

    # normalize names
    if "Price" not in df.columns:
        if "Close" in df.columns: df = df.rename(columns={"Close": "Price"})
        elif "close" in df.columns: df = df.rename(columns={"close": "Price"})
    rename_map = {"open": "Open", "high": "High", "low": "Low", "volume": "Volume"}
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    df["date"] = drop_tz(df[date_col])

    # features
    df["next_day_pct_change"] = df["Price"].pct_change().shift(-1) * 100
    df["pct_return_1d"] = df["Price"].pct_change() * 100
    df["pct_return_3d"] = df["Price"].pct_change(3) * 100
    df["pct_return_7d"] = df["Price"].pct_change(7) * 100
    df["price_ma_5d"] = df["Price"].rolling(5).mean()
    df["price_ma_10d"] = df["Price"].rolling(10).mean()
    df["pct_vol_5d"] = df["Price"].pct_change().rolling(5).std()

    keep = [
        "date", "Open", "High", "Low", "Price", "Volume",
        "next_day_pct_change",
        "pct_return_1d", "pct_return_3d", "pct_return_7d",
        "price_ma_5d", "price_ma_10d", "pct_vol_5d",
    ]
    return df[keep].copy()


def load_and_engineer_sp500() -> pd.DataFrame:
    """
    Load clean S&P 500 OHLCV and build parallel market features.
    Prefix everything with sp500_ to avoid collisions.
    """
    csv = PROJ_ROOT / "Data" / "clean" / "clean_sp500_stock.csv"
    if not csv.exists():
        raise FileNotFoundError(f"Missing S&P500 file: {csv}")

    df = pd.read_csv(csv, parse_dates=["date"]).sort_values("date")
    df["date"] = drop_tz(df["date"])

    # pct changes & rolling stats on close
    df["sp500_pct_return_1d"] = df["close"].pct_change() * 100
    df["sp500_pct_return_3d"] = df["close"].pct_change(3) * 100
    df["sp500_pct_return_7d"] = df["close"].pct_change(7) * 100
    df["sp500_price_ma_5d"]   = df["close"].rolling(5).mean()
    df["sp500_price_ma_10d"]  = df["close"].rolling(10).mean()
    df["sp500_pct_vol_5d"]    = df["close"].pct_change().rolling(5).std()

    df = df.rename(columns={
        "open":   "sp500_open",
        "high":   "sp500_high",
        "low":    "sp500_low",
        "close":  "sp500_close",
        "volume": "sp500_volume"
    })

    keep = [
        "date",
        "sp500_open","sp500_high","sp500_low","sp500_close","sp500_volume",
        "sp500_pct_return_1d","sp500_pct_return_3d","sp500_pct_return_7d",
        "sp500_price_ma_5d","sp500_price_ma_10d","sp500_pct_vol_5d",
    ]
    return df[keep].copy()


def add_relative_tsla_vs_sp500(df: pd.DataFrame) -> pd.DataFrame:
    """TSLA minus S&P500 basic spreads."""
    if {"pct_return_1d", "sp500_pct_return_1d"}.issubset(df.columns):
        df["rel_return_1d"] = df["pct_return_1d"] - df["sp500_pct_return_1d"]
    if {"pct_return_3d", "sp500_pct_return_3d"}.issubset(df.columns):
        df["rel_return_3d"] = df["pct_return_3d"] - df["sp500_pct_return_3d"]
    if {"pct_return_7d", "sp500_pct_return_7d"}.issubset(df.columns):
        df["rel_return_7d"] = df["pct_return_7d"] - df["sp500_pct_return_7d"]
    if {"pct_vol_5d", "sp500_pct_vol_5d"}.issubset(df.columns):
        df["rel_pct_vol_5d"] = df["pct_vol_5d"] - df["sp500_pct_vol_5d"]
    return df


def _prep_social(df_raw: pd.DataFrame, text_col: str) -> pd.DataFrame:
    raw = df_raw[text_col].fillna("").astype(str)

    df_raw["emoji_count"], df_raw["emoji_unique"] = zip(*raw.apply(count_emojis))
    df_raw["emoji_sent_mean_VADER"] = raw.apply(emoji_sent_mean)

    df_raw["clean_text"] = raw.apply(replace_emojis).apply(clean_text)
    df_raw["text_len"] = df_raw["clean_text"].str.len()

    sia = ensure_sia_text()
    df_raw["sentiment_compound_VADER"] = df_raw["clean_text"].apply(
        lambda t: sia.polarity_scores(t)["compound"]
    )
    return df_raw


def process_and_aggregate(path: Path, text_column_guess: Iterable[str]) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    text_col = next((c for c in text_column_guess if c in df.columns), None)
    if text_col is None:
        raise KeyError(f"Could not find text column in {path}")

    df["timestamp"] = drop_tz(df["timestamp"])
    df = _prep_social(df, text_col)
    df["date"] = df["timestamp"].dt.floor("D")

    return (
        df.groupby("date")
        .agg(
            post_count=("timestamp", "size"),
            sentiment_mean_VADER=("sentiment_compound_VADER", "mean"),
            sentiment_std_VADER=("sentiment_compound_VADER", "std"),
            median_len=("text_len", "median"),
            std_len=("text_len", "std"),
            emoji_count=("emoji_count", "sum"),
            emoji_unique=("emoji_unique", "sum"),
            emoji_sent_mean_VADER=("emoji_sent_mean_VADER", "mean"),
            sum_retweets=("retweetCount", "sum"),
            sum_replies=("replyCount", "sum"),
            sum_likes=("likeCount", "sum"),
            sum_quotes=("quoteCount", "sum"),
            sum_views=("viewCount", "sum"),
            sum_bookmarks=("bookmarkCount", "sum"),
            all_posts=("clean_text", lambda t: " <SEP> ".join(t)),
        )
        .reset_index()
    )


def add_tfidf(df: pd.DataFrame) -> pd.DataFrame:
    docs = df["all_posts"].fillna("").astype(str)
    vec = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=(TFIDF_NGRAM_LOW, TFIDF_NGRAM_HIGH),
        stop_words="english",
    )
    mat = vec.fit_transform(docs).toarray()
    vocab = vec.get_feature_names_out()

    wsum = np.zeros(len(docs))
    kw = [""] * len(docs)
    for i, row in enumerate(mat):
        if row.sum() == 0:
            continue
        idx = row.argsort()[-TFIDF_TOP_K:][::-1]
        wsum[i] = row[idx].sum()
        kw[i] = ",".join(vocab[j] for j in idx)

    out = df.copy()
    out["tfidf_top_weight_sum"] = wsum
    out["tfidf_keywords"] = kw
    out = out.sort_values("date").reset_index(drop=True)

    # safe lag
    out["tfidf_weight_sum_lag1"] = out["tfidf_top_weight_sum"].shift(1)
    return out


def zero_fill_social(df: pd.DataFrame) -> pd.DataFrame:
    prefixes = (
        "post_count", "sentiment_mean_VADER", "emoji_count", "emoji_unique",
        "sum_retweets", "sum_replies", "sum_likes", "sum_quotes",
        "sum_views", "sum_bookmarks",
    )
    cols = [c for c in df.columns if c.startswith(prefixes)]
    df[cols] = df[cols].fillna(0)
    return df


def drop_constant_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    const_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    df = df.drop(columns=const_cols)
    return df, const_cols


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────
def main() -> None:
    # 1. TSLA price block
    price_df = load_and_engineer_stock()

    # 1b. S&P500 block
    spx_df = load_and_engineer_sp500()
    price_df = price_df.merge(spx_df, on="date", how="left")

    # 1c. Relative features
    price_df = add_relative_tsla_vs_sp500(price_df)

    # 2. Social data
    tweets = process_and_aggregate(
        PROJ_ROOT / "Data" / "clean" / "clean_musk_tweets.csv",
        text_column_guess=["text"],
    )
    retweets = process_and_aggregate(
        PROJ_ROOT / "Data" / "clean" / "clean_musk_retweets.csv",
        text_column_guess=["tweet", "original_content", "text"],
    )

    daily = tweets.merge(
        retweets,
        on="date",
        how="outer",
        suffixes=("_tweet", "_retweet"),
    ).fillna(0)

    daily["all_posts"] = (
        daily["all_posts_tweet"].astype(str)
        + " <SEP> "
        + daily["all_posts_retweet"].astype(str)
    ).str.strip(" <SEP> ")
    daily.drop(columns=["all_posts_tweet", "all_posts_retweet"], inplace=True)

    # 3. Merge & TF‑IDF
    price_df["date"] = drop_tz(price_df["date"])
    daily["date"]     = drop_tz(daily["date"])

    merged   = price_df.merge(daily, on="date", how="left")
    final_df = add_tfidf(merged)
    final_df = zero_fill_social(final_df)

    # Always drop
    final_df = final_df.drop(columns=[c for c in ALWAYS_DROP if c in final_df.columns],
                             errors="ignore")

    # Flags
    if DROP_MISSING_ROWS:
        final_df = final_df.dropna(subset=["price_ma_5d", "price_ma_10d", "next_day_pct_change"])
    if DROP_NO_SOCIAL_ROWS:
        mask = (final_df["post_count_tweet"] > 0) | (final_df["post_count_retweet"] > 0)
        final_df = final_df[mask].copy()
    if DROP_OHLCV_COLS:
        final_df = final_df.drop(columns=["Open", "High", "Low", "Price", "Volume"], errors="ignore")
    if DROP_SP500_OHLCV_COLS:
        final_df = final_df.drop(
            columns=["sp500_open","sp500_high","sp500_low","sp500_close","sp500_volume"],
            errors="ignore"
        )
    if DROP_ALL_POSTS_TEXT and "all_posts" in final_df.columns:
        final_df = final_df.drop(columns=["all_posts"])
    if DROP_CONSTANT_COLS:
        final_df, dropped_consts = drop_constant_columns(final_df)
    else:
        dropped_consts = []

    # 4. Save
    out_full = PROJ_ROOT / OUT_FULL
    out_full.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(out_full, index=False)

    print(f"✅ Saved dataset → {out_full} ({len(final_df):,} rows × {len(final_df.columns)} cols)")
    print("\n--- QA summary ---")
    print("Top 10 columns by missing values:")
    print(final_df.isna().sum().sort_values(ascending=False).head(10))
    if dropped_consts:
        print(f"\nDropped constant columns: {dropped_consts}")


if __name__ == "__main__":
    main()
