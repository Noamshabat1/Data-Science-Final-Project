import warnings
warnings.filterwarnings("ignore")

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

from sentence_transformers import SentenceTransformer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import nltk
try:
    nltk.data.find("vader_lexicon")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = DATA_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

KEYWORD_PATTERNS = {
    "tesla": r"\btesla\b",
    "stock": r"\bstock\b|\btsla\b",
    "car": r"\bcar\b|\bvehicle\b|\bmodel\b",
    "business": r"\bbusiness\b|\bcompany\b|\bmarket\b",
    "people": r"\bpeople\b|\bteam\b|\bemployee\b|\bworker\b",
    "tech": r"\btech\b|\btechnology\b|\bAI\b|\brobot\b|\bsoftware\b",
    "energy": r"\benergy\b|\bbattery\b|\bsolar\b|\bcharging\b",
    "space": r"\bspace\b|\bspacex\b|\brocket\b|\bmars\b",
    "money": r"\bmoney\b|\bdollar\b|\bcost\b|\bprice\b|\bfunding\b|\bprofit\b"
}

TFIDF_CONFIG = {
    "ngram_range": (1, 2),
    "max_features": 6000,
    "min_df": 5,
    "max_df": 0.8,
    "stop_words": "english"
}

def load_all_tweets():
    tweets_path = DATA_DIR / "clean" / "clean_musk_tweets.csv"
    tweets = pd.read_csv(tweets_path)
    tweets["timestamp"] = pd.to_datetime(tweets["timestamp"])
    return tweets

def load_tesla_stock():
    stock_path = DATA_DIR / "clean" / "clean_tesla_stock.csv"
    return pd.read_csv(stock_path)

def aggregate_daily_tweets(tweets_df):
    tweets_df["date"] = tweets_df["timestamp"].dt.date
    
    agg_data = tweets_df.groupby("date").agg({
        "text": lambda x: " ".join(x.dropna().astype(str)),
        "likeCount": ["count", "sum", "mean"],
        "retweetCount": ["sum", "mean"],
        "replyCount": ["sum", "mean"],
        "quoteCount": ["sum", "mean"],
        "viewCount": ["sum", "mean"],
        "bookmarkCount": ["sum", "mean"]
    }).reset_index()
    
    agg_data.columns = ["date", "all_posts"] + [
        "_".join(filter(None, col)).strip() for col in agg_data.columns[2:]
    ]
    
    agg_data.rename(columns={"likeCount_count": "tweet_count"}, inplace=True)
    
    return agg_data

def add_keyword_features(agg_data):
    for kw, pattern in KEYWORD_PATTERNS.items():
        agg_data[f"kw_{kw}"] = agg_data["all_posts"].str.count(pattern, flags=re.IGNORECASE)
    return agg_data

def add_sentiment_features(agg_data):
    print("· Generating sentiment feature")
    
    analyzer = SentimentIntensityAnalyzer()
    
    def compute_sentiment(text):
        if pd.isna(text) or text.strip() == "":
            return 0.0
        return analyzer.polarity_scores(text)["compound"]
    
    agg_data["sentiment_mean"] = agg_data["all_posts"].apply(compute_sentiment)
    return agg_data

def add_tfidf_features(agg_data):
    print("· Generating TF‑IDF features")
    
    texts = agg_data["all_posts"].fillna("").astype(str)
    
    vectorizer = TfidfVectorizer(**TFIDF_CONFIG)
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    pca = PCA(n_components=10, random_state=42)
    tfidf_pca = pca.fit_transform(tfidf_matrix.toarray())
    
    tfidf_cols = [f"tfidf_{i}" for i in range(10)]
    tfidf_df = pd.DataFrame(tfidf_pca, columns=tfidf_cols, index=agg_data.index)
    
    return pd.concat([agg_data, tfidf_df], axis=1)

def add_embedding_features(agg_data):
    print("· Generating MiniLM embeddings")
    
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = agg_data["all_posts"].fillna("").astype(str).tolist()
    
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=False)
    
    pca = PCA(n_components=8, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings)
    
    embed_cols = [f"embed_{i}" for i in range(8)]
    embed_df = pd.DataFrame(embeddings_pca, columns=embed_cols, index=agg_data.index)
    
    return pd.concat([agg_data, embed_df], axis=1)

def add_technical_indicators(stock_df):
    stock_df = stock_df.copy()
    stock_df["Date"] = pd.to_datetime(stock_df["Date"])
    
    stock_df = stock_df.sort_values("Date")
    stock_df["ret_1d"] = stock_df["Close"].pct_change()
    stock_df["ret_5d"] = stock_df["Close"].pct_change(periods=5)
    stock_df["vol_5d"] = stock_df["ret_1d"].rolling(5).std()
    stock_df["rsi_14"] = calculate_rsi(stock_df["Close"], window=14)
    
    return stock_df

def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def create_overview_figure(tweets_agg, stock_clean):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    tweet_dates = pd.to_datetime(tweets_agg["date"])
    stock_dates = pd.to_datetime(stock_clean["Date"])
    
    axes[0, 0].plot(tweet_dates, tweets_agg["tweet_count"], color="steelblue", linewidth=1.5)
    axes[0, 0].set_title("Daily Tweet Volume", fontsize=12, pad=10)
    axes[0, 0].set_ylabel("Number of Tweets")
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(stock_dates, stock_clean["Close"], color="darkgreen", linewidth=1.5)
    axes[0, 1].set_title("Tesla Stock Price", fontsize=12, pad=10)
    axes[0, 1].set_ylabel("Price ($)")
    axes[0, 1].grid(True, alpha=0.3)
    
    sentiment_colors = tweets_agg["sentiment_mean"].apply(lambda x: "green" if x > 0 else "red")
    axes[1, 0].scatter(tweet_dates, tweets_agg["sentiment_mean"], c=sentiment_colors, alpha=0.6, s=30)
    axes[1, 0].axhline(y=0, color="gray", linestyle="--", alpha=0.7)
    axes[1, 0].set_title("Daily Sentiment Score", fontsize=12, pad=10)
    axes[1, 0].set_ylabel("Sentiment Score")
    axes[1, 0].grid(True, alpha=0.3)
    
    stock_clean_subset = stock_clean.dropna(subset=["vol_5d"])
    axes[1, 1].plot(pd.to_datetime(stock_clean_subset["Date"]), stock_clean_subset["vol_5d"], 
                    color="orange", linewidth=1.5)
    axes[1, 1].set_title("5‑Day Volatility", fontsize=12, pad=10)
    axes[1, 1].set_ylabel("Volatility")
    axes[1, 1].grid(True, alpha=0.3)
    
    for ax in axes.flat:
        ax.tick_params(axis="x", rotation=45)
    
    save_to = MODEL_DIR / "data_overview.png"
    fig.suptitle("Figure 1 – data‑processing overview", fontsize=14, y=0.98)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    fig.savefig(save_to, dpi=300)
    plt.close()
    print(f"Saved Figure 1 to {save_to}")

def load_and_merge_data() -> pd.DataFrame:
    tweets = load_all_tweets()
    stock = load_tesla_stock()
    
    print("\n=== BUILDING DATASET ===")
    tweets_agg = aggregate_daily_tweets(tweets)
    tweets_agg = add_keyword_features(tweets_agg)
    tweets_agg = add_sentiment_features(tweets_agg)
    tweets_agg = add_tfidf_features(tweets_agg)
    tweets_agg = add_embedding_features(tweets_agg)
    
    stock_clean = add_technical_indicators(stock)
    
    tweets_agg["date"] = pd.to_datetime(tweets_agg["date"])
    stock_clean["Date"] = pd.to_datetime(stock_clean["Date"])
    
    merged = pd.merge(tweets_agg, stock_clean, left_on="date", right_on="Date", how="inner")
    
    create_overview_figure(tweets_agg, stock_clean)
    
    out_path = MODEL_DIR / "model_data.csv"
    merged.to_csv(out_path, index=False)
    print(f"Saved dataset to {out_path}  shape={merged.shape}\n")
    return merged

if __name__ == "__main__":
    df_final = load_and_merge_data()
    print("Sample of the final dataset:")
    print(df_final.head())
