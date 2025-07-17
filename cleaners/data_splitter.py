import os
import sys
import pandas as pd
import logging


def setup_logging():
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def load_data(input_path: str) -> pd.DataFrame:
    usecols = [
        "fullText",
        "createdAt",
        "isRetweet",
        "isReply",
        "retweetCount",
        "replyCount",
        "likeCount",
        "quoteCount",
        "viewCount",
        "bookmarkCount",
        "quote"
    ]
    df = pd.read_csv(input_path, usecols=usecols, dtype=str)
    logging.info(f"Loaded {len(df):,} rows from {input_path}")
    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "fullText": "text",
        "createdAt": "timestamp"
    }
    df = df.rename(columns=rename_map, errors="raise")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["text_length"] = df["text"].str.len().fillna(0).astype(int)
    return df


def main_split_data(df: pd.DataFrame):
    # Create boolean masks for comprehensive classification
    retweet_mask = pd.Series(False, index=df.index, dtype=bool)
    reply_mask = pd.Series(False, index=df.index, dtype=bool)
    
    # Retweet detection: boolean flag OR text pattern
    retweet_mask = retweet_mask | (df["isRetweet"] == "True")
    rt_pattern = r'^RT\s+@(\w+):\s*(.*)'
    retweet_mask = retweet_mask | df["text"].str.contains(rt_pattern, regex=True, na=False)
    
    # Reply detection: boolean flag OR text pattern (excluding retweets)
    reply_mask = reply_mask | (df["isReply"] == "True")
    at_pattern = r'^@\w+'
    reply_mask = reply_mask | df["text"].str.contains(at_pattern, regex=True, na=False)
    
    # Apply masks to split data
    retweets = df[retweet_mask].copy()
    replies = df[reply_mask & (~retweet_mask)].copy()  # Exclude retweets from replies
    originals = df[~retweet_mask & ~reply_mask].copy()

    total = len(df)
    categorized = len(retweets) + len(replies) + len(originals)
    logging.info(f"Split complete: {categorized}/{total} posts categorized")
    
    return retweets, replies, originals


def drop_flags(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["isRetweet", "isReply"], errors="ignore")


def save_df(df: pd.DataFrame, path: str):
    try:
        df.to_csv(path, index=False)
        logging.info(f"Saved {len(df):,} rows to {path}")
    except Exception as e:
        logging.error(f"Failed to save {path}: {e}")


def preview(df: pd.DataFrame, name: str):
    n = min(3, len(df))
    if n == 0:
        logging.info(f"No rows to preview for {name}")
        return
    sample = df.sample(n, random_state=0)
    logging.info(f"--- Preview of {name} ({n} rows) ---")
    for _, row in sample.iterrows():
        ts = row["timestamp"]
        text = row["text"].replace("\n", " ")
        logging.info(f"{ts.date()} → {text[:80]}...")


def main_data_splitter():
    setup_logging()

    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
    input_path = os.path.join(project_root, "Data", "original", "all_musk_posts.csv")
    output_dir = os.path.join(project_root, "Data", "splitted")
    os.makedirs(output_dir, exist_ok=True)

    df = load_data(input_path)
    df = standardize_columns(df)
    retweets, replies, originals = main_split_data(df)

    for subset, fname in [
        (retweets, "musk_retweets.csv"),
        (replies, "musk_replies.csv"),
        (originals, "musk_tweets.csv")
    ]:
        clean = drop_flags(subset)
        save_df(clean, os.path.join(output_dir, fname))

    preview(retweets, "Retweets")
    preview(replies, "Replies")
    preview(originals, "Original Tweets")

    total = len(df)
    logging.info(
        "Split summary: total=%d, retweets=%d (%.1f%%), replies=%d (%.1f%%), originals=%d (%.1f%%)",
        total,
        len(retweets), len(retweets) / total * 100,
        len(replies), len(replies) / total * 100,
        len(originals), len(originals) / total * 100,
    )


if __name__ == "__main__":
    main_data_splitter()