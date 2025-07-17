# Noam vertion.
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
        "id",
        "twitterUrl",
        "fullText",
        "createdAt",
        "isRetweet",
        "isReply",
        "retweetCount",
        "replyCount",
        "likeCount",
        "quoteCount",
        "viewCount",
        "bookmarkCount"
    ]
    df = pd.read_csv(input_path, usecols=usecols, dtype=str)
    logging.info(f"Loaded {len(df):,} rows from {input_path}")
    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "id": "tweet_id",
        "twitterUrl": "tweet_url",
        "fullText": "text",
        "createdAt": "timestamp"
    }
    df = df.rename(columns=rename_map, errors="raise")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["text_length"] = df["text"].str.len().fillna(0).astype(int)
    return df


def split_data(df: pd.DataFrame):
    retweets = df[df["isRetweet"] == "True"].copy()
    replies = df[df["isReply"] == "True"].copy()
    originals = df[(df["isRetweet"] != "True") & (df["isReply"] != "True")].copy()

    total = len(df)
    assert len(retweets) + len(replies) + len(originals) == total, (
        f"Row count mismatch: {len(retweets) + len(replies) + len(originals)} != {total}"
    )
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


def main():
    setup_logging()

    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
    input_path = os.path.join(project_root, "Data", "original", "all_musk_posts.csv")
    output_dir = os.path.join(project_root, "Data", "processed")
    os.makedirs(output_dir, exist_ok=True)

    df = load_data(input_path)
    df = standardize_columns(df)
    retweets, replies, originals = split_data(df)

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
    main()