# build_full_model_data.py

import os
import re
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer


def clean_text(text: str) -> str:
    """Remove URLs, mentions, hashtags, normalize whitespace, and drop literal 'nan' tokens."""
    if not text or text.lower() == 'nan':
        return ''
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'\bnan\b', '', text, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', text).strip()


def load_and_engineer_stock(project_root: str) -> pd.DataFrame:
    """Load stock data and compute returns & technical indicators."""
    path = os.path.join(project_root, 'Data', 'clean', 'clean_stock.csv')
    df = pd.read_csv(path, parse_dates=['Date'])
    df['date'] = df['Date'].dt.date
    df['next_day_pct_change'] = df['Close'].pct_change().shift(-1)*1000
    df['close_price_5d_moving_average'] = df['Close'].rolling(5).mean()
    df['close_price_10d_moving_average'] = df['Close'].rolling(10).mean()
    df['return_5d_volatility'] = df['Close'].pct_change().rolling(5).std()
    return df[[
        'date', 'Open', 'High', 'Low', 'Close', 'Volume',
        'next_day_pct_change',
        'close_price_5d_moving_average',
        'close_price_10d_moving_average',
        'return_5d_volatility'
    ]]

def process_and_aggregate_tweets(project_root: str) -> pd.DataFrame:
    """Load tweets, compute sentiment/engagement, aggregate daily, mark tweet boundaries."""
    path = os.path.join(project_root, 'Data', 'processed', 'musk_tweets.csv')
    tweets = pd.read_csv(path, parse_dates=['timestamp'])

    # 1) Turn real NaNs into empty strings
    tweets['text'] = tweets['text'].fillna('')

    # 2) Clean text & compute sentiment
    tweets['clean_text'] = tweets['text'].astype(str).apply(clean_text)
    tweets['text_length'] = tweets['clean_text'].str.len()
    nltk.download('vader_lexicon', quiet=True)
    sia = SentimentIntensityAnalyzer()
    tweets['sentiment'] = tweets['clean_text'].apply(lambda t: sia.polarity_scores(t)['compound'])

    # 3) Engagement counts
    for col in ['retweetCount', 'replyCount', 'likeCount', 'quoteCount', 'viewCount', 'bookmarkCount']:
        if col in tweets.columns:
            tweets[col] = pd.to_numeric(tweets[col], errors='coerce').fillna(0).astype(int)

    # 4) Date & time buckets
    tweets['date'] = tweets['timestamp'].dt.date
    tweets['hour'] = tweets['timestamp'].dt.hour
    tweets['time_bucket'] = tweets['hour'].apply(
        lambda h: 'morning' if h < 12 else ('afternoon' if h < 18 else 'evening')
    )

    # 5) Numeric aggregates
    agg = tweets.groupby('date').agg(
        tweet_count=('clean_text', 'count'),
        tweet_sentiment_mean=('sentiment', 'mean'),
        tweet_sentiment_std=('sentiment', 'std'),
        median_text_length=('text_length', 'median'),
        std_text_length=('text_length', 'std'),
        sum_retweets=('retweetCount', 'sum'),
        sum_replies=('replyCount', 'sum'),
        sum_likes=('likeCount', 'sum'),
        sum_quotes=('quoteCount', 'sum'),
        sum_views=('viewCount', 'sum'),
        sum_bookmarks=('bookmarkCount', 'sum')
    ).reset_index()

    # 6) Time-of-day counts
    tod = tweets.groupby(['date', 'time_bucket']).size().unstack(fill_value=0).reset_index()
    for b in ['morning', 'afternoon', 'evening']:
        if b not in tod.columns:
            tod[b] = 0
    tod = tod.rename(columns={
        'morning': 'count_morning',
        'afternoon': 'count_afternoon',
        'evening': 'count_evening'
    })

    # 7) Concatenate all posts with explicit separator
    daily_text = tweets.groupby('date').agg(
        all_posts=('clean_text', lambda texts: ' <TWEET_SEP> '.join(texts))
    ).reset_index()

    daily = agg.merge(tod, on='date', how='left') \
        .merge(daily_text, on='date', how='left')
    daily['all_posts'] = daily['all_posts'].fillna('').astype(str)
    return daily


def add_tfidf_summary(daily_df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    """
    Compute TF-IDF summary features:
      - daily_top5_tfidf_weight_sum: sum of top_k TF-IDF scores
      - tfidf_top_keywords: comma-separated top_k terms
      - daily_top5_tfidf_weight_sum_lag1: prior day's sum
      - daily_top5_tfidf_weight_sum_lead1: next day's sum

    Empty or placeholder days yield 0.0 and ''.
    """
    # Strip any residual literal 'nan' from docs
    docs = [doc.replace('nan', '').strip() for doc in daily_df['all_posts'].astype(str).tolist()]

    vec = TfidfVectorizer(max_features=500, ngram_range=(1, 2), stop_words='english')
    mat = vec.fit_transform(docs)
    feats = vec.get_feature_names_out()
    arr = mat.toarray()

    sums, keywords = [], []
    for text, row in zip(docs, arr):
        if not text:
            sums.append(0.0)
            keywords.append('')
        else:
            idx = row.argsort()[-top_k:][::-1]
            sums.append(row[idx].sum())
            keywords.append(','.join(feats[i] for i in idx))

    df = daily_df.copy()
    df['daily_top5_tfidf_weight_sum'] = sums
    df['tfidf_top_keywords'] = keywords
    df = df.sort_values('date').reset_index(drop=True)
    df['daily_top5_tfidf_weight_sum_lag1'] = df['daily_top5_tfidf_weight_sum'].shift(1)
    df['daily_top5_tfidf_weight_sum_lead1'] = df['daily_top5_tfidf_weight_sum'].shift(-1)
    return df


def main():
    try:
        script_dir = os.path.dirname(__file__)
    except NameError:
        script_dir = os.getcwd()
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir))

    # Flags
    drop_missing_rows = True  # drop rows missing lookbacks or target
    drop_no_tweets = False  # drop days without tweets

    stock_df = load_and_engineer_stock(project_root)
    daily_tweets = process_and_aggregate_tweets(project_root)
    merged = stock_df.merge(daily_tweets, on='date', how='left')
    final_df = add_tfidf_summary(merged, top_k=5)

    if drop_missing_rows:
        final_df = final_df.dropna(subset=[
            'next_day_pct_change',
            'close_price_10d_moving_average'
        ])
    if drop_no_tweets:
        final_df = final_df[final_df['tweet_count'] > 0].copy()

    out_dir = os.path.join(project_root, 'Data', 'model')
    os.makedirs(out_dir, exist_ok=True)
    final_df.to_csv(os.path.join(out_dir, 'model_data_full.csv'), index=False)
    print("Saved full modeling dataset with TF-IDF summary.")


if __name__ == '__main__':
    main()
