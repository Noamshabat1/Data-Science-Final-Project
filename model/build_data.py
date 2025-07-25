import pandas as pd
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def add_sentiment_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add sentiment analysis features for the text data.
    Returns sentiment scores as new columns.
    """
    print("\nGenerating sentiment features...")
    
    # Initialize VADER sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()
    
    # Get text data
    texts = df["all_posts"].fillna("").astype(str).tolist()
    
    # Calculate sentiment scores for each day's posts
    sentiment_scores = []
    for text in texts:
        scores = analyzer.polarity_scores(text)
        sentiment_scores.append(scores)
    
    # Convert to DataFrame for easier handling
    sentiment_df = pd.DataFrame(sentiment_scores)
    
    # Add sentiment columns to the main dataframe
    df['sentiment_compound'] = sentiment_df['compound']  # Overall sentiment (-1 to +1)
    df['sentiment_positive'] = sentiment_df['pos']       # Positive sentiment (0 to 1)
    df['sentiment_negative'] = sentiment_df['neg']       # Negative sentiment (0 to 1)
    df['sentiment_neutral'] = sentiment_df['neu']        # Neutral sentiment (0 to 1)
    
    print(f"Added 4 sentiment features:")
    print(f"  - Compound sentiment (overall): {df['sentiment_compound'].mean():.3f} ± {df['sentiment_compound'].std():.3f}")
    print(f"  - Positive sentiment: {df['sentiment_positive'].mean():.3f} ± {df['sentiment_positive'].std():.3f}")
    print(f"  - Negative sentiment: {df['sentiment_negative'].mean():.3f} ± {df['sentiment_negative'].std():.3f}")
    print(f"  - Neutral sentiment: {df['sentiment_neutral'].mean():.3f} ± {df['sentiment_neutral'].std():.3f}")
    
    return df


def add_tfidf_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add TF-IDF features for the text data.
    Returns reduced TF-IDF values as new columns.
    """
    print("\nGenerating TF-IDF features...")
    
    # Get text data
    texts = df["all_posts"].fillna("").astype(str).tolist()
    
    # Create TF-IDF vectorizer
    tfidf = TfidfVectorizer(
        max_features=1000,  # Limit vocabulary size
        ngram_range=(1, 2),  # Use unigrams and bigrams
        stop_words='english',  # Remove common English stop words
        min_df=2,  # Ignore terms that appear in less than 2 documents
        max_df=0.95  # Ignore terms that appear in more than 95% of documents
    )
    
    # Fit and transform the texts
    tfidf_matrix = tfidf.fit_transform(texts)
    
    # Use SVD to reduce dimensionality (similar to embeddings)
    n_components = 10  # Use 10 TF-IDF components
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    tfidf_reduced = svd.fit_transform(tfidf_matrix)
    
    # Add TF-IDF dimensions as new columns
    for i in range(n_components):
        df[f'tfidf_{i}'] = tfidf_reduced[:, i]
    
    print(f"Added {n_components} TF-IDF components (explained variance: {svd.explained_variance_ratio_.sum():.3f})")
    return df


def add_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add sentence embeddings for the text data.
    Returns average embedding values as new columns.
    """
    print("\nGenerating text embeddings...")
    # Load the model (first time will download it)
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Get embeddings for each day's posts
    texts = df["all_posts"].fillna("").astype(str).tolist()
    embeddings = model.encode(texts, show_progress_bar=True)

    # Add embedding dimensions as new columns
    n_components = 8  # We'll use top 8 dimensions
    for i in range(n_components):
        df[f'embed_{i}'] = embeddings[:, i]

    print(f"Added {n_components} embedding dimensions")
    return df

def load_and_merge_data():
    """Load and merge all data sources into a single dataset"""
    # Load the CSV files using absolute paths
    replies_df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data/clean/clean_musk_replies.csv'))
    retweets_df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data/clean/clean_musk_retweets.csv'))
    tweets_df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data/clean/clean_musk_tweets.csv'))
    tesla_df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data/clean/clean_tesla_stock.csv'))

    # Common numeric columns to keep
    numeric_columns = [
        'retweetCount',
        'replyCount',
        'likeCount',
        'quoteCount',
        'viewCount',
        'bookmarkCount'
    ]

    # Add source column to each dataframe
    replies_df['source'] = 'replies'
    retweets_df['source'] = 'retweets'
    tweets_df['source'] = 'tweets'

    # Keep only numeric columns and timestamp
    replies_data = replies_df[numeric_columns + ['timestamp', 'source', 'text']]
    retweets_data = retweets_df[numeric_columns + ['timestamp', 'source', 'tweet']]  # using 'tweet' instead of 'text'
    tweets_data = tweets_df[numeric_columns + ['timestamp', 'source', 'text']]

    # Rename 'tweet' to 'text' in retweets data for consistency
    retweets_data = retweets_data.rename(columns={'tweet': 'text'})

    # Concatenate all dataframes
    merged_df = pd.concat([replies_data, retweets_data, tweets_data], ignore_index=True)

    # Replace NaN values with 0 for numeric columns
    merged_df[numeric_columns] = merged_df[numeric_columns].fillna(0)

    # Convert timestamp to datetime
    merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'])

    # Extract date from timestamp for grouping
    merged_df['date'] = merged_df['timestamp'].dt.date

    # Group text by date before summing numeric values
    text_by_date = merged_df.groupby('date').agg(
        all_posts=('text', lambda t: " <SEP> ".join(t.fillna("").astype(str)))
    ).reset_index()

    # Group by date and sum numeric values
    merged_df = merged_df.groupby('date')[numeric_columns].sum().reset_index()

    # Merge back the text data
    merged_df = merged_df.merge(text_by_date, on='date', how='left')

    # Convert date back to datetime for consistency
    merged_df['timestamp'] = pd.to_datetime(merged_df['date'])

    # Prepare Tesla stock data
    tesla_df['Date'] = pd.to_datetime(tesla_df['Date']).dt.date
    tesla_df = tesla_df[['Date', 'Close']]
    tesla_df = tesla_df.rename(columns={'Date': 'date', 'Close': 'tesla_close'})

    # Sort Tesla data by date before forward filling
    tesla_df = tesla_df.sort_values('date')
    # Forward fill missing values in Tesla data
    tesla_df['tesla_close'] = tesla_df['tesla_close'].ffill()

    # Merge with Tesla stock data
    merged_df = pd.merge(merged_df, tesla_df, on='date', how='left')

    # Forward fill any remaining missing values after merge
    merged_df['tesla_close'] = merged_df['tesla_close'].ffill()
    # Backward fill for any remaining NaN values at the beginning
    merged_df['tesla_close'] = merged_df['tesla_close'].bfill()

    # Drop the date column and keep timestamp
    merged_df = merged_df.drop('date', axis=1)

    # Sort by timestamp
    merged_df = merged_df.sort_values('timestamp')

    # Drop timestamp column
    merged_df = merged_df.drop('timestamp', axis=1)

    # Apply sentiment analysis
    merged_df = add_sentiment_features(merged_df)
    
    # Apply TF-IDF features
    merged_df = add_tfidf_features(merged_df)
    
    # Apply embeddings
    merged_df = add_embeddings(merged_df)

    # Drop all_posts column after feature extraction
    merged_df = merged_df.drop('all_posts', axis=1)

    # Create model data directory if it doesn't exist
    model_data_dir = os.path.join(PROJECT_ROOT, 'data', 'model')
    os.makedirs(model_data_dir, exist_ok=True)

    # Save the merged dataframe
    output_path = os.path.join(model_data_dir, 'model_data.csv')
    merged_df.to_csv(output_path, index=False)
    print(f"\nSaved raw merged data to: {output_path}")
    print("📝 Note: Normalization will be applied during modeling to prevent data leakage")

    return merged_df

if __name__ == "__main__":
    merged_data = load_and_merge_data()
    print(f"Total number of days: {len(merged_data)}")
    print("\nSample of merged daily data:")
    print(merged_data.head())
    print("\nNumeric columns statistics (daily sums):")
    print(merged_data.describe())

    # Print number of NaN values in each column to verify
    print("\nNumber of NaN values in each column:")
    print(merged_data.isna().sum())
