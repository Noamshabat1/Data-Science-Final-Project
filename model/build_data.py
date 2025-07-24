import pandas as pd
import os
import numpy as np
from sentence_transformers import SentenceTransformer

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

    # Apply embeddings
    merged_df = add_embeddings(merged_df)

    # Drop all_posts column after embedding
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
