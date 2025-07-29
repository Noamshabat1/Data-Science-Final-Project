import pandas as pd
import re
import os

def clean_tweets_data():
    input_file = 'data/splitted/musk_tweets.csv'
    output_file = 'data/clean/clean_musk_tweets.csv'
    
    print("=" * 60)
    print("TWEETS DATA CLEANER")
    print("=" * 60)
    
    try:
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df):,} original tweets from {input_file}")
        print(f"  Columns: {len(df.columns)}")
    except Exception as e:
        print(f"Error loading file: {e}")
        return None
    
    print(f"\nInitial data shape: {df.shape}")
    
    if 'text' not in df.columns:
        print(f"\nError: 'text' column not found in {input_file}")
        return None
    
    print(f"\ndata Overview:")
    print(f"- Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"- Missing text: {df['text'].isna().sum()}")
    
    initial_count = len(df)
    print(f"\n" + "-" * 40)
    print("CLEANING PROCESS")
    print("-" * 40)
    
    df_clean = df.copy()
    
    df_clean = df_clean.dropna(subset=['text'])
    removed_empty = initial_count - len(df_clean)
    if removed_empty > 0:
        print(f"Removed {removed_empty} tweets with missing text")
    
    def clean_tweet_text(text):
        if pd.isna(text):
            return ""
        
        text = str(text)
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'\t', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    print("Cleaning tweet text...")
    df_clean['text'] = df_clean['text'].apply(clean_tweet_text)
    
    df_clean = df_clean[df_clean['text'].str.contains(r'[a-zA-Z]', regex=True, na=False)]
    before_alpha_filter = len(df_clean)
    df_clean = df_clean[df_clean['text'].str.len() > 0]
    removed_non_alpha = before_alpha_filter - len(df_clean)
    if removed_non_alpha > 0:
        print(f"Removed {removed_non_alpha} tweets with only non-alphanumeric characters")
    
    before_dedup = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=['text'], keep='first')
    removed_duplicates = before_dedup - len(df_clean)
    if removed_duplicates > 0:
        print(f"Removed {removed_duplicates} duplicate tweets")
    
    df_clean = df_clean.sort_values(by='timestamp')
    
    print(f"\n" + "-" * 40)
    print("CLEANING SUMMARY")
    print("-" * 40)
    print(f"Original tweets: {initial_count:,}")
    print(f"Final tweets: {len(df_clean):,}")
    print(f"Removed total: {initial_count - len(df_clean):,}")
    print(f"data retention: {len(df_clean) / initial_count * 100:.1f}%")
    
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df_clean.to_csv(output_file, index=False)
        print(f"\nSUCCESS Saved cleaned tweets to: {output_file}")
        print(f"   Rows: {len(df_clean):,}")
        print(f"   Columns: {len(df_clean.columns)}")
        
        return df_clean
    except Exception as e:
        print(f"Error saving file: {e}")
        return None

if __name__ == "__main__":
    clean_tweets_data()
