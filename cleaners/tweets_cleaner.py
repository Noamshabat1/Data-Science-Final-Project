import pandas as pd
import os


def clean_tweets_data():
    """
    Cleans and processes the original tweets data with tweet-specific operations:
    - Cleans tweet text content
    - Handles hashtags, mentions, and URLs
    - Removes duplicates and invalid tweets
    - Adds text analytics
    """

    # Define file paths
    input_file = os.path.join("..", "Data", "splitted", "musk_tweets.csv")
    output_file = os.path.join("..", "Data", "clean", "clean_musk_tweets.csv")

    print("=" * 60)
    print("TWEETS DATA CLEANER")
    print("=" * 60)

    # Load the tweets data
    try:
        df = pd.read_csv(input_file)
        print(f"✓ Loaded {len(df):,} original tweets from {input_file}")
        print(f"  Columns: {len(df.columns)}")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return

    print(f"\nInitial data shape: {df.shape}")

    # Display basic info
    print(f"\nData Overview:")
    print(f"- Date range: {df['createdAt'].min()} to {df['createdAt'].max()}")
    print(f"- Missing fullText: {df['fullText'].isna().sum()}")

    # Start cleaning process
    print(f"\n" + "-" * 40)
    print("CLEANING PROCESS")
    print("-" * 40)

    # 1. Remove tweets with missing text
    initial_count = len(df)
    df_clean = df.dropna(subset=['fullText']).copy()
    removed_empty = initial_count - len(df_clean)
    if removed_empty > 0:
        print(f"✓ Removed {removed_empty} tweets with missing text")

    # 2. Clean tweet text content
    def clean_tweet_text(text):
        """Comprehensive tweet text cleaning"""
        if pd.isna(text):
            return ""

        text = str(text)

        # Remove extra whitespace and normalize
        text = ' '.join(text.split())

        # Remove zero-width characters
        text = text.replace('\u200b', '').replace('\u200c', '').replace('\u200d', '')

        # Remove or replace common Twitter artifacts
        # Keep hashtags and mentions as they're meaningful

        return text.strip()

    print("✓ Cleaning tweet text...")
    df_clean['fullText'] = df_clean['fullText'].apply(clean_tweet_text)

    # 3. Remove exact duplicates
    before_dedup = len(df_clean)
    df_clean = df_clean.drop_duplicates(keep='first')
    removed_duplicates = before_dedup - len(df_clean)
    if removed_duplicates > 0:
        print(f"✓ Removed {removed_duplicates} duplicate tweets")

    df_clean = df_clean.sort_values(by='createdAt')

    # Final statistics
    print(f"\n" + "-" * 40)
    print("CLEANING SUMMARY")
    print("-" * 40)
    print(f"Original tweets: {initial_count:,}")
    print(f"Final tweets: {len(df_clean):,}")
    print(f"Removed total: {initial_count - len(df_clean):,}")
    print(f"Data retention: {len(df_clean) / initial_count * 100:.1f}%")

    # 10. Save cleaned data
    try:
        df_clean.to_csv(output_file, index=False)
        print(f"\n✅ Saved cleaned tweets to: {output_file}")
        print(f"   Rows: {len(df_clean):,}")
        print(f"   Columns: {len(df_clean.columns)}")


    except Exception as e:
        print(f"❌ Error saving file: {e}")


if __name__ == "__main__":
    clean_tweets_data()
