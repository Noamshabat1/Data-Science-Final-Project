import pandas as pd
import os


def clean_tweets_data():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
    input_file = os.path.join(project_root, "data", "splitted", "musk_tweets.csv")
    output_file = os.path.join(project_root, "data", "clean", "clean_musk_tweets.csv")

    print("=" * 60)
    print("TWEETS DATA CLEANER")
    print("=" * 60)

    try:
        df = pd.read_csv(input_file)
        print(f"✓ Loaded {len(df):,} original tweets from {input_file}")
        print(f"  Columns: {len(df.columns)}")
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    print(f"\nInitial data shape: {df.shape}")

    print(f"\ndata Overview:")
    print(f"- Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"- Missing text: {df['text'].isna().sum()}")

    print(f"\n" + "-" * 40)
    print("CLEANING PROCESS")
    print("-" * 40)

    initial_count = len(df)
    df_clean = df.dropna(subset=['text']).copy()
    removed_empty = initial_count - len(df_clean)
    if removed_empty > 0:
        print(f"✓ Removed {removed_empty} tweets with missing text")

    def clean_tweet_text(text):
        if pd.isna(text):
            return ""
        text = str(text)
        text = ' '.join(text.split())
        text = text.replace('\u200b', '').replace('\u200c', '').replace('\u200d', '')
        return text.strip()

    print("✓ Cleaning tweet text...")
    df_clean['text'] = df_clean['text'].apply(clean_tweet_text)

    before_alpha_filter = len(df_clean)
    df_clean = df_clean[df_clean['text'].str.contains(r'[a-zA-Z0-9]', regex=True, na=False)]
    removed_non_alpha = before_alpha_filter - len(df_clean)
    if removed_non_alpha > 0:
        print(f"✓ Removed {removed_non_alpha} tweets with only non-alphanumeric characters")

    before_dedup = len(df_clean)
    df_clean = df_clean.drop_duplicates(keep='first')
    removed_duplicates = before_dedup - len(df_clean)
    if removed_duplicates > 0:
        print(f"✓ Removed {removed_duplicates} duplicate tweets")

    df_clean = df_clean.sort_values(by='timestamp')

    print(f"\n" + "-" * 40)
    print("CLEANING SUMMARY")
    print("-" * 40)
    print(f"Original tweets: {initial_count:,}")
    print(f"Final tweets: {len(df_clean):,}")
    print(f"Removed total: {initial_count - len(df_clean):,}")
    print(f"data retention: {len(df_clean) / initial_count * 100:.1f}%")

    try:
        df_clean.to_csv(output_file, index=False)
        print(f"\n✅ Saved cleaned tweets to: {output_file}")
        print(f"   Rows: {len(df_clean):,}")
        print(f"   Columns: {len(df_clean.columns)}")


    except Exception as e:
        print(f"Error saving file: {e}")
        return None

    return {
        'initial_count': initial_count,
        'removed_empty': removed_empty,
        'removed_non_alpha': removed_non_alpha,
        'removed_duplicates': removed_duplicates,
        'final_count': len(df_clean),
        'retention_rate': len(df_clean) / initial_count * 100
    }


if __name__ == "__main__":
    clean_tweets_data()
