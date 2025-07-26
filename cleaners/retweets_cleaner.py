import pandas as pd
import os
import re
from datetime import datetime


def clean_retweets_data():
    """
    Cleans and processes the retweets data with retweet-specific operations:
    - Extracts original tweet content from RT format
    - Cleans retweet text
    - Handles retweet-specific metadata
    - Removes invalid/empty retweets
    """

    # Define file paths
    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
    input_file = os.path.join(project_root, "data", "splitted", "musk_retweets.csv")
    output_file = os.path.join(project_root, "data", "clean", "clean_musk_retweets.csv")

    print("=" * 60)
    print("RETWEETS DATA CLEANER")
    print("=" * 60)

    # Load the retweets data
    try:
        df = pd.read_csv(input_file)
        print(f"✓ Loaded {len(df):,} retweets from {input_file}")
        print(f"  Columns: {len(df.columns)}")
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    print(f"\nInitial data shape: {df.shape}")

    # Display basic info
    print(f"\ndata Overview:")
    print(f"- Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"- Missing text: {df['text'].isna().sum()}")

    # Start cleaning process
    print(f"\n" + "-" * 40)
    print("CLEANING PROCESS")
    print("-" * 40)

    # 1. Remove retweets with missing text
    initial_count = len(df)
    df_clean = df.dropna(subset=['text']).copy()
    removed_empty = initial_count - len(df_clean)
    if removed_empty > 0:
        print(f"✓ Removed {removed_empty} retweets with missing text")

    # 2. Clean and extract retweet content
    def extract_retweet_content(text):
        """Extract the original content from RT format"""
        if pd.isna(text):
            return None, None

        text_str = str(text).strip()

        # Pattern to match RT @username: content
        rt_pattern = r'^RT\s+@(\w+):\s*(.*)'
        match = re.match(rt_pattern, text_str, re.DOTALL)

        if match:
            original_author = match.group(1)
            original_content = match.group(2).strip()
            return original_author, original_content
        else:
            # If not standard RT format, mark for removal
            return None, None

    print("✓ Extracting retweet content...")

    # Apply content extraction
    extraction_results = df_clean['text'].apply(extract_retweet_content)
    df_clean['originalAuthor'] = [x[0] for x in extraction_results]
    df_clean['tweet'] = [x[1] for x in extraction_results]

    # 3. Remove retweets that couldn't be properly parsed (non-RT format)
    before_parsing = len(df_clean)
    df_clean = df_clean.dropna(subset=['originalAuthor', 'tweet'])
    removed_unparseable = before_parsing - len(df_clean)
    if removed_unparseable > 0:
        print(f"✓ Removed {removed_unparseable} non-RT format entries")

    # 3.1. Remove text column after extracting originalAuthor and originalContent
    if 'text' in df_clean.columns:
        df_clean = df_clean.drop(columns=['text'])
        print("✓ Removed text column")

    # 4. Remove duplicates based on original content
    before_dedup = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=['tweet'], keep='first')
    removed_duplicates = before_dedup - len(df_clean)
    if removed_duplicates > 0:
        print(f"✓ Removed {removed_duplicates} duplicate retweets")

    # 5. Clean timestamps
    print("✓ Processing timestamps...")
    df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'])

    # 6. Clean and standardize original content text
    def clean_text_content(text):
        """Basic text cleaning for retweet content"""
        if pd.isna(text):
            return ""

        text = str(text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove zero-width characters
        text = text.replace('\u200b', '').replace('\u200c', '').replace('\u200d', '')
        return text

    print("✓ Cleaning original content text...")
    df_clean['tweet'] = df_clean['tweet'].apply(clean_text_content)
    
    # Remove rows that have only non-alphanumeric characters
    before_alpha_filter = len(df_clean)
    df_clean = df_clean[df_clean['tweet'].str.contains(r'[a-zA-Z0-9]', regex=True, na=False)]
    removed_non_alpha = before_alpha_filter - len(df_clean)
    if removed_non_alpha > 0:
        print(f"✓ Removed {removed_non_alpha} tweets with only non-alphanumeric characters")

    # 7. Add retweet-specific analytics
    print("✓ Adding retweet analytics...")

    # Count retweets by original author
    author_counts = df_clean['originalAuthor'].value_counts()
    df_clean['authorRetweetCount'] = df_clean['originalAuthor'].map(author_counts.get)

    # Text length statistics
    df_clean['originalTextLength'] = df_clean['tweet'].str.len()

    # 8. Remove very short or invalid retweets
    before_length_filter = len(df_clean)
    df_clean = df_clean[df_clean['originalTextLength'] >= 5]  # At least 5 characters
    removed_short = before_length_filter - len(df_clean)
    if removed_short > 0:
        print(f"✓ Removed {removed_short} very short retweets")

    # 9. Sort by creation date
    df_clean = df_clean.sort_values(by='timestamp')

    # Final statistics
    print(f"\n" + "-" * 40)
    print("CLEANING SUMMARY")
    print("-" * 40)
    print(f"Original retweets: {initial_count:,}")
    print(f"Final retweets: {len(df_clean):,}")
    print(f"Removed total: {initial_count - len(df_clean):,}")
    print(f"data retention: {len(df_clean) / initial_count * 100:.1f}%")

    # 10. Save cleaned data
    try:
        df_clean.to_csv(output_file, index=False)
        print(f"\n✅ Saved cleaned retweets to: {output_file}")
        print(f"   Rows: {len(df_clean):,}")
        print(f"   Columns: {len(df_clean.columns)}")

        # Show sample of cleaned data
        print(f"\nSample cleaned retweets:")
        print("-" * 50)
        for i in range(min(3, len(df_clean))):
            row = df_clean.iloc[i]
            print(f"{i + 1}. Author: @{row['originalAuthor']}")
            print(f"   Content: {row['tweet'][:100]}...")
            print(f"   Date: {row['timestamp']}")
            print(f"   Length: {row['originalTextLength']} chars")
            print()

    except Exception as e:
        print(f"❌ Error saving file: {e}")
        return None
    
    return {
        'initial_count': initial_count,
        'removed_empty': removed_empty,
        'removed_unparseable': removed_unparseable,
        'removed_duplicates': removed_duplicates,
        'removed_non_alpha': removed_non_alpha,
        'removed_short': removed_short,
        'final_count': len(df_clean),
        'retention_rate': len(df_clean) / initial_count * 100
    }


if __name__ == "__main__":
    clean_retweets_data()
