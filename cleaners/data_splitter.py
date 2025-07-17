import pandas as pd
import os


def remove_classification_columns(df, category_name):
    """
    Removes isReply and isRetweet columns from the dataframe.

    Args:
        df: pandas DataFrame to clean
        category_name: string describing the category for logging

    Returns:
        pandas DataFrame with classification columns removed
    """
    columns_to_remove = []

    # Check and collect columns to remove
    if 'isReply' in df.columns:
        columns_to_remove.append('isReply')
    if 'isRetweet' in df.columns:
        columns_to_remove.append('isRetweet')

    # Remove the columns if any exist
    if columns_to_remove:
        df_cleaned = df.drop(columns=columns_to_remove)
        print(f"   Removed classification columns from {category_name}: {columns_to_remove}")
        return df_cleaned
    else:
        print(f"   No classification columns found in {category_name}")
        return df


columns_to_remove = ['twitterUrl', 'isConversationControlled', 'possiblySensitive', 'isPinned', ]


def preprocess_musk_posts():
    """
    Preprocesses Elon Musk's posts from CSV and splits them into three categories:
    1. Original Tweets (not replies, not retweets)
    2. Replies (isReply = True)
    3. Retweets (isRetweet = True OR fullText starts with "RT @")
    """

    # Define file paths
    input_file = os.path.join("..", "Data", "original", "all_musk_posts.csv")
    output_dir = os.path.join("..", "Data", "splitted")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print("Loading data from all_musk_posts.csv...")

    # Load the CSV file with proper handling of mixed types
    try:
        df = pd.read_csv(input_file, low_memory=False)
        print(f"Successfully loaded {len(df)} posts")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return

    # Display basic info about the dataset
    print(f"\nDataset Info:")
    print(f"Total posts: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    # Check for missing values in key columns
    print(f"\nMissing values in key columns:")
    print(f"isReply: {df['isReply'].isna().sum()}")
    print(f"isRetweet: {df['isRetweet'].isna().sum()}")
    if 'fullText' in df.columns:
        print(f"fullText: {df['fullText'].isna().sum()}")

    for column in columns_to_remove:
        if column in df.columns:
            df = df.drop(columns=[column])
        else:
            print(f"\n⚠️  '{column}' column not found, skipping removal")

    # Do not fill NaN values - keep them as NaN for proper analysis

    # Filter rows where URL starts with 'https://x.com/elonmusk'
    print("\nFiltering posts by URL...")
    initial_count = len(df)

    # Check if 'url' column exists
    if 'url' in df.columns:
        # Remove rows where URL doesn't start with 'https://x.com/elonmusk'
        df = df[df['url'].astype(str).str.startswith('https://x.com/elonmusk', na=False)]
        filtered_count = len(df)
        removed_count = initial_count - filtered_count

    else:
        print("⚠️  Warning: 'url' column not found, skipping URL filter")

    # Split the data into three categories
    print("\nSplitting data into categories...")

    # Check for retweets using multiple conditions:
    # 1. isRetweet = True, OR
    # 2. fullText starts with "RT @" pattern
    retweet_mask = pd.Series(False, index=df.index, dtype=bool)

    # Condition 1: isRetweet column is True
    if 'isRetweet' in df.columns:
        retweet_mask = retweet_mask | (df['isRetweet'] == True)

    # Condition 2: fullText starts with "RT @"
    if 'fullText' in df.columns:
        rt_pattern = r'^RT\s+@(\w+):\s*(.*)'
        text_series = df['fullText']
        text_mask = text_series.str.contains(rt_pattern, regex=True, na=False)  # type: ignore
        retweet_mask = retweet_mask | text_mask

    retweets = df[retweet_mask].copy()

    print(f"Retweets: {len(retweets)}")

    # 2. Replies (isReply = True OR starts with "@", but not retweets)
    reply_mask = pd.Series(False, index=df.index, dtype=bool)

    # Condition 1: isReply column is True
    if 'isReply' in df.columns:
        reply_mask = reply_mask | (df['isReply'] == True)

    # Condition 2: fullText starts with "@"
    if 'fullText' in df.columns:
        at_pattern = r'^@\w+'
        at_series = df['fullText']
        at_mask = at_series.str.contains(at_pattern, regex=True, na=False)  # type: ignore
        reply_mask = reply_mask | at_mask

    # Exclude retweets from replies
    replies = df[reply_mask & (~retweet_mask)].copy()
    print(f"Replies: {len(replies)}")

    # 3. Original Tweets (neither replies nor retweets)
    if 'isReply' in df.columns:
        original_tweets = df[(df['isReply'] != True) & ~retweet_mask].copy()
    else:
        original_tweets = df[~retweet_mask].copy()
    print(f"Original Tweets: {len(original_tweets)}")

    # Verify totals
    total_categorized = len(retweets) + len(replies) + len(original_tweets)
    print(f"Total categorized: {total_categorized}")
    print(f"Original total: {len(df)}")

    # Remove classification columns from all categories
    print("\nRemoving classification columns from split data...")
    retweets = remove_classification_columns(retweets, "retweets")
    replies = remove_classification_columns(replies, "replies")
    original_tweets = remove_classification_columns(original_tweets, "original tweets")

    # Save each category to separate CSV files
    print("\nSaving split data to CSV files...")

    try:
        # Function to clean and save dataframe
        def clean_and_save(df, file_path, category_name):
            # Get original column count
            original_cols = len(df.columns)

            # Find columns with all NaN values
            all_nan_cols = df.columns[df.isna().all()].tolist()

            # Drop columns with all NaN values
            if all_nan_cols:
                df_cleaned = df.drop(columns=all_nan_cols)
                print(f"   Removed {len(all_nan_cols)} columns with all NaN values: {all_nan_cols}")
            else:
                df_cleaned = df
                print(f"   No columns with all NaN values found")

            # Save cleaned dataframe
            df_cleaned.to_csv(file_path, index=False)

            final_cols = len(df_cleaned.columns)
            print(f"✓ Saved {len(df)} {category_name} to {file_path}")
            print(f"   Columns: {original_cols} → {final_cols}")

            return df_cleaned

        # Save retweets
        retweets_file = os.path.join(output_dir, "musk_retweets.csv")
        print(f"\nProcessing Retweets:")
        retweets_cleaned = clean_and_save(retweets, retweets_file, "retweets")

        # Save replies
        replies_file = os.path.join(output_dir, "musk_replies.csv")
        print(f"\nProcessing Replies:")
        replies_cleaned = clean_and_save(replies, replies_file, "replies")

        # Save original tweets
        tweets_file = os.path.join(output_dir, "musk_tweets.csv")
        print(f"\nProcessing Original Tweets:")
        tweets_cleaned = clean_and_save(original_tweets, tweets_file, "original tweets")

        print(f"\nAll files saved successfully in {output_dir}")

        # Display summary statistics
        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)
        print(f"Filtered dataset: {len(df):,} posts")
        print(f"├── Retweets: {len(retweets):,} ({len(retweets) / len(df) * 100:.1f}%)")
        print(f"├── Replies: {len(replies):,} ({len(replies) / len(df) * 100:.1f}%)")
        print(f"└── Original Tweets: {len(original_tweets):,} ({len(original_tweets) / len(df) * 100:.1f}%)")

    except Exception as e:
        print(f"Error saving files: {e}")


if __name__ == "__main__":
    preprocess_musk_posts()