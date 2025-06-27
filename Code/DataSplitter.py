import pandas as pd
import os

def preprocess_musk_posts():
    """
    Preprocesses Elon Musk's posts from CSV and splits them into three categories:
    1. Original Tweets (not replies, not retweets)
    2. Replies (isReply = True)
    3. Retweets (isRetweet = True)
    """
    
    # Define file paths
    input_file = os.path.join("..", "Data", "all_musk_posts.csv")
    output_dir = os.path.join("..", "Data", "split_posts")
    
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
    
    # Fill NaN values with False for boolean columns
    df['isReply'] = df['isReply'].fillna(False)
    df['isRetweet'] = df['isRetweet'].fillna(False)
    
    # Split the data into three categories
    print("\nSplitting data into categories...")
    
    # 1. Retweets (isRetweet = True)
    retweets = df[df['isRetweet'] == True].copy()
    print(f"Retweets: {len(retweets)}")
    
    # 2. Replies (isReply = True, but not retweets)
    replies = df[(df['isReply'] == True) & (df['isRetweet'] == False)].copy()
    print(f"Replies: {len(replies)}")
    
    # 3. Original Tweets (neither replies nor retweets)
    original_tweets = df[(df['isReply'] == False) & (df['isRetweet'] == False)].copy()
    print(f"Original Tweets: {len(original_tweets)}")
    
    # Verify totals
    total_categorized = len(retweets) + len(replies) + len(original_tweets)
    print(f"Total categorized: {total_categorized}")
    print(f"Original total: {len(df)}")
    
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
        print("\n" + "="*50)
        print("SUMMARY")
        print("="*50)
        print(f"Original dataset: {len(df):,} posts")
        print(f"├── Retweets: {len(retweets):,} ({len(retweets)/len(df)*100:.1f}%)")
        print(f"├── Replies: {len(replies):,} ({len(replies)/len(df)*100:.1f}%)")
        print(f"└── Original Tweets: {len(original_tweets):,} ({len(original_tweets)/len(df)*100:.1f}%)")
        
    except Exception as e:
        print(f"Error saving files: {e}")

def display_sample_posts():
    """
    Display sample posts from each category for verification
    """
    output_dir = os.path.join("..", "Data", "split_posts")
    
    files = {
        "Retweets": os.path.join(output_dir, "musk_retweets.csv"),
        "Replies": os.path.join(output_dir, "musk_replies.csv"), 
        "Original Tweets": os.path.join(output_dir, "musk_tweets.csv")
    }
    
    print("\n" + "="*60)
    print("SAMPLE POSTS FROM EACH CATEGORY")
    print("="*60)
    
    for category, file_path in files.items():
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                print(f"\n{category.upper()} - Sample (showing first 2):")
                print("-" * 40)
                
                for i in range(min(2, len(df))):
                    post = df.iloc[i]
                    text = str(post['fullText'])[:100] + "..." if len(str(post['fullText'])) > 100 else str(post['fullText'])
                    print(f"{i+1}. {text}")
                    print(f"   Created: {post['createdAt']}")
                    print(f"   isReply: {post['isReply']}, isRetweet: {post['isRetweet']}")
                    print()
                    
            except Exception as e:
                print(f"Error reading {category}: {e}")

if __name__ == "__main__":
    # Run the preprocessing
    preprocess_musk_posts()
    
    # Display sample posts for verification
    display_sample_posts()
