import pandas as pd
import re
import os
from datetime import datetime

def clean_retweets_data():
    input_file = 'data/splitted/musk_retweets.csv'
    output_file = 'data/clean/clean_musk_retweets.csv'
    
    print("=" * 60)
    print("RETWEETS DATA CLEANER")
    print("=" * 60)
    
    try:
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df):,} retweets from {input_file}")
        print(f"  Columns: {len(df.columns)}")
    except Exception as e:
        print(f"Error loading file: {e}")
        return None
    
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
        print(f"Removed {removed_empty} retweets with missing text")
    
    def extract_retweet_content(text):
        if pd.isna(text) or not isinstance(text, str):
            return None, None
        
        rt_pattern = r'^RT @([^:]+):\s*(.*)'
        match = re.match(rt_pattern, text.strip())
        
        if match:
            author = match.group(1).strip()
            content = match.group(2).strip()
            return author, content
        else:
            return None, None
    
    print("Extracting retweet content...")
    
    extracted_data = df_clean['text'].apply(extract_retweet_content)
    df_clean['originalAuthor'] = extracted_data.apply(lambda x: x[0] if x[0] is not None else None)
    df_clean['originalContent'] = extracted_data.apply(lambda x: x[1] if x[1] is not None else None)
    
    before_parsing = len(df_clean)
    df_clean = df_clean.dropna(subset=['originalAuthor', 'originalContent'])
    removed_unparseable = before_parsing - len(df_clean)
    if removed_unparseable > 0:
        print(f"Removed {removed_unparseable} non-RT format entries")
    
    if 'text' in df_clean.columns:
        df_clean = df_clean.drop(columns=['text'])
        print("Removed text column")
    
    before_dedup = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=['originalAuthor', 'originalContent'], keep='first')
    removed_duplicates = before_dedup - len(df_clean)
    if removed_duplicates > 0:
        print(f"Removed {removed_duplicates} duplicate retweets")
    
    print("Processing timestamps...")
    df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'])
    
    df_clean = df_clean.rename(columns={'originalContent': 'tweet'})
    
    numeric_columns = ['retweetCount', 'replyCount', 'likeCount', 'quoteCount', 'viewCount', 'bookmarkCount']
    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    def clean_text_content(text):
        if pd.isna(text):
            return ""
        
        text = str(text)
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text
    
    print("Cleaning original content text...")
    df_clean['tweet'] = df_clean['tweet'].apply(clean_text_content)
    
    before_alpha_filter = len(df_clean)
    df_clean = df_clean[df_clean['tweet'].str.contains(r'[a-zA-Z]', regex=True, na=False)]
    removed_non_alpha = before_alpha_filter - len(df_clean)
    if removed_non_alpha > 0:
        print(f"Removed {removed_non_alpha} tweets with only non-alphanumeric characters")
    
    print("Adding retweet analytics...")
    
    author_counts = df_clean['originalAuthor'].value_counts()
    df_clean['authorRetweetCount'] = df_clean['originalAuthor'].map(author_counts)
    
    df_clean['originalTextLength'] = df_clean['tweet'].str.len()
    
    df_clean['retweetRatio'] = df_clean['retweetCount'] / (df_clean['likeCount'] + 1)
    
    df_clean['engagementScore'] = (
        df_clean['likeCount'].fillna(0) + 
        df_clean['retweetCount'].fillna(0) * 2 + 
        df_clean['replyCount'].fillna(0) * 1.5
    )
    
    before_length_filter = len(df_clean)
    df_clean = df_clean[df_clean['originalTextLength'] >= 10]
    removed_short = before_length_filter - len(df_clean)
    if removed_short > 0:
        print(f"Removed {removed_short} very short retweets")
    
    df_clean = df_clean.sort_values('timestamp')
    
    print(f"\n" + "-" * 40)
    print("CLEANING SUMMARY")
    print("-" * 40)
    print(f"Original retweets: {initial_count:,}")
    print(f"Final retweets: {len(df_clean):,}")
    print(f"Removed total: {initial_count - len(df_clean):,}")
    print(f"data retention: {len(df_clean) / initial_count * 100:.1f}%")
    
    try:
        df_clean.to_csv(output_file, index=False)
        print(f"\nSUCCESS Saved cleaned retweets to: {output_file}")
        print(f"   Rows: {len(df_clean):,}")
        print(f"   Columns: {len(df_clean.columns)}")
        
        print(f"\nSample cleaned retweets:")
        print("-" * 50)
        for i, (idx, row) in enumerate(df_clean.head(3).iterrows()):
            print(f"{i + 1}. Author: @{row['originalAuthor']}")
            print(f"   Content: {row['tweet'][:100]}...")
            print(f"   Date: {row['timestamp']}")
            print(f"   Length: {row['originalTextLength']} chars")
            print()
        
    except Exception as e:
        print(f"ERROR Error saving file: {e}")
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
