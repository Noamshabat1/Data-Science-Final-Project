import pandas as pd
import os
import re
from datetime import datetime

def clean_tweets_data():
    """
    Cleans and processes the original tweets data with tweet-specific operations:
    - Cleans tweet text and hashtags
    - Handles URLs and media links
    - Removes duplicates and invalid tweets
    - Processes tweet metrics and engagement
    """
    
    # Define file paths
    input_file = os.path.join("..", "Data", "split_posts", "musk_tweets.csv")
    output_file = os.path.join("..", "Data", "split_posts", "clean_musk_tweets.csv")
    
    print("="*60)
    print("TWEETS DATA CLEANER")
    print("="*60)
    
    # Load the tweets data
    try:
        df = pd.read_csv(input_file)
        print(f"✓ Loaded {len(df):,} tweets from {input_file}")
        print(f"  Columns: {len(df.columns)}")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return
    
    print(f"\nInitial data shape: {df.shape}")
    
    # Display basic info
    print(f"\nData Overview:")
    print(f"- Date range: {df['createdAt'].min()} to {df['createdAt'].max()}")
    print(f"- Missing fullText: {df['fullText'].isna().sum()}")
    print(f"- Average likes: {df['likeCount'].mean():.0f}")
    print(f"- Average retweets: {df['retweetCount'].mean():.0f}")
    
    # Start cleaning process
    print(f"\n" + "-"*40)
    print("CLEANING PROCESS")
    print("-"*40)
    
    # 1. Remove tweets with missing text
    initial_count = len(df)
    df_clean = df.dropna(subset=['fullText']).copy()
    removed_empty = initial_count - len(df_clean)
    if removed_empty > 0:
        print(f"✓ Removed {removed_empty} tweets with missing text")
    
    # 2. Clean and extract tweet content
    def extract_tweet_content(text):
        """Extract and clean tweet content"""
        if pd.isna(text):
            return None, [], [], []
            
        text_str = str(text).strip()
        
        # Extract hashtags
        hashtags = re.findall(r'#(\w+)', text_str)
        
        # Extract mentions
        mentions = re.findall(r'@(\w+)', text_str)
        
        # Extract URLs
        urls = re.findall(r'https?://[^\s]+', text_str)
        
        return text_str, hashtags, mentions, urls
    
    print("✓ Extracting tweet content...")
    
    # Apply content extraction
    extraction_results = df_clean['fullText'].apply(extract_tweet_content)
    df_clean['originalText'] = [x[0] for x in extraction_results]
    df_clean['hashtags'] = [x[1] for x in extraction_results]
    df_clean['mentions'] = [x[2] for x in extraction_results]
    df_clean['urls'] = [x[3] for x in extraction_results]
    
    # 3. Add content features
    df_clean['hashtagCount'] = df_clean['hashtags'].apply(len)
    df_clean['mentionCount'] = df_clean['mentions'].apply(len)
    df_clean['urlCount'] = df_clean['urls'].apply(len)
    df_clean['hasHashtags'] = df_clean['hashtagCount'] > 0
    df_clean['hasMentions'] = df_clean['mentionCount'] > 0
    df_clean['hasUrls'] = df_clean['urlCount'] > 0
    
    # 4. Clean timestamps
    print("✓ Processing timestamps...")
    df_clean['createdAt'] = pd.to_datetime(df_clean['createdAt'])
    df_clean['year'] = df_clean['createdAt'].dt.year
    df_clean['month'] = df_clean['createdAt'].dt.month
    df_clean['day_of_week'] = df_clean['createdAt'].dt.day_name()
    df_clean['hour'] = df_clean['createdAt'].dt.hour
    df_clean['is_weekend'] = df_clean['createdAt'].dt.weekday >= 5
    
    # 5. Clean and standardize text content
    def clean_text_content(text):
        """Advanced text cleaning for tweets"""
        if pd.isna(text):
            return ""
        
        text = str(text)
        
        # Remove HTML entities
        text = re.sub(r'&amp;', '&', text)
        text = re.sub(r'&lt;', '<', text)
        text = re.sub(r'&gt;', '>', text)
        text = re.sub(r'&quot;', '"', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove zero-width characters
        text = text.replace('\u200b', '').replace('\u200c', '').replace('\u200d', '')
        
        # Standardize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        return text
    
    print("✓ Cleaning text content...")
    df_clean['cleanedText'] = df_clean['originalText'].apply(clean_text_content)
    
    # 6. Create text without URLs and mentions for analysis
    def create_pure_text(text, urls, mentions):
        """Remove URLs and mentions to get pure tweet content"""
        if pd.isna(text):
            return ""
        
        clean_text = str(text)
        
        # Remove URLs
        for url in urls:
            clean_text = clean_text.replace(url, '')
        
        # Remove mentions
        for mention in mentions:
            clean_text = clean_text.replace(f'@{mention}', '')
        
        # Clean up extra spaces
        clean_text = ' '.join(clean_text.split())
        
        return clean_text
    
    print("✓ Creating pure text content...")
    df_clean['pureText'] = df_clean.apply(
        lambda row: create_pure_text(row['cleanedText'], row['urls'], row['mentions']), 
        axis=1
    )
    
    # 7. Add engagement and content metrics
    print("✓ Adding engagement metrics...")
    
    # Fill missing engagement metrics with 0
    engagement_cols = ['likeCount', 'retweetCount', 'replyCount', 'quoteCount']
    for col in engagement_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(0)
    
    # Calculate engagement metrics
    if all(col in df_clean.columns for col in engagement_cols):
        df_clean['totalEngagement'] = (
            df_clean['likeCount'] + 
            df_clean['retweetCount'] + 
            df_clean['replyCount'] + 
            df_clean['quoteCount']
        )
        df_clean['engagementRate'] = df_clean['totalEngagement'] / (df_clean['viewCount'].fillna(1).replace(0, 1))
    
    # Text length metrics
    df_clean['textLength'] = df_clean['cleanedText'].str.len()
    df_clean['pureTextLength'] = df_clean['pureText'].str.len()
    df_clean['wordCount'] = df_clean['pureText'].str.split().str.len()
    
    # Content type classification
    df_clean['isLinkTweet'] = df_clean['hasUrls'] & (df_clean['pureTextLength'] < 50)
    df_clean['isMediaTweet'] = df_clean['cleanedText'].str.contains('https://t.co/', na=False)
    df_clean['isQuestionTweet'] = df_clean['cleanedText'].str.contains(r'\?', na=False)
    
    # Sentiment indicators (basic)
    df_clean['hasEmoji'] = df_clean['cleanedText'].str.contains(r'[😀-🙿]|[🚀-🛿]|[⚡️💎🔥👍👎❤️😂😍🤔💯]', na=False)
    df_clean['hasExclamation'] = df_clean['cleanedText'].str.contains('!', na=False)
    
    # 8. Remove very short tweets (likely corrupted or incomplete)
    before_length_filter = len(df_clean)
    df_clean = df_clean[df_clean['pureTextLength'] >= 3]  # At least 3 characters of actual content
    removed_short = before_length_filter - len(df_clean)
    if removed_short > 0:
        print(f"✓ Removed {removed_short} very short tweets")
    
    # 9. Remove exact duplicates
    before_dedup = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset='pureText', keep='first')
    removed_duplicates = before_dedup - len(df_clean)
    if removed_duplicates > 0:
        print(f"✓ Removed {removed_duplicates} duplicate tweets")
    
    # 10. Sort by creation date
    df_clean = df_clean.sort_values(by='createdAt')
    
    # Final statistics
    print(f"\n" + "-"*40)
    print("CLEANING SUMMARY")
    print("-"*40)
    print(f"Original tweets: {initial_count:,}")
    print(f"Final tweets: {len(df_clean):,}")
    print(f"Removed total: {initial_count - len(df_clean):,}")
    print(f"Data retention: {len(df_clean)/initial_count*100:.1f}%")
    
    print(f"\nTweet content statistics:")
    print(f"- Average text length: {df_clean['textLength'].mean():.1f} chars")
    print(f"- Average word count: {df_clean['wordCount'].mean():.1f} words")
    print(f"- Tweets with hashtags: {df_clean['hasHashtags'].sum():,} ({df_clean['hasHashtags'].mean()*100:.1f}%)")
    print(f"- Tweets with mentions: {df_clean['hasMentions'].sum():,} ({df_clean['hasMentions'].mean()*100:.1f}%)")
    print(f"- Tweets with URLs: {df_clean['hasUrls'].sum():,} ({df_clean['hasUrls'].mean()*100:.1f}%)")
    print(f"- Tweets with emoji: {df_clean['hasEmoji'].sum():,} ({df_clean['hasEmoji'].mean()*100:.1f}%)")
    
    if 'totalEngagement' in df_clean.columns:
        print(f"\nEngagement statistics:")
        print(f"- Average total engagement: {df_clean['totalEngagement'].mean():.0f}")
        print(f"- Average likes: {df_clean['likeCount'].mean():.0f}")
        print(f"- Average retweets: {df_clean['retweetCount'].mean():.0f}")
    
    print(f"\nMost used hashtags:")
    all_hashtags = [tag for tags in df_clean['hashtags'] for tag in tags]
    if all_hashtags:
        hashtag_counts = pd.Series(all_hashtags).value_counts().head(10)
        print(hashtag_counts)
    
    print(f"\nTweets by year:")
    print(df_clean['year'].value_counts().sort_index())
    
    # 11. Save cleaned data
    try:
        df_clean.to_csv(output_file, index=False)
        print(f"\n✅ Saved cleaned tweets to: {output_file}")
        print(f"   Rows: {len(df_clean):,}")
        print(f"   Columns: {len(df_clean.columns)}")
        
        # Show sample of cleaned data
        print(f"\nSample cleaned tweets:")
        print("-" * 50)
        for i in range(min(3, len(df_clean))):
            row = df_clean.iloc[i]
            print(f"{i+1}. Content: {row['pureText'][:100]}...")
            print(f"   Hashtags: {row['hashtags']}")
            print(f"   Date: {row['createdAt']}")
            print(f"   Length: {row['textLength']} chars, {row['wordCount']} words")
            if 'totalEngagement' in row:
                print(f"   Engagement: {row['totalEngagement']:.0f}")
            print()
            
    except Exception as e:
        print(f"❌ Error saving file: {e}")

if __name__ == "__main__":
    clean_tweets_data() 