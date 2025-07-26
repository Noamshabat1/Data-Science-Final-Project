from cleaners.data_splitter import main_data_splitter
from cleaners.tweets_cleaner import clean_tweets_data
from cleaners.replies_cleaner import clean_replies
from cleaners.retweets_cleaner import clean_retweets_data
from cleaners.tesla_stock_cleaner import clean_stock_data

def main():
    print("Starting data cleaning pipeline...")
    
    # Collect statistics from each cleaning step
    split_stats = main_data_splitter()
    tweets_stats = clean_tweets_data()
    replies_stats = clean_replies()
    retweets_stats = clean_retweets_data()
    tesla_stats = clean_stock_data()
    
    # Combine all statistics
    all_stats = {
        'split': split_stats,
        'tweets': tweets_stats,
        'replies': replies_stats,
        'retweets': retweets_stats,
    }
    
    print("\nData cleaning completed. Generating visualizations...")
    
    # Generate visualizations
    try:
        from visualization.data_cleaning_plots import generate_all_plots
        generate_all_plots(all_stats)
        print("✅ Visualizations generated successfully!")
    except Exception as e:
        print(f"❌ Error generating visualizations: {e}")
    
    return all_stats

if __name__ == "__main__":
    main() 