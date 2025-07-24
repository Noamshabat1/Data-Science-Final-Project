from cleaners.data_splitter import main_data_splitter
from cleaners.tweets_cleaner import clean_tweets_data
from cleaners.replies_cleaner import clean_replies
from cleaners.retweets_cleaner import clean_retweets_data
from cleaners.tesla_stock_cleaner import clean_stock_data
from cleaners.sp500_stock_cleaner import clean_sp500

def main():
    main_data_splitter()
    clean_tweets_data()
    clean_replies()
    clean_retweets_data()
    clean_sp500()
    clean_stock_data()

if __name__ == "__main__":
    main() 