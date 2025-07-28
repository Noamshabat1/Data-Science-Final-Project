import pandas as pd
import matplotlib.pyplot as plt

# Load data
stock = pd.read_csv('data/clean/clean_tesla_stock.csv', parse_dates=['Date'])
tweets = pd.read_csv('data/clean/clean_musk_tweets.csv', parse_dates=['timestamp'])

# Prepare tweets: aggregate by day
tweets['date'] = tweets['timestamp'].dt.date
daily_likes = tweets.groupby('date')['likeCount'].sum().reset_index()
daily_likes['date'] = pd.to_datetime(daily_likes['date'])

# Prepare stock: ensure date alignment
stock['Date'] = pd.to_datetime(stock['Date'])

# Merge on date
merged = pd.merge(stock, daily_likes, left_on='Date', right_on='date', how='left')

# Fill missing likeCounts with 0 (no tweets that day)
merged['likeCount'] = merged['likeCount'].fillna(0)

# Plot
fig, ax1 = plt.subplots(figsize=(14,6))

ax1.plot(merged['Date'], merged['Close'], color='tab:blue', label='TSLA Close Price')
ax1.set_ylabel('TSLA Close Price', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.bar(merged['Date'], merged['likeCount'], color='tab:orange', alpha=0.3, label='Total Tweet Likes')
ax2.set_ylabel('Total Tweet Likes', color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:orange')

plt.title('Tesla Stock Price vs. Elon Musk Tweet LikeCount (Daily)')
fig.tight_layout()
plt.show()