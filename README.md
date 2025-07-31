# The Musk Effect: Twitter Activity vs. Stock Price Dynamics

A data science project that analyzes and explores Elon Musk's Twitter activity (tweets, replies, retweets).

## Features

- **Data Processing Pipeline**: Automated cleaning and preprocessing of Twitter data and Tesla stock prices
- **Sentiment Analysis**: VADER sentiment analysis on social media posts
- **Community Detection**: Social network analysis using Louvain algorithm
- **Machine Learning Models**: XGBoost, Random Forest, and SVR with hyperparameter optimization
- **Feature Engineering**: Combines sentiment scores, social metrics, and technical indicators
- **Model Evaluation**: RMSE, MAE, R², directional accuracy, and comprehensive visualizations

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the complete pipeline
python main.py
```

## Project Structure

- `cleaners/`: Data cleaning modules for tweets, stock data
- `model/`: ML model training, evaluation, and output artifacts
- `community/`: Social network analysis and community detection
- `visualization/`: Data visualization and plotting utilities
- `data/`: Raw, cleaned, and processed datasets

The pipeline processes social media data, extracts sentiment features, builds predictive models, and generates performance visualizations automatically.