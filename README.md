# Tesla Stock Prediction & Elon Musk Tweet Analysis

A comprehensive data science project that analyzes the relationship between Elon Musk's Twitter activity and Tesla stock price movements using machine learning, community detection, and advanced data visualization techniques.

## 🎯 Project Overview

This project investigates whether Elon Musk's social media activity contains predictive signals for Tesla stock price movements. Through advanced NLP, machine learning, and network analysis, we explore patterns in tweet communities and their correlation with significant stock price changes.

### Key Research Questions
- Can Elon Musk's tweet patterns predict Tesla stock price movements?
- Which tweet communities show increased activity before major stock events?
- How do different types of social media engagement correlate with market performance?

## 🚀 Features

### 📊 Data Processing & Cleaning
- **Multi-source data integration**: Tweets, retweets, replies, and Tesla stock data
- **Automated data cleaning pipeline**: Handles missing values, duplicates, and data quality issues
- **Smart data splitting**: Separates different types of social media content for focused analysis

### 🤖 Machine Learning Models
- **Stock price prediction**: XGBoost and Random Forest models with optimized hyperparameters
- **Advanced preprocessing**: Target normalization, feature engineering, and SMOTE data augmentation
- **Performance metrics**: R², RMSE, directional accuracy, and comprehensive model evaluation

### 🕸️ Community Detection
- **Tweet similarity analysis**: Using sentence transformers and cosine similarity
- **Louvain community detection**: Identifies clusters of related tweets
- **Predictive community analysis**: Finds communities that increase activity before major stock movements

### 📈 Visualization & Analysis
- **Interactive plots**: Stock price predictions, community networks, and engagement trends
- **Time series analysis**: Chronological comparison of predictions vs. actual prices
- **Community insights**: Visualizations showing which tweet topics correlate with market events

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd Data-Science-Final-Project

# Install dependencies
pip install -r requirements.txt

# Run the complete data processing pipeline
python data_processing_runner.py
```

### Dependencies
- **Data Analysis**: pandas, numpy, scipy
- **Machine Learning**: scikit-learn, xgboost, joblib
- **NLP & Text Analysis**: sentence-transformers, nltk, wordcloud
- **Network Analysis**: networkx, python-louvain
- **Visualization**: matplotlib, seaborn

## 📁 Project Structure

```
├── cleaners/                   # Data cleaning modules
│   ├── data_splitter.py       # Splits raw data into tweets/retweets/replies
│   ├── tweets_cleaner.py      # Cleans and processes tweet data
│   ├── replies_cleaner.py     # Processes reply data
│   ├── retweets_cleaner.py    # Handles retweet cleaning
│   └── tesla_stock_cleaner.py # Stock data preprocessing
├── data/                      # Data storage
│   ├── original/              # Raw data files
│   ├── clean/                 # Processed data
│   ├── splitted/              # Data split by type
│   └── model/                 # ML-ready datasets
├── model/                     # Machine learning pipeline
│   ├── build_data.py          # Feature engineering
│   ├── models.py              # Model training and evaluation
│   └── output/                # Trained models and results
├── community/                 # Community detection analysis
│   ├── community_detection.py # Main community analysis
│   └── cache_utils.py         # Caching utilities
├── visualization/             # Data visualization
│   ├── data_cleaning_plots.py # Cleaning process visualizations
│   └── sentiment_vs_stock.py  # Sentiment analysis plots
└── data_processing_runner.py  # Main pipeline orchestrator
```

## 🚀 Usage

### Quick Start
```bash
# Run the complete analysis pipeline
python data_processing_runner.py
```

### Individual Components

#### 1. Machine Learning Analysis
```bash
cd model
python models.py
```
**Outputs**: 
- Trained models saved to `model/output/`
- Performance metrics and visualizations
- Feature importance analysis

#### 2. Community Detection
```bash
cd community  
python community_detection.py
```
**Outputs**:
- Community network visualizations
- Tweet communities that predict stock movements
- Modularity scores and community statistics

#### 3. Data Visualization
```bash
cd visualization
python data_cleaning_plots.py
```
**Outputs**:
- Data quality and cleaning visualizations
- Summary statistics plots

## 📊 Key Results & Insights

### Machine Learning Performance
- **Best Model**: XGBoost with R² ≈ 0.87 and 92%+ directional accuracy
- **Predictive Features**: Tweet sentiment, engagement metrics, and temporal patterns
- **Model Robustness**: Validated with cross-validation and multiple evaluation metrics

### Community Analysis Findings
- **Predictive Communities**: Identified specific tweet communities that increase activity 1-2 days before major stock movements
- **Market Correlation**: Communities discussing "tesla", "amp", and company developments show strongest predictive signals
- **Temporal Patterns**: Tweet volume spikes precede significant stock price changes by 24-48 hours

### Data Quality Insights
- **Data Retention**: 85%+ data retention rate after comprehensive cleaning
- **Engagement Patterns**: Peak tweet activity correlates with market volatility periods
- **Content Analysis**: Technical discussions and company announcements show highest market correlation

## 🔬 Technical Methodology

### Data Preprocessing
1. **Cleaning Pipeline**: Removes duplicates, handles missing values, standardizes formats
2. **Feature Engineering**: Creates sentiment scores, engagement metrics, and temporal features
3. **Data Augmentation**: SMOTE regression for balanced training sets

### Machine Learning Approach
1. **Model Selection**: Tree-based models (XGBoost, Random Forest) optimized for financial time series
2. **Hyperparameter Optimization**: Grid search with cross-validation
3. **Evaluation Metrics**: Multiple metrics including directional accuracy for trading applications

### Community Detection Algorithm
1. **Similarity Computation**: Sentence transformers for semantic tweet similarity
2. **Graph Construction**: K-nearest neighbors approach with cosine similarity
3. **Community Identification**: Louvain algorithm with post-processing merger of small communities

## 📈 Model Performance

| Model | R² Score | RMSE | Directional Accuracy |
|-------|----------|------|---------------------|
| XGBoost (Optimized) | 0.865 | 0.384 | 92.2% |
| Random Forest | 0.862 | 0.388 | 92.6% |
| SVR Baseline | -0.170 | 1.131 | 51.1% |