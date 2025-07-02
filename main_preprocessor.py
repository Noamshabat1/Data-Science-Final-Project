#!/usr/bin/env python3
"""
Main Data Preprocessing Pipeline
===============================

This script orchestrates the complete data preprocessing pipeline for Elon Musk's posts:
1. Split posts into categories (tweets, replies, retweets)
2. Clean each category with category-specific processing
3. Generate summary statistics and reports

Usage:
    python main_preprocessor.py

Output:
    - Split data files in Data/split_posts/
    - Cleaned data files in Data/split_posts/
    - Processing logs and summary reports
"""

import os
import sys
import time
import pandas as pd
from datetime import datetime

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def print_banner(title):
    """Print a formatted banner for section headers"""
    print("\n" + "="*70)
    print(f" {title.upper()} ")
    print("="*70)

def print_step(step_num, description):
    """Print a formatted step description"""
    print(f"\n[STEP {step_num}] {description}")
    print("-" * 50)

def check_file_exists(filepath, description=""):
    """Check if a file exists and print status"""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print(f"✓ {description}: {filepath} ({size:,} bytes)")
        return True
    else:
        print(f"✗ {description}: {filepath} (NOT FOUND)")
        return False

def get_csv_info(filepath):
    """Get basic info about a CSV file"""
    try:
        df = pd.read_csv(filepath)
        return len(df), len(df.columns)
    except Exception as e:
        print(f"   Error reading {filepath}: {e}")
        return 0, 0

def run_preprocessing_pipeline():
    """Run the complete preprocessing pipeline"""
    
    print_banner("Elon Musk Posts - Data Preprocessing Pipeline")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define paths
    data_dir = os.path.join("Data")
    original_dir = os.path.join(data_dir, "original")
    split_dir = os.path.join(data_dir, "splitted")

    
    # Check if input data exists
    input_file = os.path.join(original_dir, "all_musk_posts.csv")
    if not check_file_exists(input_file, "Input data"):
        print("❌ Cannot proceed without input data file!")
        return
    
    try:
        # ====================================================================
        # STEP 1: Data Splitting
        # ====================================================================
        print_step(1, "Data Splitting - Categorizing Posts")
        
        print("Importing DataSplitter...")
        from cleaners.data_splitter import preprocess_musk_posts
        
        print("Running data splitter...")
        start_time = time.time()
        preprocess_musk_posts()
        split_time = time.time() - start_time
        
        print(f"✓ Data splitting completed in {split_time:.2f} seconds")
        
        # Verify split files were created
        split_files = {
            'tweets': os.path.join(split_dir, "musk_tweets.csv"),
            'replies': os.path.join(split_dir, "musk_replies.csv"),
            'retweets': os.path.join(split_dir, "musk_retweets.csv")
        }
        
        print("\nVerifying split files:")
        for category, filepath in split_files.items():
            if check_file_exists(filepath, f"{category.title()}"):
                rows, cols = get_csv_info(filepath)
                print(f"   → {rows:,} rows, {cols} columns")
        
        # ====================================================================
        # STEP 2: Clean Retweets
        # ====================================================================
        print_step(2, "Cleaning Retweets Data")
        
        print("Importing retweets cleaner...")
        from cleaners.retweets_cleaner import clean_retweets_data
        
        print("Running retweets cleaner...")
        start_time = time.time()
        clean_retweets_data()
        retweets_time = time.time() - start_time
        
        print(f"✓ Retweets cleaning completed in {retweets_time:.2f} seconds")
        
        # ====================================================================
        # STEP 3: Clean Tweets
        # ====================================================================
        print_step(3, "Cleaning Tweets Data")
        
        from cleaners.tweets_cleaner import clean_tweets_data
        
        print("Running tweets cleaner...")
        start_time = time.time()
        clean_tweets_data()
        tweets_time = time.time() - start_time
        
        print(f"✓ Tweets cleaning completed in {tweets_time:.2f} seconds")

        
        # ====================================================================
        # STEP 4: Generate Summary Report
        # ====================================================================
        print_step(5, "Generating Summary Report")
        generate_summary_report()
        
        # ====================================================================
        # COMPLETION
        # ====================================================================
        print_banner("Processing Complete")
        total_time = time.time() - start_time if 'start_time' in locals() else 0
        print(f"✅ All preprocessing steps completed successfully!")
        print(f"📊 Check the summary report above for detailed statistics")
        print(f"🕒 Total processing time: {total_time:.2f} seconds")
        print(f"📁 Output files location: {split_dir}")
        
    except Exception as e:
        print(f"\n❌ Error in preprocessing pipeline: {e}")
        print("Check the error details above and ensure all required files exist.")
        sys.exit(1)


def generate_summary_report():
    """Generate a comprehensive summary report of the preprocessing results"""
    
    print("Generating preprocessing summary report...")
    
    # Define file paths
    split_dir = os.path.join("..", "Data", "split_posts")
    
    files_to_check = {
        'Raw Data': {
            'tweets': os.path.join(split_dir, "musk_tweets.csv"),
            'replies': os.path.join(split_dir, "musk_replies.csv"),
            'retweets': os.path.join(split_dir, "musk_retweets.csv")
        },
        'Clean Data': {
            'tweets': os.path.join(split_dir, "clean_musk_tweets.csv"),
            'replies': os.path.join(split_dir, "clean_musk_replies.csv"),
            'retweets': os.path.join(split_dir, "clean_musk_retweets.csv")
        }
    }
    
    print("\n" + "="*60)
    print(" PREPROCESSING SUMMARY REPORT ")
    print("="*60)
    
    for data_type, files in files_to_check.items():
        print(f"\n{data_type}:")
        print("-" * 30)
        
        total_raw = 0
        total_clean = 0
        
        for category, filepath in files.items():
            if os.path.exists(filepath):
                rows, cols = get_csv_info(filepath)
                print(f"  {category.title():<10}: {rows:>8,} rows, {cols:>2} columns")
                
                if data_type == 'Raw Data':
                    total_raw += rows
                else:
                    total_clean += rows
            else:
                print(f"  {category.title():<10}: {'Not found':>15}")
        
        if data_type == 'Clean Data' and total_raw > 0 and total_clean > 0:
            retention = (total_clean / total_raw) * 100 if total_raw > 0 else 0
            print(f"  {'Total':<10}: {total_clean:>8,} rows ({retention:.1f}% retention)")
    
    print(f"\n📁 All output files saved to: {split_dir}")
    print(f"🕒 Report generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    run_preprocessing_pipeline() 