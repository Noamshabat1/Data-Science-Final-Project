#!/usr/bin/env python3
import pandas as pd
import numpy as np

def create_test_data():
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'day_of_week': np.random.randint(0, 7, n_samples),
        'likeCount_mean': np.random.normal(150000, 50000, n_samples),
        'sentiment_score': np.random.normal(0, 0.3, n_samples)
    }
    
    return pd.DataFrame(data)

def show_problem():
    df = create_test_data()
    print("=== DEMONSTRATING THE PROBLEM ===")
    print("\nDataFrame sample:")
    print(df.head())
    print(f"\nday_of_week column type: {df['day_of_week'].dtype}")
    print(f"day_of_week unique values: {sorted(df['day_of_week'].unique())}")
    
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.bar(df['day_of_week'], df['likeCount_mean'])
        plt.xlabel('Day of Week (Numeric)')
        plt.ylabel('Like Count Mean')
        plt.title('Problem: Numeric day values instead of day names')
        plt.show()
    except ImportError:
        print("Matplotlib not available for demonstration")

def create_day_mapping():
    day_mapping = {
        0: 'Monday',
        1: 'Tuesday', 
        2: 'Wednesday',
        3: 'Thursday',
        4: 'Friday',
        5: 'Saturday',
        6: 'Sunday'
    }
    return day_mapping

def fix_notebook_code():
    fix_code = '''
# Fix for the visualization notebook - replace the plotting code:

import matplotlib.pyplot as plt
import pandas as pd

# Create day name mapping
day_mapping = {
    0: 'Monday',
    1: 'Tuesday', 
    2: 'Wednesday',
    3: 'Thursday',
    4: 'Friday',
    5: 'Saturday',
    6: 'Sunday'
}

# Apply mapping to create day names
df['day_name'] = df['day_of_week'].map(day_mapping)

# Order the days correctly for plotting
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df['day_name'] = pd.Categorical(df['day_name'], categories=day_order, ordered=True)

# Sort by day order
df_sorted = df.sort_values('day_name')

# Create the plot with proper day names
plt.figure(figsize=(12, 6))
plt.bar(df_sorted['day_name'], df_sorted['likeCount_mean'])
plt.xlabel('Day of Week')
plt.ylabel('Average Like Count')
plt.title('Average Tweet Engagement by Day of Week')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
'''
    
    print("To fix the notebook, replace the plotting code with:")
    print(fix_code)

def main():
    show_problem()
    
    day_mapping = create_day_mapping()
    
    print("\nDay mapping dictionary:")
    print(day_mapping)
    
    fix_notebook_code()

if __name__ == "__main__":
    main() 