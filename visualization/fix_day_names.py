import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def convert_day_integers_to_names():
    """
    Example of how to convert integer day values (0-6) to day names for plotting.
    This fixes the issue where plots show integers instead of day names.
    """
    
    # Sample data with integer day values (0-6)
    # Monday = 0, Tuesday = 1, Wednesday = 2, Thursday = 3, Friday = 4, Saturday = 5, Sunday = 6
    sample_data = {
        'day': [0, 1, 2, 3, 4, 5, 6],
        'count': [10, 15, 12, 18, 20, 8, 5]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Method 1: Create a mapping dictionary
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_mapping = {i: day_names[i] for i in range(7)}
    
    # Method 2: Use pandas day_name() function
    # This is what should be used when creating the day column
    # df['day_name'] = pd.to_datetime(df['date']).dt.day_name()
    
    # Create the plot with proper day names
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Using integer values (current issue)
    ax1.bar(df['day'], df['count'], color='red', alpha=0.7)
    ax1.set_xlabel('Day of Week (Integer)')
    ax1.set_ylabel('Count')
    ax1.set_title('Current Issue: Integer Day Values')
    ax1.set_xticks(df['day'])
    ax1.set_xticklabels(df['day'])
    
    # Plot 2: Using day names (fixed)
    ax2.bar(range(len(df)), df['count'], color='green', alpha=0.7)
    ax2.set_xlabel('Day of Week')
    ax2.set_ylabel('Count')
    ax2.set_title('Fixed: Day Names')
    ax2.set_xticks(range(len(df)))
    ax2.set_xticklabels([day_mapping[i] for i in df['day']], rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()
    
    return day_mapping

def fix_notebook_code():
    """
    Code to fix the notebook plotting issue.
    Replace the existing plotting code with this:
    """
    
    fix_code = """
# Original problematic code:
# dow_activity = all_posts.groupby('day_of_week').size()
# axes[2].bar(dow_activity.index, dow_activity.values, color='orange', alpha=0.7)

# Fixed code:
dow_activity = all_posts.groupby('day_of_week').size()

# Convert integer day values to day names for the plot
day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_mapping = {i: day_names[i] for i in range(7)}

# Create the plot with proper day names
axes[2].bar(range(len(dow_activity)), dow_activity.values, color='orange', alpha=0.7)
axes[2].set_xticks(range(len(dow_activity)))
axes[2].set_xticklabels([day_mapping[i] for i in dow_activity.index], rotation=45, ha='right')
axes[2].set_xlabel('Day of Week')
axes[2].set_ylabel('Total Posts')
axes[2].set_title('Twitter/X Activity by Day of Week', fontsize=14, fontweight='bold')
"""
    
    print("To fix the notebook, replace the plotting code with:")
    print(fix_code)
    
    return fix_code

if __name__ == "__main__":
    # Demonstrate the fix
    day_mapping = convert_day_integers_to_names()
    fix_code = fix_notebook_code()
    
    print("\nDay mapping dictionary:")
    print(day_mapping) 