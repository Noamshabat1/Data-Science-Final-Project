#!/usr/bin/env python3
"""
Fix for converting integer day values to day names in plots.

This script demonstrates how to fix the issue where plots show integer day values
(0-6) instead of day names (Monday, Tuesday, etc.).
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_sample_data():
    """Create sample data similar to the notebook structure."""
    # Create sample data with integer day values
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    
    # Create sample posts data
    sample_data = []
    for date in dates:
        # Random number of posts per day
        num_posts = np.random.poisson(5)  # Average 5 posts per day
        for _ in range(num_posts):
            sample_data.append({
                'timestamp': date,
                'text': f'Sample post on {date.strftime("%Y-%m-%d")}',
                'likeCount': np.random.randint(0, 1000)
            })
    
    df = pd.DataFrame(sample_data)
    
    # Add day of week as integer (0-6)
    df['day'] = df['timestamp'].dt.day_of_week
    
    # Add day of week as name (for comparison)
    df['day_name'] = df['timestamp'].dt.day_name()
    
    return df

def demonstrate_the_problem():
    """Show the current issue with integer day values."""
    print("=== DEMONSTRATING THE PROBLEM ===")
    
    df = create_sample_data()
    
    # Group by day (integer values)
    daily_activity = df.groupby('day').size()
    
    print("Current data structure:")
    print(daily_activity)
    print("\nThis shows integer values (0-6) instead of day names!")
    
    # Create a plot showing the problem
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Problematic plot with integer values
    ax1.bar(daily_activity.index, daily_activity.values, color='red', alpha=0.7)
    ax1.set_xlabel('Day of Week (Integer)')
    ax1.set_ylabel('Number of Posts')
    ax1.set_title('PROBLEM: Integer Day Values')
    ax1.set_xticks(daily_activity.index)
    ax1.set_xticklabels(daily_activity.index)
    
    # Plot 2: Fixed plot with day names
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_mapping = {i: day_names[i] for i in range(7)}
    
    ax2.bar(range(len(daily_activity)), daily_activity.values, color='green', alpha=0.7)
    ax2.set_xlabel('Day of Week')
    ax2.set_ylabel('Number of Posts')
    ax2.set_title('SOLUTION: Day Names')
    ax2.set_xticks(range(len(daily_activity)))
    ax2.set_xticklabels([day_mapping[i] for i in daily_activity.index], rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('day_names_comparison.png', dpi=300, bbox_inches='tight')
    print("\nComparison plot saved as 'day_names_comparison.png'")
    
    return daily_activity, day_mapping

def provide_fix_code():
    """Provide the exact code to fix the notebook."""
    fix_code = """
# ===== FIX FOR THE NOTEBOOK =====

# Original problematic code:
# dow_activity = all_posts.groupby('day_of_week').size()
# axes[2].bar(dow_activity.index, dow_activity.values, color='orange', alpha=0.7)

# REPLACE WITH THIS FIXED CODE:

# Step 1: Create the day mapping
day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_mapping = {i: day_names[i] for i in range(7)}

# Step 2: Get the activity data
dow_activity = all_posts.groupby('day_of_week').size()

# Step 3: Create the plot with proper day names
axes[2].bar(range(len(dow_activity)), dow_activity.values, color='orange', alpha=0.7)
axes[2].set_xticks(range(len(dow_activity)))
axes[2].set_xticklabels([day_mapping[i] for i in dow_activity.index], rotation=45, ha='right')
axes[2].set_xlabel('Day of Week')
axes[2].set_ylabel('Total Posts')
axes[2].set_title('Twitter/X Activity by Day of Week', fontsize=14, fontweight='bold')

# ===== ALTERNATIVE FIX =====
# If you want to use day names from the start:

# Instead of:
# all_posts['day_of_week'] = all_posts['timestamp'].dt.day_name()

# Use:
all_posts['day_of_week'] = all_posts['timestamp'].dt.day_name()

# Then the plotting becomes simpler:
dow_activity = all_posts.groupby('day_of_week').size()
axes[2].bar(range(len(dow_activity)), dow_activity.values, color='orange', alpha=0.7)
axes[2].set_xticks(range(len(dow_activity)))
axes[2].set_xticklabels(dow_activity.index, rotation=45, ha='right')
"""
    
    print("\n=== FIX CODE FOR NOTEBOOK ===")
    print(fix_code)
    
    return fix_code

def main():
    """Run the demonstration and provide the fix."""
    print("Day Names Fix for Plotting")
    print("=" * 40)
    
    # Demonstrate the problem
    daily_activity, day_mapping = demonstrate_the_problem()
    
    # Provide the fix
    fix_code = provide_fix_code()
    
    print("\n=== SUMMARY ===")
    print("The issue is that the plot is using integer day values (0-6) instead of day names.")
    print("The fix involves creating a mapping from integers to day names and using it for x-axis labels.")
    print("\nDay mapping:")
    for i, name in day_mapping.items():
        print(f"  {i} -> {name}")

if __name__ == "__main__":
    main() 