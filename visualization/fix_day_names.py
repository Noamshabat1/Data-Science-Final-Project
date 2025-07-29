import numpy as np
import pandas as pd

day_mapping = {
    0: 'Monday',
    1: 'Tuesday',
    2: 'Wednesday', 
    3: 'Thursday',
    4: 'Friday',
    5: 'Saturday',
    6: 'Sunday'
}

day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

def convert_day_integers_to_names(df, day_column='day_of_week'):
    df_fixed = df.copy()
    df_fixed['day_name'] = df_fixed[day_column].map(day_mapping)
    df_fixed['day_name'] = pd.Categorical(df_fixed['day_name'], categories=day_order, ordered=True)
    return df_fixed

def group_and_plot_by_day(df, value_column, plot_title="Activity by Day", save_path=None):
    try:
        import matplotlib.pyplot as plt
        
        daily_stats = df.groupby('day_name')[value_column].agg(['count', 'mean', 'sum']).reset_index()
        daily_stats = daily_stats.sort_values('day_name')
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].bar(daily_stats['day_name'], daily_stats['count'], color='skyblue', alpha=0.8)
        axes[0].set_title('Count by Day')
        axes[0].set_ylabel('Count')
        plt.setp(axes[0].get_xticklabels(), rotation=45, ha='right')
        
        axes[1].bar(daily_stats['day_name'], daily_stats['mean'], color='lightgreen', alpha=0.8) 
        axes[1].set_title('Average by Day')
        axes[1].set_ylabel('Average Value')
        plt.setp(axes[1].get_xticklabels(), rotation=45, ha='right')
        
        axes[2].bar(daily_stats['day_name'], daily_stats['sum'], color='lightcoral', alpha=0.8)
        axes[2].set_title('Total by Day')
        axes[2].set_ylabel('Total Value')
        plt.setp(axes[2].get_xticklabels(), rotation=45, ha='right')
        
        plt.suptitle(plot_title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
        
        return daily_stats
        
    except ImportError:
        print("Matplotlib not available. Returning statistics only.")
        daily_stats = df.groupby('day_name')[value_column].agg(['count', 'mean', 'sum']).reset_index()
        return daily_stats.sort_values('day_name')

def demonstrate_fix():
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'day_of_week': np.random.randint(0, 7, 200),
        'likeCount': np.random.normal(15000, 5000, 200),
        'post_length': np.random.normal(100, 30, 200)
    })
    
    print("Original data (with integer days):")
    print(sample_data.head())
    print(f"\nday_of_week type: {sample_data['day_of_week'].dtype}")
    
    fixed_data = convert_day_integers_to_names(sample_data)
    
    print("\nFixed data (with day names):")
    print(fixed_data[['day_of_week', 'day_name', 'likeCount']].head())
    print(f"\nday_name type: {fixed_data['day_name'].dtype}")
    
    stats = group_and_plot_by_day(fixed_data, 'likeCount', 
                                 "Like Count by Day of Week", 
                                 "fixed_day_plot.png")
    
    print("\nDaily statistics:")
    print(stats)

if __name__ == "__main__":
    demonstrate_fix() 