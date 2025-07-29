import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

plt.style.use('default')
sns.set_palette("husl")

def ensure_output_dir():
    output_dir = Path('visualization/plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def create_data_flow_chart(stats):
    stages = ['Original', 'After Remove Empty', 'After Remove Duplicates', 'After Text Filter', 'Final']
    
    data_types = ['split', 'tweets', 'replies', 'retweets']
    colors = plt.cm.Set3(np.linspace(0, 1, len(data_types)))
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    bar_width = 0.2
    x_positions = np.arange(len(stages))
    
    for i, (data_type, color) in enumerate(zip(data_types, colors)):
        if data_type == 'split':
            values = [
                stats[data_type]['total'],
                stats[data_type]['total'],
                stats[data_type]['total'], 
                stats[data_type]['total'],
                stats[data_type]['total']
            ]
        else:
            values = [
                stats[data_type]['initial_count'],
                stats[data_type]['initial_count'] - stats[data_type].get('removed_empty', 0),
                stats[data_type]['initial_count'] - stats[data_type].get('removed_empty', 0) - stats[data_type].get('removed_duplicates', 0),
                stats[data_type]['initial_count'] - stats[data_type].get('removed_empty', 0) - stats[data_type].get('removed_duplicates', 0) - stats[data_type].get('removed_non_alpha', 0),
                stats[data_type]['final_count']
            ]
        
        bars = ax.bar(x_positions + i * bar_width, values, bar_width, 
                     label=data_type.title(), color=color, alpha=0.8)
        
        for j, (bar, value) in enumerate(zip(bars, values)):
            if value > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values) * 0.01,
                       f'{value:,}', ha='center', va='bottom', fontsize=8, rotation=0)
    
    ax.set_xlabel('Processing Stage', fontweight='bold')
    ax.set_ylabel('Number of Records', fontweight='bold')
    ax.set_title('Data Cleaning Pipeline - Record Count Flow', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x_positions + bar_width * 1.5)
    ax.set_xticklabels(stages)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_removal_analysis(stats):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    data_types = ['tweets', 'replies', 'retweets']
    
    for idx, data_type in enumerate(data_types):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        removal_categories = []
        removal_counts = []
        
        if 'removed_empty' in stats[data_type] and stats[data_type]['removed_empty'] > 0:
            removal_categories.append('Empty Text')
            removal_counts.append(stats[data_type]['removed_empty'])
        
        if 'removed_duplicates' in stats[data_type] and stats[data_type]['removed_duplicates'] > 0:
            removal_categories.append('Duplicates')
            removal_counts.append(stats[data_type]['removed_duplicates'])
        
        if 'removed_non_alpha' in stats[data_type] and stats[data_type]['removed_non_alpha'] > 0:
            removal_categories.append('Non-alphanumeric')
            removal_counts.append(stats[data_type]['removed_non_alpha'])
        
        if 'removed_unparseable' in stats[data_type] and stats[data_type]['removed_unparseable'] > 0:
            removal_categories.append('Unparseable')
            removal_counts.append(stats[data_type]['removed_unparseable'])
        
        if 'removed_short' in stats[data_type] and stats[data_type]['removed_short'] > 0:
            removal_categories.append('Too Short')
            removal_counts.append(stats[data_type]['removed_short'])
        
        removal_categories.append('Retained')
        removal_counts.append(stats[data_type]['final_count'])
        
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(removal_categories)))
        colors[-1] = plt.cm.RdYlGn(0.8)
        
        bars = ax.bar(removal_categories, removal_counts, color=colors, alpha=0.8)
        
        for bar, count in zip(bars, removal_counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(removal_counts) * 0.01,
                       f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title(f'{data_type.title()} - Removal Analysis', fontweight='bold')
        ax.set_ylabel('Count')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
    
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    return fig

def create_retention_summary(stats):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    data_types = ['split', 'tweets', 'replies', 'retweets']
    
    retention_rates = []
    final_counts = []
    initial_counts = []
    
    for data_type in data_types:
        if data_type == 'split':
            initial = stats[data_type]['total']
            final = stats[data_type]['total']
            retention = 100.0
        else:
            initial = stats[data_type]['initial_count']
            final = stats[data_type]['final_count']
            retention = stats[data_type]['retention_rate']
        
        retention_rates.append(retention)
        final_counts.append(final)
        initial_counts.append(initial)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(data_types)))
    
    bars1 = ax1.bar(data_types, retention_rates, color=colors, alpha=0.8)
    ax1.set_ylabel('Retention Rate (%)', fontweight='bold')
    ax1.set_title('Data Retention Rates by Type', fontweight='bold')
    ax1.set_ylim(0, 105)
    
    for bar, rate in zip(bars1, retention_rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax1.grid(axis='y', alpha=0.3)
    
    x_pos = np.arange(len(data_types))
    width = 0.35
    
    bars2 = ax2.bar(x_pos - width/2, initial_counts, width, label='Initial', 
                   color='lightcoral', alpha=0.8)
    bars3 = ax2.bar(x_pos + width/2, final_counts, width, label='Final', 
                   color='lightblue', alpha=0.8)
    
    ax2.set_ylabel('Count', fontweight='bold')
    ax2.set_title('Initial vs Final Record Counts', fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([dt.title() for dt in data_types])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    for bars in [bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, height + max(initial_counts + final_counts) * 0.01,
                        f'{int(height):,}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig

def generate_all_plots(stats):
    output_dir = ensure_output_dir()
    
    plots = [
        ('data_flow_chart', create_data_flow_chart),
        ('removal_analysis', create_removal_analysis), 
        ('retention_summary', create_retention_summary)
    ]
    
    print("Generating data cleaning visualizations...")
    
    for plot_name, plot_func in plots:
        fig = plot_func(stats)
        save_path = output_dir / f"{plot_name}.png"
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"SUCCESS Saved {plot_name} to {save_path}")
        plt.close(fig)
    
    print(f"\nRESULTS All plots saved to: {output_dir.absolute()}")

if __name__ == "__main__":
    sample_stats = {
        'split': {'total': 50000},
        'tweets': {'initial_count': 25000, 'removed_empty': 500, 'removed_duplicates': 1000, 'removed_non_alpha': 200, 'final_count': 23300, 'retention_rate': 93.2},
        'replies': {'initial_count': 15000, 'removed_empty': 200, 'removed_duplicates': 300, 'final_count': 14500, 'retention_rate': 96.7},
        'retweets': {'initial_count': 10000, 'removed_empty': 100, 'removed_duplicates': 400, 'removed_unparseable': 200, 'removed_short': 50, 'final_count': 9250, 'retention_rate': 92.5}
    }
    
    generate_all_plots(sample_stats)
