import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import os

# Set style for better-looking plots
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.alpha'] = 0.3

def create_output_dir():
    """Create output directory for plots"""
    output_dir = Path("visualization/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def plot_data_flow_bar_chart(stats):
    """Create a bar chart showing data flow through cleaning pipeline"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Prepare data for the flow chart
    categories = []
    initial_counts = []
    final_counts = []
    retention_rates = []
    
    # Split data
    if stats['split']:
        categories.extend(['Original Posts Total', 'Tweets', 'Replies', 'Retweets'])
        initial_counts.extend([stats['split']['total'], stats['split']['total'], 
                              stats['split']['total'], stats['split']['total']])
        final_counts.extend([stats['split']['total'], stats['split']['originals'],
                            stats['split']['replies'], stats['split']['retweets']])
        retention_rates.extend([100, 
                               stats['split']['originals']/stats['split']['total']*100,
                               stats['split']['replies']/stats['split']['total']*100,
                               stats['split']['retweets']/stats['split']['total']*100])
    
    # Cleaning data
    for data_type in ['tweets', 'replies', 'retweets']:
        if stats[data_type] and stats[data_type].get('retention_rate'):
            categories.append(f'Clean {data_type.title()}')
            initial_counts.append(stats[data_type]['initial_count'])
            final_counts.append(stats[data_type]['final_count'])
            retention_rates.append(stats[data_type]['retention_rate'])
    
    # Stock data
    for stock_type in ['tesla', 'sp500']:
        if stats[stock_type] and stats[stock_type].get('retention_rate'):
            categories.append(f'Clean {stock_type.upper()} Stock')
            initial_counts.append(stats[stock_type]['initial_count'])
            final_counts.append(stats[stock_type]['final_count'])
            retention_rates.append(stats[stock_type]['retention_rate'])
    
    x = np.arange(len(categories))
    width = 0.35
    
    # Create bars
    bars1 = ax.bar(x - width/2, initial_counts, width, label='Initial Count', alpha=0.8)
    bars2 = ax.bar(x + width/2, final_counts, width, label='Final Count', alpha=0.8)
    
    # Add retention rate as text on bars
    for i, (bar1, bar2, rate) in enumerate(zip(bars1, bars2, retention_rates)):
        height = max(bar1.get_height(), bar2.get_height())
        ax.text(i, height + max(initial_counts) * 0.02, f'{rate:.1f}%', 
                ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('data Processing Steps')
    ax.set_ylabel('Number of Records')
    ax.set_title('data Flow Through Cleaning Pipeline\n(Retention rates shown above bars)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_removal_reasons(stats):
    """Create subplots showing removal reasons for each data type"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('data Removal Breakdown by Type', fontsize=16, fontweight='bold')
    
    # Tweets removal reasons
    if stats['tweets']:
        ax = axes[0, 0]
        tweets_data = stats['tweets']
        reasons = ['Empty Text', 'Non-Alphanumeric', 'Duplicates', 'Retained']
        counts = [tweets_data.get('removed_empty', 0),
                 tweets_data.get('removed_non_alpha', 0),
                 tweets_data.get('removed_duplicates', 0),
                 tweets_data.get('final_count', 0)]
        
        colors = ['#ff9999', '#ffcc99', '#99ccff', '#99ff99']
        wedges, texts, autotexts = ax.pie(counts, labels=reasons, autopct='%1.1f%%', 
                                         colors=colors, startangle=90)
        ax.set_title('Tweets Cleaning Results')
    
    # Replies removal reasons
    if stats['replies']:
        ax = axes[0, 1]
        replies_data = stats['replies']
        reasons = ['Duplicates', 'Empty Text', 'Retained']
        counts = [replies_data.get('duplicates_removed', 0),
                 replies_data.get('empty_text_removed', 0),
                 replies_data.get('final_count', 0)]
        
        colors = ['#ffcc99', '#ff9999', '#99ff99']
        wedges, texts, autotexts = ax.pie(counts, labels=reasons, autopct='%1.1f%%', 
                                         colors=colors, startangle=90)
        ax.set_title('Replies Cleaning Results')
    
    # Retweets removal reasons
    if stats['retweets']:
        ax = axes[1, 0]
        retweets_data = stats['retweets']
        reasons = ['Empty Text', 'Unparseable', 'Duplicates', 'Non-Alpha', 'Too Short', 'Retained']
        counts = [retweets_data.get('removed_empty', 0),
                 retweets_data.get('removed_unparseable', 0),
                 retweets_data.get('removed_duplicates', 0),
                 retweets_data.get('removed_non_alpha', 0),
                 retweets_data.get('removed_short', 0),
                 retweets_data.get('final_count', 0)]
        
        colors = ['#ff9999', '#ff6666', '#ffcc99', '#99ccff', '#cc99ff', '#99ff99']
        wedges, texts, autotexts = ax.pie(counts, labels=reasons, autopct='%1.1f%%', 
                                         colors=colors, startangle=90)
        ax.set_title('Retweets Cleaning Results')
    
    # Stock data summary
    ax = axes[1, 1]
    stock_labels = []
    stock_retention = []
    
    if stats['tesla']:
        stock_labels.append('Tesla Stock')
        stock_retention.append(stats['tesla'].get('retention_rate', 0))
    
    if stats['sp500']:
        stock_labels.append('S&P 500 Stock')
        stock_retention.append(stats['sp500'].get('retention_rate', 0))
    
    if stock_labels:
        bars = ax.bar(stock_labels, stock_retention, color=['#ff6b6b', '#4ecdc4'])
        ax.set_ylabel('Retention Rate (%)')
        ax.set_title('Stock data Cleaning Results')
        ax.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, rate in zip(bars, stock_retention):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    return fig

def plot_summary_statistics(stats):
    """Create a summary table of all cleaning statistics"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare summary data
    summary_data = []
    
    # Split statistics
    if stats['split']:
        split_data = stats['split']
        summary_data.append(['data Splitting', split_data['total'],
                           split_data['total'], '100.0%', 'Split into categories'])
        summary_data.append(['├─ Tweets', split_data['total'], 
                           split_data['originals'], 
                           f"{split_data['originals']/split_data['total']*100:.1f}%", 'Original posts'])
        summary_data.append(['├─ Replies', split_data['total'], 
                           split_data['replies'], 
                           f"{split_data['replies']/split_data['total']*100:.1f}%", 'Reply posts'])
        summary_data.append(['└─ Retweets', split_data['total'], 
                           split_data['retweets'], 
                           f"{split_data['retweets']/split_data['total']*100:.1f}%", 'Retweeted posts'])
    
    # Cleaning statistics
    for data_type in ['tweets', 'retweets']: # removed 'replies'
        if stats[data_type]:
            data = stats[data_type]
            summary_data.append([f'{data_type.title()} Cleaning', 
                               data.get('initial_count', 0),
                               data.get('final_count', 0),
                               f"{data.get('retention_rate', 0):.1f}%",
                               'Text cleaning & deduplication'])
    
    # # Stock data
    # for stock_type in ['tesla', 'sp500']:
    #     if stats[stock_type]:
    #         data = stats[stock_type]
    #         summary_data.append([f'{stock_type.upper()} Stock Cleaning',
    #                            data.get('initial_count', 0),
    #                            data.get('final_count', 0),
    #                            f"{data.get('retention_rate', 0):.1f}%",
    #                            'data validation & formatting'])
    
    # Create table
    columns = ['Processing Step', 'Initial Count', 'Final Count', 'Retention Rate', 'Description']
    table = ax.table(cellText=summary_data, colLabels=columns, 
                    cellLoc='center', loc='center')
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Color code the header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color code retention rates
    for i, row in enumerate(summary_data):
        retention = float(row[3].strip('%'))
        if retention >= 95:
            color = '#E8F5E8'  # Light green
        elif retention >= 80:
            color = '#FFF3CD'  # Light yellow
        else:
            color = '#F8D7DA'  # Light red
        
        for j in range(len(columns)):
            table[(i+1, j)].set_facecolor(color)
    
    ax.set_title('data Cleaning Pipeline Summary', fontsize=16, fontweight='bold', pad=20)
    
    return fig

def generate_all_plots(stats):
    """Generate all visualization plots and save them"""
    output_dir = create_output_dir()
    
    print("Generating data cleaning visualizations...")
    
    # Generate plots
    plots = {
        # 'data_flow': plot_data_flow_bar_chart(stats),
        # 'removal_reasons': plot_removal_reasons(stats),
        'summary_statistics': plot_summary_statistics(stats)
    }
    
    # Save plots
    for plot_name, fig in plots.items():
        save_path = output_dir / f"{plot_name}.png"
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved {plot_name} to {save_path}")
        plt.close(fig)
    
    print(f"\n📊 All plots saved to: {output_dir.absolute()}")
    
    # # Also show the plots if running interactively
    # if __name__ == "__main__":
    #     plt.show()

# if __name__ == "__main__":
#     # Example usage with dummy data
#     dummy_stats = {
#         'split': {'total': 50000, 'originals': 15000, 'replies': 20000, 'retweets': 15000},
#         'tweets': {'initial_count': 15000, 'removed_empty': 100, 'removed_non_alpha': 200, 
#                   'removed_duplicates': 200, 'final_count': 14500, 'retention_rate': 96.7},
#         'replies': {'initial_count': 20000, 'duplicates_removed': 150, 'empty_text_removed': 50, 
#                    'final_count': 19800, 'retention_rate': 99.0},
#         'retweets': {'initial_count': 15000, 'removed_empty': 100, 'removed_unparseable': 1500,
#                     'removed_duplicates': 500, 'removed_non_alpha': 100, 'removed_short': 50,
#                     'final_count': 12750, 'retention_rate': 85.0},
#         'tesla': {'initial_count': 3000, 'removed_invalid_row': 1, 'final_count': 2999, 'retention_rate': 99.97},
#         'sp500': {'initial_count': 3500, 'removed_invalid': 5, 'final_count': 3495, 'retention_rate': 99.86}
#     }
    
#     generate_all_plots(dummy_stats)
