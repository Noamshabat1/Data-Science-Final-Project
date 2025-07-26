import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# --- Configuration ---
# Define the path to your final dataset
DATA_FILE = Path(__file__).resolve().parents[1] / "Data/model/model_data_full.csv"
# Define where to save the plot
OUTPUT_PLOT_FILE = Path(__file__).resolve().parent / "sentiment_vs_stock_plot.png"

# --- Main Plotting Function ---
def plot_sentiment_vs_stock(data_path: Path, output_path: Path):
    """
    Loads the final model data and plots tweet sentiment against the next day's stock price change.

    Args:
        data_path: Path to the model_data_full.csv file.
        output_path: Path to save the generated plot image.
    """
    if not data_path.exists():
        print(f"❌ Error: Data file not found at {data_path}")
        print("Please run the `model/build_full_model_data.py` script first.")
        return

    # 1. Load and prepare the data
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').set_index('date')

    # Select the relevant columns
    sentiment_col = 'sentiment_mean_VADER_tweet'
    stock_col = 'next_day_pct_change'

    if sentiment_col not in df.columns or stock_col not in df.columns:
        print(f"❌ Error: Required columns ('{sentiment_col}', '{stock_col}') not in the dataframe.")
        return

    # 2. Create the dual-axis plot
    fig, ax1 = plt.subplots(figsize=(18, 8))

    # Plot Stock Price Change on the left axis
    color1 = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Next Day TSLA % Change', color=color1, fontsize=12)
    ax1.plot(df.index, df[stock_col], color=color1, alpha=0.7, label='Stock % Change')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.axhline(0, color='grey', linestyle='--', linewidth=0.8) # Add a zero line for reference

    # Create the second axis for Sentiment
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Mean Tweet Sentiment (VADER)', color=color2, fontsize=12)
    ax2.plot(df.index, df[sentiment_col], color=color2, alpha=0.7, linestyle='--', label='Tweet Sentiment')
    ax2.tick_params(axis='y', labelcolor=color2)

    # 3. Enhance and Save
    plt.title("Elon Musk's Daily Tweet Sentiment vs. Next Day Tesla Stock Price Change", fontsize=16, pad=20)
    fig.tight_layout() # Adjust layout to make room for labels

    # Formatting the date axis
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.xaxis.set_minor_locator(mdates.MonthLocator(bymonthday=1, interval=3))

    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # Save the plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved successfully to: {output_path}")

# --- Execution ---
if __name__ == "__main__":
    plot_sentiment_vs_stock(DATA_FILE, OUTPUT_PLOT_FILE)
