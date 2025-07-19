import os
import pandas as pd


def clean_stock_data():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir))  # goes up from cleaners/ to project root
    input_path = os.path.join(project_root, 'data', 'original', 'tesla_stock_data_2000_2025.csv')
    output_dir = os.path.join(project_root, 'data', 'clean')

    raw_stock = pd.read_csv(input_path, header=0)
    clean_stock = raw_stock.drop(index=[0, 1]).reset_index(drop=True)

    clean_stock['Date'] = pd.to_datetime(clean_stock['Price']).dt.date

    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        clean_stock[col] = pd.to_numeric(clean_stock[col], errors='coerce')

    clean_stock = clean_stock[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'clean_stock.csv')
    clean_stock.to_csv(output_path, index=False)

    print(f"Cleaned stock data saved to: {output_path}")


if __name__ == "__main__":
    clean_stock_data()