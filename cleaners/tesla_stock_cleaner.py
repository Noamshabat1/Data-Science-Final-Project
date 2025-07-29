import os
import pandas as pd


def clean_stock_data() -> None:
    input_path = "data/original/tesla_stock.csv"
    output_dir = "data/clean"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "clean_tesla_stock.csv")

    raw_stock = pd.read_csv(input_path, header=0)
    clean_stock = raw_stock.drop(index=[1]).reset_index(drop=True)

    clean_stock = clean_stock.drop('Adj Close', axis=1)

    clean_stock["Date"] = pd.to_datetime(clean_stock["Date"]).dt.date

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        clean_stock[col] = pd.to_numeric(clean_stock[col], errors="coerce")

    clean_stock = (clean_stock[["Date", "Open", "High", "Low", "Close", "Volume"]]
                   .sort_values("Date")
                   .reset_index(drop=True))
    clean_stock.to_csv(output_path, index=False)

    preview = clean_stock.head(3).to_string(index=False)

    print(f"SUCCESS  Cleaned stock data saved to: {output_path}\n"
          f"--- preview (top 3 rows) ---\n{preview}")


if __name__ == "__main__":
    clean_stock_data()
