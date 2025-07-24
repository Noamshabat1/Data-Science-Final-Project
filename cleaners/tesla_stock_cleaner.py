import os
import pandas as pd


def clean_stock_data() -> None:
    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
    input_path = os.path.join(project_root, "data", "original", "tesla_stock_data_2000_2025.csv")
    output_dir = os.path.join(project_root, "data", "clean")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "clean_tesla_stock.csv")

    # Read, drop the two non‑data rows
    raw_stock = pd.read_csv(input_path, header=0)
    clean_stock = raw_stock.drop(index=[0, 1]).reset_index(drop=True)

    # Rename columns so Date is correct
    clean_stock.rename(columns={"Price": "Date", "Close": "Price"}, inplace=True)

    # Parse Date
    clean_stock["Date"] = pd.to_datetime(clean_stock["Date"]).dt.date

    # Convert numeric columns
    for col in ["Open", "High", "Low", "Price", "Volume"]:
        clean_stock[col] = pd.to_numeric(clean_stock[col], errors="coerce")

    # Reorder, sort, save
    clean_stock = (clean_stock[["Date", "Open", "High", "Low", "Price", "Volume"]]
                   .sort_values("Date")
                   .reset_index(drop=True))
    clean_stock.to_csv(output_path, index=False)

    # Preview first 3 rows
    preview = clean_stock.head(3).to_string(index=False)

    print(f"✅  Cleaned stock data saved to: {output_path}\n"
          f"--- preview (top 3 rows) ---\n{preview}")


if __name__ == "__main__":
    clean_stock_data()
