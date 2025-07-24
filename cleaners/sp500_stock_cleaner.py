# from pathlib import Path
# import pandas as pd
# import yfinance as yf
# import os
#
# ROOT = Path(__file__).resolve().parents[1]   # project root
# out  = ROOT / "Data" / "original" / "sp500_index.csv"
# out.parent.mkdir(parents=True, exist_ok=True)
#
# df = yf.download("^GSPC", start="2010-01-01", progress=False, auto_adjust=False)
# df.reset_index().to_csv(out, index=False)
#
# print("cwd:", os.getcwd())
# print("saved to:", out.resolve())
# print("exists?", out.exists())

# cleaners/sp500_index_cleaner.py
from __future__ import annotations
from pathlib import Path
import pandas as pd

def clean_sp500() -> None:
    """
    Clean the downloaded S&P 500 (^GSPC) CSV to a lean schema.

    Why we keep only these columns:
    - Avoid redundancy (e.g., Adj Close ≈ Close).
    - Keep exactly one OHLCV set for simple merging and to reduce multicollinearity.
    - Smaller, cleaner file → faster downstream processing.

    Output: Data/clean/clean_sp500_stock.csv with columns:
        date, open, high, low, close, volume
    """
    root = Path(__file__).resolve().parents[1]
    raw  = root / "Data" / "original" / "sp500_index.csv"
    outd = root / "Data" / "clean"; outd.mkdir(parents=True, exist_ok=True)
    out  = outd / "clean_sp500_stock.csv"

    df = pd.read_csv(raw)
    # yfinance columns: Date, Open, High, Low, Close, Adj Close, Volume
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Close"]).sort_values("Date")

    keep = ["Date", "Open", "High", "Low", "Close", "Volume"]
    df = df[keep].rename(columns=str.lower)

    df.to_csv(out, index=False)
    print(f"✅ Clean rows: {len(df)}  → {out.resolve()}")

if __name__ == "__main__":
    clean_sp500()
