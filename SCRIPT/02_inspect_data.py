"""
02_inspect_data.py
Inspect and clean the downloaded price data.
"""

import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "DATA")


def inspect_and_clean():
    prices = pd.read_csv(os.path.join(DATA_DIR, "prices.csv"), index_col="Date", parse_dates=True)

    print("=== Raw Data Inspection ===\n")
    print(f"Shape: {prices.shape}")
    print(f"\nFirst 5 rows:\n{prices.head()}\n")
    print(f"Missing values per column:\n{prices.isnull().sum()}\n")
    print(f"Data types:\n{prices.dtypes}\n")
    print(f"Basic statistics:\n{prices.describe().round(2)}\n")

    prices_clean = prices.ffill().bfill()

    missing_after = prices_clean.isnull().sum().sum()
    print(f"Missing values after cleaning: {missing_after}")

    output_path = os.path.join(DATA_DIR, "prices_clean.csv")
    prices_clean.to_csv(output_path)
    print(f"Cleaned data saved to {output_path}")

    return prices_clean


if __name__ == "__main__":
    inspect_and_clean()
