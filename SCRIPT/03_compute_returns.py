"""
03_compute_returns.py
Compute daily returns from cleaned price data.
"""

import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "DATA")


def compute_returns():
    prices = pd.read_csv(os.path.join(DATA_DIR, "prices_clean.csv"), index_col="Date", parse_dates=True)

    returns = prices.pct_change().dropna()

    print("=== Daily Returns ===\n")
    print(f"Shape: {returns.shape}")
    print(f"\nFirst 5 rows:\n{returns.head().round(4)}\n")
    print(f"Annualized mean return (approx):\n{(returns.mean() * 252).round(4)}\n")
    print(f"Annualized volatility:\n{(returns.std() * np.sqrt(252)).round(4)}\n")

    output_path = os.path.join(DATA_DIR, "returns.csv")
    returns.to_csv(output_path)
    print(f"Returns saved to {output_path}")

    return returns


if __name__ == "__main__":
    compute_returns()
