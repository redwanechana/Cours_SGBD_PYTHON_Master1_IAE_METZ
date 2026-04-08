"""
01_download_data.py
Download historical stock prices from Yahoo Finance.
"""

import yfinance as yf
import pandas as pd
import os

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
START_DATE = "2019-01-01"
END_DATE = "2024-12-31"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "DATA")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def download_prices():
    print(f"Downloading data for: {', '.join(TICKERS)}")
    print(f"Period: {START_DATE} to {END_DATE}\n")

    data = yf.download(TICKERS, start=START_DATE, end=END_DATE, auto_adjust=True)
    prices = data["Close"]
    prices.columns = TICKERS
    prices.index.name = "Date"

    output_path = os.path.join(OUTPUT_DIR, "prices.csv")
    prices.to_csv(output_path)
    print(f"Prices saved to {output_path}")
    print(f"Shape: {prices.shape[0]} rows x {prices.shape[1]} columns")

    return prices


if __name__ == "__main__":
    download_prices()
