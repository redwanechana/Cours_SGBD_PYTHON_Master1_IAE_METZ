"""
main.py
Run the entire pipeline in one command.
"""

print("=" * 60)
print("  STOCK PRICE PREDICTION - ARIMA vs LSTM")
print("  Master 1 Finance - IAE Metz 2026")
print("=" * 60)

from importlib import import_module

print("\n[1/6] Downloading data from Yahoo Finance...")
import_module("01_download_data").download_prices()

print("\n[2/6] Inspecting and cleaning data...")
import_module("02_inspect_data").inspect_and_clean()

print("\n[3/6] Computing daily returns...")
import_module("03_compute_returns").compute_returns()

print("\n[4/6] Running ARIMA prediction...")
import_module("04_arima_prediction").arima_prediction()

print("\n[5/6] Running LSTM prediction...")
import_module("05_lstm_prediction").lstm_prediction()

print("\n[6/6] Generating visualizations...")
import_module("06_visualizations")

print("\n" + "=" * 60)
print("  PIPELINE COMPLETE - All files saved in DATA/")
print("=" * 60)
