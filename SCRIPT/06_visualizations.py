"""
06_visualizations.py
Generate all charts for the project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "DATA")

plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["figure.figsize"] = (14, 6)
plt.rcParams["figure.dpi"] = 150


def plot_prices():
    prices = pd.read_csv(os.path.join(DATA_DIR, "prices_clean.csv"), index_col="Date", parse_dates=True)
    fig, ax = plt.subplots()
    for col in prices.columns:
        ax.plot(prices.index, prices[col], label=col, linewidth=1.2)
    ax.set_title("Historical Stock Prices (2019-2024)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "chart_01_prices.png"))
    plt.close()
    print("Chart 1 saved: chart_01_prices.png")


def plot_returns():
    returns = pd.read_csv(os.path.join(DATA_DIR, "returns.csv"), index_col="Date", parse_dates=True)
    fig, axes = plt.subplots(1, len(returns.columns), figsize=(18, 5))
    for i, col in enumerate(returns.columns):
        axes[i].hist(returns[col], bins=50, alpha=0.7, color=f"C{i}", edgecolor="black", linewidth=0.5)
        axes[i].set_title(col, fontweight="bold")
        axes[i].set_xlabel("Daily Return")
        axes[i].axvline(x=0, color="red", linestyle="--", linewidth=0.8)
    fig.suptitle("Distribution of Daily Returns", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "chart_02_returns_distribution.png"))
    plt.close()
    print("Chart 2 saved: chart_02_returns_distribution.png")


def plot_arima_forecast():
    results = pd.read_csv(os.path.join(DATA_DIR, "arima_results.csv"), index_col="Date", parse_dates=True)
    fig, ax = plt.subplots()
    ax.plot(results.index, results["Actual"], label="Actual", linewidth=1.5, color="black")
    ax.plot(results.index, results["ARIMA_Forecast"], label="ARIMA Forecast", linewidth=1.2, color="red", linestyle="--")
    ax.set_title("ARIMA(5,1,0) - Forecast vs Actual (AAPL)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "chart_03_arima_forecast.png"))
    plt.close()
    print("Chart 3 saved: chart_03_arima_forecast.png")


def plot_lstm_forecast():
    results = pd.read_csv(os.path.join(DATA_DIR, "lstm_results.csv"), index_col="Date", parse_dates=True)
    fig, ax = plt.subplots()
    ax.plot(results.index, results["Actual"], label="Actual", linewidth=1.5, color="black")
    ax.plot(results.index, results["LSTM_Forecast"], label="LSTM Forecast", linewidth=1.2, color="blue", linestyle="--")
    ax.set_title("LSTM - Forecast vs Actual (AAPL)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "chart_04_lstm_forecast.png"))
    plt.close()
    print("Chart 4 saved: chart_04_lstm_forecast.png")


def plot_model_comparison():
    arima = pd.read_csv(os.path.join(DATA_DIR, "arima_metrics.csv"))
    lstm = pd.read_csv(os.path.join(DATA_DIR, "lstm_metrics.csv"))
    metrics = pd.concat([arima, lstm], ignore_index=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metric_names = ["MAE", "RMSE", "MAPE"]
    colors = ["#e74c3c", "#3498db"]

    for i, metric in enumerate(metric_names):
        bars = axes[i].bar(metrics["Model"], metrics[metric], color=colors, edgecolor="black", linewidth=0.5)
        axes[i].set_title(metric, fontsize=13, fontweight="bold")
        axes[i].set_ylabel(f"{metric} ({'$' if metric != 'MAPE' else '%'})")
        for bar, val in zip(bars, metrics[metric]):
            axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                         f"{val:.2f}", ha="center", fontweight="bold")

    fig.suptitle("Model Comparison: ARIMA vs LSTM", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "chart_05_model_comparison.png"))
    plt.close()
    print("Chart 5 saved: chart_05_model_comparison.png")


if __name__ == "__main__":
    plot_prices()
    plot_returns()
    plot_arima_forecast()
    plot_lstm_forecast()
    plot_model_comparison()
    print("\nAll charts generated successfully.")
