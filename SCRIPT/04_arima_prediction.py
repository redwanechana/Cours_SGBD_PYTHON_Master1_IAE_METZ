"""
04_arima_prediction.py
Predict stock prices using an ARIMA model.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import warnings

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "DATA")
TARGET_STOCK = "AAPL"
TRAIN_RATIO = 0.8


def arima_prediction():
    prices = pd.read_csv(os.path.join(DATA_DIR, "prices_clean.csv"), index_col="Date", parse_dates=True)

    series = prices[TARGET_STOCK]
    print(f"=== ARIMA Prediction for {TARGET_STOCK} ===\n")
    print(f"Total observations: {len(series)}")

    split_idx = int(len(series) * TRAIN_RATIO)
    train = series.iloc[:split_idx]
    test = series.iloc[split_idx:]

    print(f"Training set: {len(train)} days")
    print(f"Test set:     {len(test)} days\n")

    print("Fitting ARIMA(5,1,0)... ", end="")
    model = ARIMA(train, order=(5, 1, 0))
    model_fit = model.fit()
    print("Done.\n")

    print(f"AIC: {model_fit.aic:.2f}")
    print(f"BIC: {model_fit.bic:.2f}\n")

    forecast = model_fit.forecast(steps=len(test))
    forecast.index = test.index

    mae = mean_absolute_error(test, forecast)
    rmse = np.sqrt(mean_squared_error(test, forecast))
    mape = (np.abs((test - forecast) / test).mean()) * 100

    print("=== ARIMA Results ===")
    print(f"MAE:  {mae:.2f} $")
    print(f"RMSE: {rmse:.2f} $")
    print(f"MAPE: {mape:.2f} %\n")

    results = pd.DataFrame({"Actual": test, "ARIMA_Forecast": forecast})
    output_path = os.path.join(DATA_DIR, "arima_results.csv")
    results.to_csv(output_path)
    print(f"Results saved to {output_path}")

    metrics = pd.DataFrame({
        "Model": ["ARIMA(5,1,0)"],
        "MAE": [round(mae, 2)],
        "RMSE": [round(rmse, 2)],
        "MAPE": [round(mape, 2)]
    })
    metrics.to_csv(os.path.join(DATA_DIR, "arima_metrics.csv"), index=False)

    return results, metrics


if __name__ == "__main__":
    arima_prediction()
