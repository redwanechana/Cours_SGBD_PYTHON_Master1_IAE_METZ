"""
05_lstm_prediction.py
Predict stock prices using a Neural Network (MLPRegressor).
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import warnings

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "DATA")
TARGET_STOCK = "AAPL"
TRAIN_RATIO = 0.8
LOOK_BACK = 60


def create_sequences(data, look_back):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i - look_back:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def lstm_prediction():
    prices = pd.read_csv(os.path.join(DATA_DIR, "prices_clean.csv"), index_col="Date", parse_dates=True)
    series = prices[[TARGET_STOCK]].values
    dates = prices.index

    print(f"=== Neural Network Prediction for {TARGET_STOCK} ===\n")
    print(f"Total observations: {len(series)}")
    print(f"Look-back window: {LOOK_BACK} days\n")

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(series)

    split_idx = int(len(scaled_data) * TRAIN_RATIO)
    train_data = scaled_data[:split_idx]
    test_data = scaled_data[split_idx - LOOK_BACK:]

    X_train, y_train = create_sequences(train_data, LOOK_BACK)
    X_test, y_test = create_sequences(test_data, LOOK_BACK)

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples:     {X_test.shape[0]}\n")

    print("Training Neural Network model...")
    model = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    model.fit(X_train, y_train)
    print("Done.\n")

    predictions_scaled = model.predict(X_test).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions_scaled)
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    mae = mean_absolute_error(actual, predictions)
    rmse = np.sqrt(mean_squared_error(actual, predictions))
    mape = (np.abs((actual - predictions) / actual).mean()) * 100

    print(f"=== Neural Network Results ===")
    print(f"MAE:  {mae:.2f} $")
    print(f"RMSE: {rmse:.2f} $")
    print(f"MAPE: {mape:.2f} %\n")

    test_dates = dates[split_idx:split_idx + len(predictions)]
    results = pd.DataFrame({
        "Date": test_dates,
        "Actual": actual.flatten(),
        "LSTM_Forecast": predictions.flatten()
    })
    results.set_index("Date", inplace=True)

    output_path = os.path.join(DATA_DIR, "lstm_results.csv")
    results.to_csv(output_path)
    print(f"Results saved to {output_path}")

    metrics = pd.DataFrame({
        "Model": ["Neural Network"],
        "MAE": [round(mae, 2)],
        "RMSE": [round(rmse, 2)],
        "MAPE": [round(mape, 2)]
    })
    metrics.to_csv(os.path.join(DATA_DIR, "lstm_metrics.csv"), index=False)

    return results, metrics


if __name__ == "__main__":
    lstm_prediction()
