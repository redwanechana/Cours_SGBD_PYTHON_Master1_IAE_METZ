"""
05_lstm_prediction.py
Predict stock prices using a Long Short-Term Memory (LSTM) neural network.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "DATA")
TARGET_STOCK = "AAPL"
TRAIN_RATIO = 0.8
LOOK_BACK = 60
EPOCHS = 50
BATCH_SIZE = 32


def create_sequences(data, look_back):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i - look_back:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def lstm_prediction():
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    prices = pd.read_csv(os.path.join(DATA_DIR, "prices_clean.csv"), index_col="Date", parse_dates=True)
    series = prices[[TARGET_STOCK]].values
    dates = prices.index

    print(f"=== LSTM Prediction for {TARGET_STOCK} ===\n")
    print(f"Total observations: {len(series)}")
    print(f"Look-back window: {LOOK_BACK} days")
    print(f"Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}\n")

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(series)

    split_idx = int(len(scaled_data) * TRAIN_RATIO)
    train_data = scaled_data[:split_idx]
    test_data = scaled_data[split_idx - LOOK_BACK:]

    X_train, y_train = create_sequences(train_data, LOOK_BACK)
    X_test, y_test = create_sequences(test_data, LOOK_BACK)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples:     {X_test.shape[0]}\n")

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(LOOK_BACK, 1)),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mean_squared_error")
    model.summary()

    print("\nTraining LSTM model...")
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1, verbose=1)

    predictions_scaled = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions_scaled)
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    mae = mean_absolute_error(actual, predictions)
    rmse = np.sqrt(mean_squared_error(actual, predictions))
    mape = (np.abs((actual - predictions) / actual).mean()) * 100

    print(f"\n=== LSTM Results ===")
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
        "Model": ["LSTM"],
        "MAE": [round(mae, 2)],
        "RMSE": [round(rmse, 2)],
        "MAPE": [round(mape, 2)]
    })
    metrics.to_csv(os.path.join(DATA_DIR, "lstm_metrics.csv"), index=False)

    return results, metrics


if __name__ == "__main__":
    lstm_prediction()
