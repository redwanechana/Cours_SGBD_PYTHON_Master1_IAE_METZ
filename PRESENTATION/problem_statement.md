# Problem Statement

## Stock Price Prediction on a US Tech Portfolio

### Master 1 Finance — IAE Metz — Academic Group Project

Redwane Chana, Walid Litouche

Team Wared Capital 🔥

---

## Context

Financial markets are inherently volatile and unpredictable. Investors and portfolio managers constantly seek reliable tools to anticipate short-term price movements and make informed decisions. With the rise of data science and machine learning, quantitative methods have become essential in modern finance.

One of the key challenges in financial analysis is **short-term stock price prediction**. While no model can perfectly forecast market behavior, comparing different approaches helps identify which methods capture market dynamics most effectively.

---

## Research Question

Can we accurately predict the short-term price of Apple (AAPL) stock using time series analysis, and how do a classical statistical model (ARIMA) and a neural network (MLPRegressor) compare in terms of prediction accuracy?

---

## Methodology

We compare two fundamentally different approaches:

1. **ARIMA (5,1,0)** — A classical autoregressive model that assumes linear relationships in the data. It uses the 5 previous values and first-order differencing to achieve stationarity.

2. **Neural Network (MLPRegressor)** — A multi-layer perceptron with 3 hidden layers (128, 64, 32 neurons) that can capture non-linear patterns in the data. It uses a 60-day look-back window.

Both models are trained on 80% of the data (January 2019 – mid-2023) and evaluated on the remaining 20% (mid-2023 – December 2024).

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **MAE** (Mean Absolute Error) | Average absolute difference between predicted and actual values |
| **RMSE** (Root Mean Squared Error) | Penalizes larger errors more heavily than MAE |
| **MAPE** (Mean Absolute Percentage Error) | Error expressed as a percentage of actual values |

---

## Expected Outcome

We expect the neural network to outperform ARIMA due to its ability to model non-linear relationships. However, we also acknowledge the inherent limitations of any predictive model applied to financial markets, where external factors (news, geopolitics, earnings reports) play a significant role.

---

## Data Source

All data is sourced from **Yahoo Finance** via the `yfinance` Python library. The dataset covers 5 major US stocks (AAPL, MSFT, GOOGL, AMZN, TSLA) from January 2019 to December 2024, representing 1,509 trading days.

---

*Master 1 Finance — Academic Project — IAE Metz 2026*
