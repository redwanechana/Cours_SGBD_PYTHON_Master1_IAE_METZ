"""
app.py
Streamlit dashboard for interactive visualization of prediction results.
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "DATA")

st.set_page_config(page_title="Stock Prediction Dashboard", layout="wide")
st.title("Stock Price Prediction - ARIMA vs LSTM")
st.markdown("**Master 1 Finance - IAE Metz 2026**")
st.markdown("---")

st.sidebar.header("Navigation")
page = st.sidebar.radio("Section", ["Overview", "ARIMA Results", "LSTM Results", "Model Comparison"])

@st.cache_data
def load_data():
    data = {}
    for f in ["prices_clean", "returns", "arima_results", "lstm_results", "arima_metrics", "lstm_metrics"]:
        path = os.path.join(DATA_DIR, f"{f}.csv")
        if os.path.exists(path):
            data[f] = pd.read_csv(path, index_col=0 if "metrics" not in f else None, parse_dates="metrics" not in f)
    return data

data = load_data()

if page == "Overview":
    st.header("Historical Prices")
    if "prices_clean" in data:
        st.line_chart(data["prices_clean"])
    st.header("Daily Returns")
    if "returns" in data:
        st.line_chart(data["returns"])

elif page == "ARIMA Results":
    st.header("ARIMA(5,1,0) - Forecast vs Actual")
    if "arima_results" in data:
        st.line_chart(data["arima_results"])
    if "arima_metrics" in data:
        st.subheader("Metrics")
        st.dataframe(data["arima_metrics"])

elif page == "LSTM Results":
    st.header("LSTM - Forecast vs Actual")
    if "lstm_results" in data:
        st.line_chart(data["lstm_results"])
    if "lstm_metrics" in data:
        st.subheader("Metrics")
        st.dataframe(data["lstm_metrics"])

elif page == "Model Comparison":
    st.header("ARIMA vs LSTM")
    if "arima_metrics" in data and "lstm_metrics" in data:
        comparison = pd.concat([data["arima_metrics"], data["lstm_metrics"]], ignore_index=True)
        st.dataframe(comparison)
        col1, col2, col3 = st.columns(3)
        for col, metric in zip([col1, col2, col3], ["MAE", "RMSE", "MAPE"]):
            col.metric(
                label=f"Best {metric}",
                value=f"{comparison[metric].min():.2f}",
                delta=f"{comparison.loc[comparison[metric].idxmin(), 'Model']}"
            )
