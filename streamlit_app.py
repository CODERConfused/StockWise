import streamlit as st
import yfinance as yf
import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objs as go
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
import time

st.set_page_config(page_title="StockWise", layout="wide")

if "last_period" not in st.session_state:
    st.session_state["last_period"] = None


def search_company(query):
    try:
        ticker = yf.Ticker(query)
        if ticker.info and "symbol" in ticker.info:
            return ticker.info["symbol"]
        search = yf.Ticker(query).search()
        if search:
            return search[0]["symbol"]
    except:
        pass
    try:
        search = yf.Ticker(query + ".").info
        if "symbol" in search:
            return search["symbol"]
    except:
        pass
    return None


def get_stock_data(ticker, period):
    try:
        data = yf.download(ticker, period=period)
        if data.empty:
            st.error(f"The ticker '{ticker}' does not exist or has no data.")
            return None
        return data
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None


def get_news(ticker):
    stock = yf.Ticker(ticker)
    news = stock.news
    return news


def get_stock_info(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        "Description": info.get("longBusinessSummary", "No description available"),
        "Market Cap": info.get("marketCap", "N/A"),
        "EPS": info.get("trailingEps", "N/A"),
        "P/E Ratio": info.get("trailingPE", "N/A"),
        "Dividend Yield": info.get("dividendYield", "N/A"),
        "Analyst Rating": info.get("recommendationKey", "N/A"),
        "52 Week High": info.get("fiftyTwoWeekHigh", "N/A"),
        "52 Week Low": info.get("fiftyTwoWeekLow", "N/A"),
        "Volume": info.get("volume", "N/A"),
    }


def predict_stock(ticker):
    tss = TimeSeriesSplit(n_splits=300)
    data = yf.download(ticker, period="max")

    def split_fit(data, model, predictors):
        for _ in tss.split(data):
            model.fit(data[predictors], data["Close"])

    def pred(data, model, predictors):
        preds = model.predict(data[predictors])
        preds = pd.Series(preds, index=data.index)
        return preds

    def model(data, model_instance, predictors, target_col, future_days):
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)

        split_fit(data, model_instance, predictors)

        last_date = data.index[-1]
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1), periods=future_days
        )
        future_data = pd.DataFrame(index=future_dates, columns=data.columns)

        for predictor in predictors:
            if predictor in ["Close Diff", "Daily Return", "Price Dir"]:
                future_data[predictor] = 0  # Initialize with neutral values
            else:
                future_data[predictor] = data[predictor].iloc[-1]

        future_preds = []
        for i in range(future_days):
            day_pred = pred(future_data.iloc[i : i + 1], model_instance, predictors)
            future_preds.append(day_pred[0])
            if i < future_days - 1:
                future_data.loc[future_dates[i + 1], "Close"] = day_pred[0]
                future_data.loc[future_dates[i + 1], "Close Diff"] = (
                    day_pred[0] - future_data.loc[future_dates[i], "Close"]
                )
                future_data.loc[future_dates[i + 1], "Daily Return"] = (
                    future_data.loc[future_dates[i + 1], "Close Diff"]
                    / future_data.loc[future_dates[i], "Close"]
                )
                future_data.loc[future_dates[i + 1], "Price Dir"] = (
                    1 if future_data.loc[future_dates[i + 1], "Close Diff"] > 0 else 0
                )

        return pd.Series(future_preds)

    def make_features(data):
        data["Tomorrow Close"] = data["Close"].shift(-1)
        data["Close Diff"] = data["Tomorrow Close"] - data["Close"]
        data["Daily Return"] = data["Close Diff"] / data["Close"]
        data["Price Dir"] = [
            1 if data["Close Diff"].loc[ei] > 0 else 0 for ei in data.index
        ]
        data["MA10"] = data["Close"].rolling(10).mean()
        data["MA50"] = data["Close"].rolling(50).mean()
        data["MA100"] = data["Close"].rolling(100).mean()
        return data.dropna()

    data = make_features(data)

    predictors = [
        "Open",
        "High",
        "Low",
        "Volume",
        "Close Diff",
        "Daily Return",
        "Price Dir",
    ]
    xgb = XGBRegressor()

    future_predictions = model(data, xgb, predictors, "Close", future_days=5)

    return future_predictions


def get_realtime_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        todays_data = stock.history(period="1d")
        return todays_data["Close"].iloc[-1]
    except IndexError:
        st.error("Cannot find data on this company.")
        return None


def get_intraday_data(ticker):
    end_time = datetime.datetime.now()
    start_time = end_time.replace(hour=0, minute=0, second=0, microsecond=0)
    data = yf.download(ticker, start=start_time, end=end_time, interval="1m")
    return data


def display_price_and_refresh(data, ticker):
    current_price = get_realtime_price(ticker)
    if current_price is None:
        return False
    price_change = current_price - data["Close"].iloc[-1]
    price_change_percent = (price_change / data["Close"].iloc[-1]) * 100
    color = "green" if price_change >= 0 else "red"

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(
            f"<h2 style='text-align: center; color: {color};'>{current_price:.2f} USD ({price_change_percent:.2f}%)</h2>",
            unsafe_allow_html=True,
        )
    with col2:
        st.write("\n")
        return st.button("Refresh Price")


def update_intraday_graph(ticker):
    data = get_intraday_data(ticker)
    fig = go.Figure()

    performance = data["Close"].iloc[-1] > data["Open"].iloc[0]
    line_color = "lime" if performance else "red"

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Close"],
            mode="lines",
            name="Price",
            line=dict(color=line_color),
        )
    )

    fig.update_layout(
        title=f"{ticker} Intraday Price",
        xaxis_title="Time",
        yaxis_title="Price",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        yaxis=dict(gridcolor="gray"),
        xaxis=dict(gridcolor="gray"),
    )
    return fig


st.title("Stock Data, News, and Prediction Viewer")

query = st.text_input("Enter a stock ticker or company name (e.g., AAPL, Apple):")
ticker = search_company(query)

data = None
if ticker:
    st.write(f"Found ticker: {ticker}")
    data = get_stock_data(ticker, "max")
    if data is None:
        st.stop()
else:
    st.error("No matching company found. Please try a different name or ticker.")
    st.stop()

if data is not None:
    current_price = get_realtime_price(ticker)
    if current_price is None:
        st.stop()
    yesterday_close = data["Close"].iloc[-2]
    percent_change = ((current_price - yesterday_close) / yesterday_close) * 100
    change_color = "green" if percent_change > 0 else "red"

    stock_info = get_stock_info(ticker)
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader(f"About {ticker}")
    with col2:
        st.markdown(
            f"<div style='background-color: {change_color}; color: white; padding: 10px; border-radius: 5px; display: inline-block;'>Change: {percent_change:.2f}%</div>",
            unsafe_allow_html=True,
        )
    st.write("---")
    st.write(stock_info["Description"])

    st.subheader(f"Key Statistics for {ticker}")
    col1, col2 = st.columns(2)
    for i, (key, value) in enumerate(stock_info.items()):
        if key != "Description":
            if i % 2 == 0:
                col1.metric(key, value)
            else:
                col2.metric(key, value)

st.write("---")
st.subheader("Graph and News")

col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("1 Day"):
        st.session_state["last_period"] = "1d"
with col2:
    if st.button("1 Week"):
        st.session_state["last_period"] = "1wk"
with col3:
    if st.button("1 Year"):
        st.session_state["last_period"] = "1y"
with col4:
    if st.button("All Time"):
        st.session_state["last_period"] = "max"


def display_graph(data, ticker, period):
    fig = go.Figure()
    performance = data["Close"].iloc[-1] > data["Close"].iloc[0]
    line_color = "lime" if performance else "red"
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Close"],
            name="Historical",
            line=dict(color=line_color),
        )
    )

    if period == "1d":
        fig.add_trace(
            go.Scatter(
                x=[pd.Timestamp.now()],
                y=[get_realtime_price(ticker)],
                mode="markers",
                marker=dict(color=line_color, size=10),
                name="Current Price",
            )
        )

    fig.update_layout(
        title=f"{ticker} Stock Price",
        xaxis_title="Date",
        yaxis_title="Price",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        yaxis=dict(gridcolor="gray"),
        xaxis=dict(gridcolor="gray"),
    )
    st.plotly_chart(fig, use_container_width=True)


if st.session_state["last_period"]:
    refresh_clicked = display_price_and_refresh(data, ticker)
    if st.session_state["last_period"] == "1d":
        graph_placeholder = st.empty()

        while True:
            fig = update_intraday_graph(ticker)
            graph_placeholder.plotly_chart(fig, use_container_width=True)
            time.sleep(60)  # Update every minute
    else:
        data = get_stock_data(ticker, st.session_state["last_period"])
        if data is not None and len(data) >= 2:
            refresh_clicked = display_price_and_refresh(data, ticker)
            display_graph(data, ticker, st.session_state["last_period"])

            if refresh_clicked:
                st.experimental_rerun()

    news = get_news(ticker)
    st.subheader(f"Recent News for {ticker}")
    for article in news[:5]:
        st.write(f"**{article['title']}**")
        st.write(
            f"Published on: {datetime.datetime.fromtimestamp(article['providerPublishTime'])}"
        )
        st.write(article["link"])
        st.write("---")

st.write("---")
st.subheader("Financial Statements")
financials_button = st.button("Show Financial Statements")

if financials_button:
    stock = yf.Ticker(ticker)
    st.subheader("Income Statement")
    st.dataframe(stock.financials)
    st.subheader("Balance Sheet")
    st.dataframe(stock.balance_sheet)
    st.subheader("Cash Flow")
    st.dataframe(stock.cashflow)

st.write("---")
st.subheader("Stock Price Forecast")
predict_button = st.button("Predict Next 5 Days")

if predict_button:
    st.write(f"Predicting next 5 days for {ticker}")
    with st.spinner("Making predictions..."):
        predictions = predict_stock(ticker)
    if predictions is not None:
        st.success("Predictions complete!")

        future_dates = pd.date_range(
            start=data.index[-1] + pd.Timedelta(days=1), periods=5
        )
        future_df = pd.DataFrame(
            index=future_dates, data=predictions, columns=["Predicted Close"]
        )

        combined_df = pd.concat([data["Close"], future_df["Predicted Close"]])

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["Close"],
                name="Historical",
                line=dict(
                    color=(
                        "green"
                        if data["Close"].iloc[-1] > data["Close"].iloc[0]
                        else "red"
                    )
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=predictions,
                name="Prediction",
                line=dict(color="blue"),
            )
        )
        fig.update_layout(
            title=f"{ticker} Stock Price Prediction",
            xaxis_title="Date",
            yaxis_title="Price",
        )
        st.plotly_chart(fig, use_container_width=True)
