import streamlit as st
import yfinance as yf
import datetime
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
import time
from requests.exceptions import RequestException
import langdetect

st.set_page_config(page_title="StockWise", layout="wide")

if "last_period" not in st.session_state:
    st.session_state["last_period"] = None

chatbot_sidebar = st.sidebar.container()


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


def get_news(ticker, max_retries=3, retry_delay=1):
    stock = yf.Ticker(ticker)
    for attempt in range(max_retries):
        try:
            news = stock.news
            if news:
                return news
            else:
                time.sleep(retry_delay)
                stock = yf.Ticker(ticker)  # Reinitialize the Ticker object
                continue
        except (RequestException, ValueError, AttributeError) as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                stock = yf.Ticker(ticker)  # Reinitialize the Ticker object
            else:
                st.warning(
                    f"Unable to fetch news for {ticker} after {max_retries} attempts."
                )
                return [
                    {
                        "title": f"News unavailable for {ticker}",
                        "link": "",
                        "providerPublishTime": time.time(),
                    }
                ]

    return [
        {
            "title": f"No news found for {ticker}",
            "link": "",
            "providerPublishTime": time.time(),
        }
    ]


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
    data = yf.download(ticker, period="max")

    n_splits = min(5, len(data) // 20)

    tss = TimeSeriesSplit(n_splits=n_splits)

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
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)

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


def display_price_and_refresh(data, ticker, key=None):
    current_price = get_realtime_price(ticker)
    if current_price is None:
        st.warning("Today is not a trading day or market is closed.")
        return False

    price_change = current_price - data["Close"].iloc[-1]
    price_change_percent = (price_change / data["Close"].iloc[-1]) * 100
    color = "green" if price_change >= 0 else "red"

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(
            f"<h2 style='text-align: center; color: {color};'>{current_price:.2f} USD ({price_change_percent:.2f}%)</h2>",
            unsafe_allow_html

