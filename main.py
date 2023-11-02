import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.express as px
from datetime import datetime


def fetch_stock_data(ticker, start_date, end_date, interval):
    """
    Fetch stock data for a given ticker, start date, and end date.
    
    :param ticker: Stock ticker symbol (e.g., 'AAPL' for Apple Inc.)
    :param start_date: Start date in the format 'YYYY-MM-DD'.
    :param end_date: End date in the format 'YYYY-MM-DD'.
    
    :return: DataFrame containing stock data for the given ticker and dates.
    """
    stock_data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    return stock_data


st.set_page_config(page_title="Stock Prediction App", page_icon="📈")

ticker = st.text_input("Enter Stock Symbol:", value="AAPL").upper()
start_date = st.date_input("Start Date:", value=pd.to_datetime("2012-01-01"))
end_date = st.date_input("End Date:", value=pd.to_datetime(datetime.now().strftime('%Y-%m-%d')))
interval = st.selectbox("Select Interval:", options=["1d", "1wk", "1mo"])


if st.button("Fetch Data"):
    data = fetch_stock_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), interval)
    st.write(f"Stock Data for {ticker} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    st.dataframe(data)
