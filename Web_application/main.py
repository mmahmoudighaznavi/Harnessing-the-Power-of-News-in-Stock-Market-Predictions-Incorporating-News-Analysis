import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from pickle import load
from keras import models
from joblib import load as joblib_load
import streamlit as st
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')



st.set_page_config(page_title="Stock Prediction App", page_icon="📈")


# # Load your image
image = Image.open('../Images/stock_pred.jpg')
st.image(image, caption="Stock Prediction")


from datetime import datetime


def fetch_stock_data(ticker, start_date, end_date, interval):
    stock_data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    return stock_data


def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    average_gain = gain.rolling(window=period).mean()
    average_loss = loss.rolling(window=period).mean()

    rs = average_gain / average_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


sma_period=200

def get_indicator(aapl_data):
    #Simple Moving Average (SMA)
    aapl_data["SMA-200"] = aapl_data.iloc[:,4].rolling(sma_period).mean()
    
    # Exponential Moving Average (EMA)
    aapl_data["EMA-12"] = aapl_data.iloc[:,4].ewm(span=12, adjust=False).mean()
    aapl_data["EMA-26"] = aapl_data.iloc[:,4].ewm(span=26, adjust=False).mean()
    
   #Average Convergence Divergence (MACD)
    aapl_data['MACD'] = aapl_data["EMA-12"] - aapl_data["EMA-26"]
    aapl_data['MACD_Signal'] = aapl_data['MACD'].ewm(span=9, adjust=False).mean()
    
    aapl_data["RSI_14"] = calculate_rsi(aapl_data['Close'], 14)
    
    return aapl_data



# @st.cache_resource(allow_output_mutation=True)
def load_model(model_name):

    if model_name == 'GRU':
        model = models.load_model('../Models/gru_model.h5')
    elif model_name == "LSTM":
        model = models.load_model('../Models/lstm_model.h5')
    else:
        st.error("Invalid model selected")
        model = None
    return model 


@st.cache_resource()
def load_scalers():
    X_scaler = joblib_load('../Scaler/X_scaler.joblib')
    y_scaler = joblib_load('../Scaler/y_scaler.joblib')
    return X_scaler, y_scaler

    

# Model selection dropdown
model_selection = st.selectbox("Select Model:", options=["GRU", "LSTM"])
model = load_model(model_selection)


# Number of days selection
n_days = st.number_input('Enter number of days to predict:', min_value=1, max_value=30, value=1)

feature_columns = ['Open', 'High', 'Low', 'Volume', 'SMA-200', 'EMA-12', 'EMA-26', 'MACD', 'MACD_Signal', 'RSI_14']

n_steps = 10


def create_windowed_data(data, n_steps):
    X, y = list(), list()
    for i in range(len(data) - n_steps):
        seq_x, seq_y = data[i:i+n_steps][feature_columns], data.iloc[i+n_steps]['Close']
        X.append(seq_x.values)
        y.append(seq_y)
    return np.array(X), np.array(y)


# Function to make predictions for the next days
def predict_next_days(model, X_scaler, y_scaler, data, n_steps, n_features, n_days):

    predictions = []
    # feature_columns = ['Open', 'High', 'Low', 'Volume', 'SMA-200', 'EMA-12', 'EMA-26', 'MACD', 'MACD_Signal', 'RSI_14']
    
    for _ in range(n_days):
        # Select the last n_steps of data for all features
        last_steps_data = data[feature_columns].tail(n_steps)

        # Flatten the data and reshape for the scaler
        last_steps_values = last_steps_data.values.flatten().reshape(1, n_steps * n_features)

        # Scale the flattened data
        last_steps_scaled = X_scaler.transform(last_steps_values)

        # Reshape the scaled data to the 3D format for the model: (1, n_steps, n_features)
        X_input = last_steps_scaled.reshape(1, n_steps, n_features)

        # Predict and inverse transform the prediction
        y_pred_scaled = model.predict(X_input)
        y_pred = y_scaler.inverse_transform(y_pred_scaled)[0, 0]

        predictions.append(y_pred)

        # Append the predicted value to the data for subsequent predictions
        predicted_row = data.tail(1).copy()
        predicted_row.iloc[0, data.columns.get_loc('Close')] = y_pred  # Update the Close value with the prediction
        data = pd.concat([data, predicted_row], ignore_index=True)

    return predictions


# Add function to categorize sentiment
def categorize_sentiment(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'


# Function to fetch news
def fetch_news(ticker):
    finviz_url = 'https://finviz.com/quote.ashx?t='
    url = finviz_url + ticker
    req = Request(url=url, headers={'user-agent': 'my-app'})
    response = urlopen(req)
    html = BeautifulSoup(response, 'html')
    news_table = html.find(id='news-table')
    parsed_data = []
    if news_table:
        for row in news_table.findAll('tr'):
            title = row.a.get_text(strip=True) if row.a else None
            if title:
                date_data = row.td.get_text(strip=True).split(' ') if row.td else ["N/A"]
                date = date_data[0] if len(date_data) > 1 else ""
                time = date_data[1] if len(date_data) > 1 else date_data[0]
                parsed_data.append([ticker, date, time, title])

    df = pd.DataFrame(parsed_data, columns = ["Ticker", "Date", "Time", "Content"])

    # apply sentiment analysis
    vader = SentimentIntensityAnalyzer()
    df['Sentiment'] = df['Content'].apply(lambda x: vader.polarity_scores(x)['compound'])
    df['Sentiment Label'] = df['Sentiment'].apply(categorize_sentiment)
    
    return df
    

# Sidebar for user input
with st.sidebar:
    st.title('Stock Prediction 📈')
    st.title("Harnessing the Power of Stock Analysis using Financial News and Deep Learning Models")
    ticker = st.text_input("Enter Stock Symbol:", value="AAPL").upper()
    start_date = st.date_input("Start Date:", value=pd.to_datetime("2012-01-01"))
    end_date = st.date_input("End Date:", value=pd.to_datetime(datetime.now().strftime('%Y-%m-%d')))
    interval = st.selectbox("Select Interval:", options=["1d", "1wk", "1mo"])



# Main app
if 'stock_data' not in st.session_state or st.button("Fetch and Predict"):
    # Fetch the stock data
    st.session_state.data = fetch_stock_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), interval)
    st.write(f"Stock Data for {ticker} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    st.dataframe(st.session_state.data)


########################### Data Preprocessing ################################

    data_with_indicators = get_indicator(st.session_state.data)


    data_with_indicators.dropna(subset=['SMA-200', 'EMA-12', 'EMA-26', 'MACD', 'MACD_Signal', 'RSI_14'], inplace=True)
    
    
    if 'Date' in data_with_indicators.columns:
        data_with_indicators.drop(columns=['Date'], inplace=True)

    if 'Adj Close' in data_with_indicators.columns:
        data_with_indicators.drop(columns=['Adj Close'], inplace=True)


    feature_columns = [col for col in data_with_indicators.columns if col != 'Close']

    X, y = create_windowed_data(data_with_indicators, n_steps)

    test_size = 0.2
    split_point = int(len(X) * (1 - test_size))

    X_train, X_test = X[:split_point], X[split_point:]
    y_train, y_test = y[:split_point], y[split_point:]

    # Load the selected model and scalers
    model = load_model(model_selection)
    X_scaler, y_scaler = load_scalers()



    if model:

        # Predict the next days
        n_steps = 10
        n_features = 10

        X_test_flattened = X_test.reshape(-1, n_steps * n_features)

        X_test_scaled = X_scaler.transform(X_test_flattened)

        X_test_scaled = X_test_scaled.reshape(-1, n_steps, n_features)

        y_pred_scaled = model.predict(X_test_scaled)

        y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()


        predictions = predict_next_days(model, X_scaler, y_scaler, data_with_indicators, n_steps, n_features, n_days)

        # Generate future dates for plotting predictions
        prediction_dates = pd.date_range(start=data_with_indicators.index[-1] + pd.Timedelta(days=1), periods=n_days)

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=y_test, mode='lines', name='Actual Stock Price'))
        fig.add_trace(go.Scatter(y=y_pred, mode='lines', name='Predicted Stock Price', line=dict(color='red', dash='dash')))
        

        fig.update_layout(title=f'Stock Price Prediction for {ticker}', xaxis_title='Date', yaxis_title='Price', legend_title='Legend')
        st.plotly_chart(fig)

        for date, pred in zip(prediction_dates, predictions):
            st.write(f"Predicted price for {date.date()} : ${pred:.2f}")


        ############ fetching news ###############

        # Add a section for fetching news
        st.write("## Stock News")
        if st.button("Fetch News"):
            news_df = fetch_news(ticker)
            if not news_df.empty:
                st.dataframe(news_df[['Time', 'Content', 'Sentiment', 'Sentiment Label']])
            else:
                st.write("No news available for the given ticker.")





