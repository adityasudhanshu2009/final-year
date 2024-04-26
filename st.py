import requests
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go

# Function to fetch historical stock data from Yahoo Finance
def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data


def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def make_predictions(model, X_test):
    predicted_prices = model.predict(X_test)
    return predicted_prices

def fetch_current_price(ticker):
    # API endpoint for current stock price
    url = f'https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey=ZSiE6L1gYeihi92JTgM5fwrkoMLKSNgA'
    
    # Sending GET request to fetch data
    response = requests.get(url)
    
    # Check if request was successful (status code 200)
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        return data[0]['price']
    else:
        st.error("Failed to fetch current price data")
        return None


# Function to calculate indicators
def calculate_indicators(df):
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    average_gain = gain.rolling(window=14, min_periods=1).mean()
    average_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = average_gain / average_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    short_window = 12
    long_window = 26
    df['ShortEMA'] = df['Close'].ewm(span=short_window, adjust=False).mean()
    df['LongEMA'] = df['Close'].ewm(span=long_window, adjust=False).mean()
    df['MACD'] = df['ShortEMA'] - df['LongEMA']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    window = 20
    df['Rolling_Mean'] = df['Close'].rolling(window=window).mean()
    df['Upper_Band'] = df['Rolling_Mean'] + (2 * df['Close'].rolling(window=window).std())
    df['Lower_Band'] = df['Rolling_Mean'] - (2 * df['Close'].rolling(window=window).std())

    return df

# Function to analyze trend
def analyze_trend(df):
    bullish_score = 0
    bearish_score = 0

    # Weight for each indicator
    weights = {
        'SMA_50': 0.4,
        'EMA_20': 0.3,
        'RSI': 0.2,
        'MACD': 0.1
    }

    # Calculate bullish and bearish scores
    for indicator, weight in weights.items():
        if indicator in df.columns:
            if df[indicator].iloc[-1] > df['Close'].iloc[-1]:
                bullish_score += weight
            else:
                bearish_score += weight

    # Classify the trend based on scores
    if bullish_score > bearish_score:
        trend = 'Bullish'
    elif bearish_score > bullish_score:
        trend = 'Bearish'
    else:
        trend = 'Neutral'

    return trend, bullish_score, bearish_score

# Function to plot trend pie chart
def plot_trend_pie_chart(bullish_score, bearish_score):
    total_score = bullish_score + bearish_score
    bullish_percentage = bullish_score / total_score * 100
    bearish_percentage = bearish_score / total_score * 100

    labels = ['Bullish', 'Bearish']
    sizes = [bullish_percentage, bearish_percentage]
    colors = ['lightgreen', 'lightcoral']
    explode = (0.1, 0)  # explode 1st slice

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=140)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    return fig1

# Streamlit app
st.title('Stock Market Indicators Analysis')

# Sidebar for user input
st.sidebar.header('User Input')
ticker = st.sidebar.text_input('Enter Ticker Symbol (e.g., AAPL)', 'AAPL')
current_price = fetch_current_price(ticker)
if current_price:
    st.sidebar.subheader("Current Price")
    st.sidebar.write(f"${current_price:.2f}")

# Sidebar for date range selection
start_date = st.sidebar.date_input('Start Date', pd.to_datetime('2022-01-01'))
end_date = st.sidebar.date_input('End Date', pd.to_datetime('2022-12-31'))

# Fetch stock data
stock_data = fetch_stock_data(ticker, start_date, end_date)
st.write(stock_data.describe())
X = stock_data.drop(['Close'], axis=1)
y = stock_data['Close']

    # Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the machine learning model
model = train_model(X_train, y_train)

    # Make predictions using the trained model
predicted_prices = make_predictions(model, X_test)

# Calculate indicators
df = calculate_indicators(stock_data)

# Analyze trend
trend, bullish_score, bearish_score = analyze_trend(df)

# Plotting Closing Price
st.subheader('Closing Price and Simple Moving Average (SMA_50)')
fig_sma = go.Figure()
fig_sma.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))
fig_sma.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], mode='lines', name='SMA_50'))
fig_sma.update_layout(xaxis_title='Date', yaxis_title='Price')
st.plotly_chart(fig_sma)

    # Plotting Closing Price and EMA_20
st.subheader('Closing Price and Exponential Moving Average (EMA_20)')
fig_ema = go.Figure()
fig_ema.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))
fig_ema.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], mode='lines', name='EMA_20'))
fig_ema.update_layout(xaxis_title='Date', yaxis_title='Price')
st.plotly_chart(fig_ema)

    # Plotting Closing Price and RSI
st.subheader('Closing Price and Relative Strength Index (RSI)')
fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))
fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI'))
fig_rsi.update_layout(xaxis_title='Date', yaxis_title='Price')
st.plotly_chart(fig_rsi)

    # Plotting Closing Price and MACD
st.subheader('Closing Price and Moving Average Convergence Divergence (MACD)')
fig_macd = go.Figure()
fig_macd.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))
fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD'))
fig_macd.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], mode='lines', name='Signal Line'))
fig_macd.update_layout(xaxis_title='Date', yaxis_title='Price')
st.plotly_chart(fig_macd)

    # Plotting Bollinger Bands
st.subheader('Bollinger Bands')
fig_bollinger = go.Figure()
fig_bollinger.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))
fig_bollinger.add_trace(go.Scatter(x=df.index, y=df['Rolling_Mean'], mode='lines', name='Rolling Mean'))
fig_bollinger.add_trace(go.Scatter(x=df.index, y=df['Upper_Band'], mode='lines', name='Upper Band'))
fig_bollinger.add_trace(go.Scatter(x=df.index, y=df['Lower_Band'], mode='lines', name='Lower Band'))
fig_bollinger.update_layout(xaxis_title='Date', yaxis_title='Price')
st.plotly_chart(fig_bollinger)


# Plotting trend pie chart at the end
st.subheader('Trend Analysis')
fig_pie_chart = plot_trend_pie_chart(bullish_score, bearish_score)
st.pyplot(fig_pie_chart)


# Frontend (Streamlit app)

import plotly.graph_objects as go

# Plotting Closing Price
st.subheader('Closing Price')
fig_close_price = go.Figure()
fig_close_price.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))
st.plotly_chart(fig_close_price, use_container_width=True)


import streamlit as st
import feedparser

# Function to fetch latest news about the stock
def fetch_top_5_latest_news(ticker):
    news_feed_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    feed = feedparser.parse(news_feed_url)
    top_5_news = feed['entries'][:5]  # Extract top 5 news articles
    return top_5_news

# Streamlit app
st.title('Stock Market News')


# Fetch latest news
latest_news = fetch_top_5_latest_news(ticker)

# Display news articles
if latest_news:
    st.subheader('Latest News')
    for article in latest_news:
        st.subheader(article['title'])
        st.write(f"Published at: {article['published']}")
        st.write(article['summary'])
else:
    st.error("No news available.")

# Display profits with breakups and promoter information


url = f"https://financialmodelingprep.com/api/v3/income-statement/{ticker}?period=annual&apikey=ZSiE6L1gYeihi92JTgM5fwrkoMLKSNgA"

# Make the API call
response = requests.get(url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse the JSON response
    data = response.json()
    
    # Convert JSON data to DataFrame
    df = pd.DataFrame(data)
    selected_columns = [
        'revenue', 
        'grossProfit',
        'researchAndDevelopmentExpenses',
        'sellingGeneralAndAdministrativeExpenses',
        'otherExpenses',
        'interestIncome',
        'totalOtherIncomeExpensesNet'
    ]
    df_selected = df[selected_columns]
    
    # Display the DataFrame
    st.write(df_selected)

def fetch_top_gainers_and_losers():
    # API endpoint for top gainers and losers
    url = "https://financialmodelingprep.com/api/v3/stock_market/gainers?apikey=ZSiE6L1gYeihi92JTgM5fwrkoMLKSNgA"
    
    # Sending GET request to fetch data
    response = requests.get(url)
    
    # Check if request was successful (status code 200)
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        return data
    else:
        st.error("Failed to fetch top gainers and losers data")
        return None

# Streamlit app

# Fetch top gainers and losers
top_gainers_and_losers = fetch_top_gainers_and_losers()
if top_gainers_and_losers:
    # Display top 5 gainers
    st.subheader("Top 5 Gainers")
    for i, gainer in enumerate(top_gainers_and_losers[:5]):
        st.write(f"{i+1}. {gainer['symbol']} -{gainer['price']}")

    # Display top 5 losers
    st.subheader("Top 5 Losers")
    for i, loser in enumerate(top_gainers_and_losers[-5:][::-1]):
        st.write(f"{i+1}. {loser['symbol']} -{loser['price']}")



import nltk
from nltk.chat.util import Chat, reflections
import streamlit as st
import requests

# Training data for the chatbot
training_data = [
    ("Hi", "Hello!"),
    ("How are you?", "I'm good, thank you."),
    ("What's your name?", "I'm a chatbot."),
    ("Who created you?", "I was created by [Your Name]."),
    ("What is the current price of [ticker]?", "The current price of [ticker] is $[price]."),
    ("What is the trend of [ticker]?", "The trend of [ticker] is [trend]."),
    # Add more patterns and responses as needed
]

# NLTK chatbot initialization
chatbot = Chat(training_data, reflections)

# Function to preprocess user query and get response from the chatbot
def handle_query(user_query):
    # Iterate through training data to find matching patterns
    for pattern, response in training_data:
        # Check if the pattern matches the user query
        if pattern in user_query:
            # Extract ticker symbol from user query
            ticker = extract_ticker_symbol(user_query)
            # If ticker symbol is found, replace placeholder with actual data
            if ticker:
                response = response.replace("[ticker]", ticker)
                # Get current price and trend of the stock
                current_price = fetch_current_price(ticker)
                trend = fetch_stock_trend(ticker)
                # Replace placeholders with actual data
                response = response.replace("[price]", str(current_price))
                response = response.replace("[trend]", trend)
            return response
    # If no matching pattern is found, let the chatbot respond based on reflections
    return chatbot.respond(user_query)

# Function to extract ticker symbol from user query
def extract_ticker_symbol(user_query):
    # Simple implementation to extract ticker symbol from text
    # You may need more sophisticated methods depending on your requirements
    words = user_query.split()
    for word in words:
        if word.isupper() and len(word) <= 5:  # Assuming ticker symbols are uppercase and <= 5 characters
            return word
    return None

# Function to fetch current price of a stock
def fetch_current_price(ticker):
    # API endpoint for current stock price
    url = f'https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey=ZSiE6L1gYeihi92JTgM5fwrkoMLKSNgA'
    # Sending GET request to fetch data
    response = requests.get(url)
    # Check if request was successful (status code 200)
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        return data[0]['price']
    else:
        st.error("Failed to fetch current price data")
        return None

# Function to fetch trend of a stock
def fetch_stock_trend(ticker):
    # Placeholder function to fetch trend of a stock
    # You can implement your own logic to determine the trend
    # For simplicity, returning a static trend for demonstration
    stock_trends = {
        'TSLA': ('bullish', 0.8, 0.2),
        'AAPL': ('bearish', 0.3, 0.7),
        # Add more ticker symbols and their corresponding trends
    }
    return stock_trends.get(ticker)

# Streamlit app
st.title("Stock Market Chatbot")

# Sidebar for chatbot interface
st.sidebar.title("Chatbot Interface")
user_query = st.sidebar.text_input("You:")
if user_query:
    bot_response = handle_query(user_query)
    st.sidebar.text_area("Bot:", bot_response, height=100)
