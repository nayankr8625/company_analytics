import streamlit as st
from PIL import Image
from sentiment_based_forecasting.data_processing import download_tickers
from sentiment_based_forecasting.ml_models import MLModels
from sentiment_based_forecasting.pipeline import PipelineTasks
from services import measure_time, logger
import matplotlib.pyplot as plt
import pandas as pd
import os
import nltk
# nltk.download('punkt')
# nltk.download('vader_lexicon')


# Set Streamlit app title and sidebar
st.set_page_config(page_title="COMPANY ANALYTICS", layout="wide")
# Create a sidebar
st.sidebar.title("WELCOME!")

# SessionState class for caching variables
class SessionState:
    def __init__(self):
        self.cache = {}

def get_session_state():
    if 'session_state' not in st.session_state:
        st.session_state.session_state = SessionState()
    return st.session_state.session_state

def clear_cache(session_state):
    session_state.cache = {}

def streamlit_app(tasks, session_state):

    # Perform data generation and analysis tasks
    st.subheader("Data Generation and Analysis")

    # Stock Data Generation
    data_gen_button = st.button("Generate Stock Data")
    if data_gen_button:
        st.write("Data generation task is running...")
        session_state.cache['stock_data'] = tasks.data_generation()
        st.write("Data generation task completed!")
        st.write(f"Downloaded latest stock data of {tasks._quote}")
        st.dataframe(session_state.cache['stock_data'].head())

    # ESG Data Generation
    esg_data_gen_button = st.button("Generate ESG Data")
    if esg_data_gen_button:
        st.write("ESG data generation task is running...")
        session_state.cache['esg_rating'] = tasks.esg_data_generation()[1]
        st.write("ESG data generation task completed!")
        st.write(f"Latest ESG rating of {tasks._quote}")
        st.json(session_state.cache['esg_rating'])

    # News Data Generation
    news_data_gen_button = st.button("Generate News Data")
    if news_data_gen_button:
        st.write("News data generation task is running...")
        session_state.cache['news_df'] = tasks.news_data_generation()
        st.write("News data generation task completed!")
        st.write("Top 5 News:")
        st.dataframe(session_state.cache['news_df'].head(20))

    st.subheader("Modeling and Forecasting")

    # ARIMA Model
    arima_button = st.button("Run ARIMA Model")
    if arima_button:
        st.write("ARIMA model task is running...")
        session_state.cache['arima_result'] = tasks.arima_model()
        st.write("ARIMA model task completed!")
        st.write("ARIMA Forecasting for Next DAY:", session_state.cache['arima_result']['arima_pred'])
        st.write("ARIMA Error:", session_state.cache['arima_result']['error_arima'])
        st.write("ARIMA Forecasting Plot:")
        st.image(session_state.cache['arima_result']['path'])

    # LSTM Model
    lstm_button = st.button("Run LSTM Model")
    if lstm_button:
        st.write("LSTM model task is running...")
        session_state.cache['lstm_result'] = tasks.lstm_model()
        st.write("LSTM model task completed!")
        st.write("LSTM Forecasting for Next DAY:", session_state.cache['lstm_result']['lstm_pred'])
        st.write("LSTM Error:", session_state.cache['lstm_result']['error_lstm'])
        st.write("LSTM Forecasting Plot:")
        st.image(session_state.cache['lstm_result']['path'])

    # Regressor Model
    regressor_button = st.button("Run Regressor Model")
    if regressor_button:
        st.write("Regressor model task is running...")
        session_state.cache['regressor_result'] = tasks.regressor_model()
        st.write("Regressor model task completed!")
        st.write("Regressor Forecasting for Next DAY:", session_state.cache['regressor_result']['lr_pred'])
        st.write("Regressor Error:", session_state.cache['regressor_result']['error_lr'])
        st.write("Regressor Forecasting Plot:")
        st.image(session_state.cache['regressor_result']['path'])

    st.subheader("Sentiment Analysis")

    # Sentiment Analysis
    sentiment_analysis_button = st.button("Run Sentiment Analysis")
    if sentiment_analysis_button:
        st.write("Sentiment analysis task is running...")
        if 'news_df' in session_state.cache:
            session_state.cache['sentiment_result'] = tasks.sentiment_analyze_task(df=session_state.cache['news_df'])
            st.write("Sentiment analysis task completed!")
            st.write("Showing Sentiment analysis charts")
            st.image(session_state.cache['sentiment_result']['path'])
        else:
            st.write("Please generate news data first.")


    st.subheader("Recommendation")
    # Recommendation
    Recommendation_button = st.button("Stock Recommendation")
    if Recommendation_button:
        arima_result = session_state.cache.get('arima_result')
        lstm_result = session_state.cache.get('lstm_result')
        regressor_result = session_state.cache.get('regressor_result')
        sentiment_result = session_state.cache.get('sentiment_result')
        stock_data = session_state.cache.get('stock_data')

        # Check if sentiment_result is None
        if any(value is None for value in (arima_result, lstm_result, regressor_result, sentiment_result)):
            st.write("Sentiment analysis and Building Forecasting model is required before making a recommendation.")
        else:
            mean = stock_data['Close'].mean()
            st.write(f'Showing Recommendation for {tasks._quote} Stock')
            st.write("Showing LSTM Forecasting Charts")
            st.image(lstm_result['path'])
            st.write("Showing ARIMA Forecasting Charts")
            st.image(arima_result['path'])
            st.write("Showing Regressor Forecasting Charts")
            st.image(regressor_result['path'])
            st.write(f'Showing Sentiment analysis chart')
            st.image(sentiment_result['path'])

            st.subheader('Showing Recommendation')
            st.write(tasks.recommendation(sentiment_result=sentiment_result,
                                          mean=mean,
                                          arima_result=arima_result,
                                          lstm_result=lstm_result,
                                          regressor_result=regressor_result))


if __name__ == '__main__':
    session_state = get_session_state()
    show_message = True
    
    st.title("COMPANY ANALYTICS")
    logo_path = "logo/kaninilogo.jpg"  # Replace with the actual path to your logo image
    logo_image = Image.open(logo_path)
    st.sidebar.image(logo_image, use_column_width=True)
    quote = st.sidebar.text_input("Enter company tickers SYMBOL")

    if quote:
        show_message  = False
        st.write("Stock Symbol Entered Now you can see latest company data")
        # Check if the stock symbol has changed
        if 'quote' in session_state.cache and session_state.cache['quote'] != quote:
            clear_cache(session_state)
        session_state.cache['quote'] = quote
        streamlit_app(tasks=PipelineTasks(quote=quote), session_state=session_state)

    if show_message:
        st.write("<span style='color: red; font-size: 24px;'>ENTER STOCK SYMBOL TO CONTINUE</span>", unsafe_allow_html=True)
    
