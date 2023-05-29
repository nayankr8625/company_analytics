import streamlit as st
from streamlit import config as st_config
from PIL import Image
from sentiment_based_forecasting.data_processing import download_tickers
from sentiment_based_forecasting.ml_models import MLModels
from sentiment_based_forecasting.pipeline import PipelineTasks
from services import measure_time, logger
import matplotlib.pyplot as plt
import pandas as pd
import os

# Set Streamlit app title and sidebar
st.set_page_config(page_title="COMPANY ANALYTICS", layout="wide")
# Create a sidebar
st.sidebar.title("WELCOME!")

def streamlit_app(quote):

    st.balloons()

    # Initialize PipelineTasks
    tasks = PipelineTasks(quote=quote)

    # Perform data generation and analysis tasks
    st.subheader("Data Generation and Analysis")


    # Stock Data Generation
    data_gen_button = st.button("Generate Stock Data")
    if data_gen_button:
        st.write("Data generation task is running...")
        tasks.stock_data = tasks.data_generation()
        st.write("Data generation task completed!")
        st.write(f"Downloaded latest stock data of {tasks._quote}")
        st.dataframe(tasks.stock_data.head())

    # ESG Data Generation
    esg_data_gen_button = st.button("Generate ESG Data")
    if esg_data_gen_button:
        st.write("ESG data generation task is running...")
        tasks.esg_rating = tasks.esg_data_generation()
        st.write("ESG data generation task completed!")
        st.write(f"Latest ESG rating of {tasks._quote}")
        st.write(tasks.esg_rating)

    # News Data Generation
    news_data_gen_button = st.button("Generate News Data")
    if news_data_gen_button:
        st.write("News data generation task is running...")
        df = tasks.news_data_generation()
        tasks.news_data = df
        st.write("News data generation task completed!")
        st.write("Top 5 News:")
        st.dataframe(tasks.news_data.head(5))

    st.subheader("Modeling and Forecasting")

    # ARIMA Model
    arima_button = st.button("Run ARIMA Model")
    if arima_button:
        st.write("ARIMA model task is running...")
        tasks.arima_result = tasks.arima_model()
        st.write("ARIMA model task completed!")
        st.write("ARIMA Forecasting for Next DAY:", tasks.arima_result['arima_pred'])
        st.write("ARIMA Error:", tasks.arima_result['error_arima'])
        st.write("ARIMA Forecasting Plot:")
        st.image(tasks.arima_result['path'])

    # LSTM Model
    lstm_button = st.button("Run LSTM Model")
    if lstm_button:
        st.write("LSTM model task is running...")
        tasks.lstm_result = tasks.lstm_model()
        st.write("LSTM model task completed!")
        st.write("LSTM Forecasting for Next DAY:", tasks.lstm_result['lstm_pred'])
        st.write("LSTM Error:", tasks.lstm_result['error_lstm'])
        st.write("LSTM Forecasting Plot:")
        st.image(tasks.lstm_result['path'])

    # Regressor Model
    regressor_button = st.button("Run Regressor Model")
    if regressor_button:
        st.write("Regressor model task is running...")
        tasks.regressor_result = tasks.regressor_model()
        st.write("Regressor model task completed!")
        st.write("Regressor Forecasting for Next DAY:", tasks.regressor_result['lr_pred'])
        st.write("Regressor Error:", tasks.regressor_result['error_lr'])
        st.write("Regressor Forecasting Plot:")
        st.image(tasks.regressor_result['path'])

    st.subheader("Sentiment Analysis")

    # Sentiment Analysis
    sentiment_analysis_button = st.button("Run Sentiment Analysis")
    if sentiment_analysis_button:
        st.write("Sentiment analysis task is running...")
        tasks.sentiment_result = tasks.sentiment_analyze_task()
        st.write("Sentiment analysis task completed!")
        st.write("Showing Sentiment analysis charts")
        st.image(tasks.sentiment_result['path'])


if __name__ == '__main__':
    st.snow()
    st.write("<h1 style='font-size: 36px;'> ENTER STOCK SYMBOL TO CONTINUE </h1>", unsafe_allow_html=True)
    st.title("COMPANY ANALYTICS")
    logo_path = "logo/kaninilogo.jpg"  # Replace with the actual path to your logo image
    logo_image = Image.open(logo_path)
    st.sidebar.image(logo_image, use_column_width=True)
    quote = st.sidebar.text_input("Enter company tickers SYMBOL")
    streamlit_app(quote=quote)


