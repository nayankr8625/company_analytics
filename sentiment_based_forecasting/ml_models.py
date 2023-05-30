import random
import keras
import math
import os
import warnings
import nltk
from sklearn.linear_model import LinearRegression
import re
import yfinance as yf
import datetime as dt
from datetime import datetime
from flask import Flask, render_template, request, flash, redirect, url_for
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import numpy as np

## INternal packages
from sentiment_based_forecasting.data_processing import download_tickers
# from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as smapi
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Building RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM

## Sentiments
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from GoogleNews import GoogleNews
from newspaper import Article
from newspaper import Config
# from wordcloud import WordCloud, STOPWORDS
# nltk.set_proxy('SYSTEM PROXY')
# nltk.download('vader_lexicon')
    
    
plt.style.use('ggplot')
# nltk.download('punkt')

# Ignore Warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class MLModels:

    def __init__(self,data,quote):

        
        self._data = data
        self._quote  =quote
    

    def ARIMA_ALGO(self):
        df = self._data
        quote = self._quote
        code_list = []
        for i in range(0, len(df)):
            code_list.append(quote)
        df2 = pd.DataFrame(code_list, columns=['Code'])
        df2 = pd.concat([df2, df], axis=1)
        df = df2
        uniqueVals = df["Code"].unique()
        len(uniqueVals)
        df = df.set_index("Code")
        # for daily basis

        def parser(x):
            x=x.strftime('%Y-%m-%d')
            return datetime.strptime(x, '%Y-%m-%d')

        def arima_model(train, test):
            history = [x for x in train]
            predictions = list()
            for t in range(len(test)):
                # model = ARIMA(history, order=(6, 1, 0))
                model = smapi.tsa.arima.ARIMA(history, order=(1, 1, 2))
                model_fit = model.fit()
                output = model_fit.forecast()
                yhat = output[0]
                print(yhat)
                predictions.append(yhat)
                obs = test[t]
                history.append(obs)
            return predictions
        for company in uniqueVals[:10]:
            data = (df.loc[company, :]).reset_index()
            data['Price'] = data['Close']
            Quantity_date = data[['Price', 'Date']]
            print(Quantity_date.info())
            print(Quantity_date.Date.values[0])
            print(type(Quantity_date.Date.values[0]))
            Quantity_date.index = Quantity_date['Date'].map(
                lambda x: parser(x))
            Quantity_date['Price'] = Quantity_date['Price'].map(
                lambda x: float(x))
            Quantity_date = Quantity_date.fillna(Quantity_date.bfill())
            Quantity_date = Quantity_date.drop(['Date'], axis=1)

            quantity = Quantity_date.values
            size = int(len(quantity) * 0.80)
            train, test = quantity[0:size], quantity[size:len(quantity)]
            # fit in model
            predictions = arima_model(train, test)
            print()
            print(
                "##############################################################################")
            arima_pred = predictions[-2]
            print("Tomorrow's", quote,
                  " Closing Price Prediction by ARIMA:", arima_pred)
            # rmse calculation
            error_arima = math.sqrt(mean_squared_error(test, predictions))
            print("ARIMA RMSE:", error_arima)
            print(
                "##############################################################################")
            return arima_pred, error_arima, test, predictions
        

    def LSTM_ALGO(self):
        df = self._data
        quote = self._quote
        # Split data into training set and test set
        dataset_train = df.iloc[0:int(0.8*len(df)), :]
        dataset_test = df.iloc[int(0.8*len(df)):, :]
        ############# NOTE #################
        # TO PREDICT STOCK PRICES OF NEXT N DAYS, STORE PREVIOUS N DAYS IN MEMORY WHILE TRAINING
        # HERE N=7
        # dataset_train=pd.read_csv('Google_Stock_Price_Train.csv')
        # 1:2, to store as numpy array else Series obj will be stored
        training_set = df.iloc[:, 4:5].values
        # select cols using above manner to select as float64 type, view in var explorer

        # Feature Scaling
        from sklearn.preprocessing import MinMaxScaler
        sc = MinMaxScaler(feature_range=(0, 1))  # Scaled values btween 0,1
        training_set_scaled = sc.fit_transform(training_set)
        # In scaling, fit_transform for training, transform for test

        # Creating data stucture with 7 timesteps and 1 output.
        # 7 timesteps meaning storing trends from 7 days before current day to predict 1 next output
        X_train = []  # memory with 7 days from day i
        y_train = []  # day i
        for i in range(7, len(training_set_scaled)):
            X_train.append(training_set_scaled[i-7:i, 0])
            y_train.append(training_set_scaled[i, 0])
        # Convert list to numpy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_forecast = np.array(X_train[-1, 1:])
        X_forecast = np.append(X_forecast, y_train[-1])
        # Reshaping: Adding 3rd dimension
        # .shape 0=row,1=col
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_forecast = np.reshape(X_forecast, (1, X_forecast.shape[0], 1))
        # For X_train=np.reshape(no. of rows/samples, timesteps, no. of cols/features

        # Initialise RNN
        regressor = Sequential()

        # Add first LSTM layer
        regressor.add(LSTM(units=50, return_sequences=True,
                      input_shape=(X_train.shape[1], 1)))
        # units=no. of neurons in layer
        # input_shape=(timesteps,no. of cols/features)
        # return_seq=True for sending recc memory. For last layer, retrun_seq=False since end of the line
        regressor.add(Dropout(0.1))

        # Add 2nd LSTM layer
        regressor.add(LSTM(units=50, return_sequences=True))
        regressor.add(Dropout(0.1))

        # Add 3rd LSTM layer
        regressor.add(LSTM(units=50, return_sequences=True))
        regressor.add(Dropout(0.1))

        # Add 4th LSTM layer
        regressor.add(LSTM(units=50))
        regressor.add(Dropout(0.1))

        # Add o/p layer
        regressor.add(Dense(units=1))

        # Compile
        regressor.compile(optimizer='adam', loss='mean_squared_error')

        # Training
        regressor.fit(X_train, y_train, epochs=25, batch_size=32)
        # For lstm, batch_size=power of 2

        # Testing
        # dataset_test=pd.read_csv('Google_Stock_Price_Test.csv')
        real_stock_price = dataset_test.iloc[:, 4:5].values

        # To predict, we need stock prices of 7 days before the test set
        # So combine train and test set to get the entire data set
        dataset_total = pd.concat(
            (dataset_train['Close'], dataset_test['Close']), axis=0)
        testing_set = dataset_total[len(
            dataset_total) - len(dataset_test) - 7:].values
        testing_set = testing_set.reshape(-1, 1)
        # -1=till last row, (-1,1)=>(80,1). otherwise only (80,0)

        # Feature scaling
        testing_set = sc.transform(testing_set)

        # Create data structure
        X_test = []
        for i in range(7, len(testing_set)):
            X_test.append(testing_set[i-7:i, 0])
            # Convert list to numpy arrays
        X_test = np.array(X_test)

        # Reshaping: Adding 3rd dimension
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # Testing Prediction
        predicted_stock_price = regressor.predict(X_test)

        # Getting original prices back from scaled values
        predicted_stock_price = sc.inverse_transform(predicted_stock_price)

        error_lstm = math.sqrt(mean_squared_error(
            real_stock_price, predicted_stock_price))

        # Forecasting Prediction
        forecasted_stock_price = regressor.predict(X_forecast)

        # Getting original prices back from scaled values
        forecasted_stock_price = sc.inverse_transform(forecasted_stock_price)

        lstm_pred = forecasted_stock_price[0, 0]
        print()
        print(
            "##############################################################################")
        print("Tomorrow's ", quote, " Closing Price Prediction by LSTM: ", lstm_pred)
        print("LSTM RMSE:", error_lstm)
        print(
            "##############################################################################")
        return lstm_pred, error_lstm , real_stock_price , predicted_stock_price
    

    def LIN_REG_ALGO(self):
        df = self._data
        quote = self._quote
        # No of days to be forcasted in future
        forecast_out = int(7)
        # Price after n days
        df['Close after n days'] = df['Close'].shift(-forecast_out)
        # New df with only relevant data
        df_new = df[['Close', 'Close after n days']]

        # Structure data for train, test & forecast
        # lables of known data, discard last 35 rows
        y = np.array(df_new.iloc[:-forecast_out, -1])
        y = np.reshape(y, (-1, 1))
        # all cols of known data except lables, discard last 35 rows
        X = np.array(df_new.iloc[:-forecast_out, 0:-1])
        # Unknown, X to be forecasted
        X_to_be_forecasted = np.array(df_new.iloc[-forecast_out:, 0:-1])

        # Traning, testing to plot graphs, check accuracy
        X_train = X[0:int(0.8*len(df)), :]
        X_test = X[int(0.8*len(df)):, :]
        y_train = y[0:int(0.8*len(df)), :]
        y_test = y[int(0.8*len(df)):, :]

        # Feature Scaling===Normalization
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        X_to_be_forecasted = sc.transform(X_to_be_forecasted)

        # Training
        clf = LinearRegression(n_jobs=-1)
        clf.fit(X_train, y_train)

        # Testing
        y_test_pred = clf.predict(X_test)
        y_test_pred = y_test_pred*(1.04)
        # fig = plt2.figure(figsize=(7.2, 4.8), dpi=65)
        # plt2.plot(y_test, label='Actual Price')
        # plt2.plot(y_test_pred, label='Predicted Price')

        # plt2.legend(loc=4)
        # plt2.savefig('../static/LR.png')
        # plt2.close(fig)

        error_lr = math.sqrt(mean_squared_error(y_test, y_test_pred))

        # Forecasting
        forecast_set = clf.predict(X_to_be_forecasted)
        forecast_set = forecast_set*(1.04)
        mean = forecast_set.mean()
        lr_pred = forecast_set[0, 0]
        print()
        print(
            "##############################################################################")
        print("Tomorrow's ", quote,
              " Closing Price Prediction by Linear Regression: ", lr_pred)
        print("Linear Regression RMSE:", error_lr)
        print(
            "##############################################################################")
        return df, lr_pred, forecast_set, mean, error_lr, y_test, y_test_pred
    
    ## Below method is to extract news using Google news api due to some error currently it's not being used.
    def collect_news(self):
        quote = self._quote
        now = dt.date.today()
        now = now.strftime('%m-%d-%Y')
        yesterday = dt.date.today() - dt.timedelta(days=1)
        yesterday = yesterday.strftime('%m-%d-%Y')
        nltk.download('punkt')
        user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:78.0) Gecko/20100101 Firefox/78.0'
        config = Config()
        config.browser_user_agent = user_agent
        config.request_timeout = 10
        company_name = quote
        # As long as the company name is valid, not empty...
        if company_name != '':
            print(
                f'Searching for and analyzing {company_name}, Please be patient, it might take a while...')

            # Extract News with Google News
            googlenews = GoogleNews(start=yesterday, end=now)
            googlenews.search(company_name)
            result = googlenews.result()
            # store the results
            df = pd.DataFrame(result)

        try:
            ls = []  # creating an empty ls
            for i in df.index:
                dict = {}  # creating an empty dictionary to append an article in every single iteration
                # providing the link
                article = Article(df['link'][i], config=config)
                try:
                    article.download()  # downloading the article
                    article.parse()  # parsing the article
                    article.nlp()  # performing natural language processing (nlp)
                except:
                    pass
                # storing results in our empty dictionary
                dict['Date'] = df['datetime'][i]
                dict['Media'] = df['media'][i]
                dict['Title'] = article.title
                dict['Article'] = article.text
                dict['Summary'] = article.summary
                dict['Key_words'] = article.keywords
                ls.append(dict)
            check_empty = not any(ls)
            # print(check_empty)
            if check_empty == False:
                news_df = pd.DataFrame(ls)  # creating dataframe
            print(f'Shape of collected data is {news_df.shape}')
            return news_df

        except Exception as e:
            # exception handling
            print("exception occurred:" + str(e))
            print('Looks like, there is some error in retrieving the data, Please try again or try with a different ticker.')
            
            
    def sentiment_analysis(self,df):
        df = df
        print(f'Shape of collected data is {df.shape}')
        #Sentiment Analysis
        def percentage(part,whole):
            return 100 * float(part)/float(whole)

        #Assigning Initial Values
        positive = 0
        negative = 0
        neutral = 0
        #Creating empty lists
        news_list = []
        neutral_list = []
        negative_list = []
        positive_list = []

        #Iterating over the tweets in the dataframe
        for news in df['Content']:
            news_list.append(news)
            analyzer = SentimentIntensityAnalyzer().polarity_scores(news)
            neg = analyzer['neg']
            neu = analyzer['neu']
            pos = analyzer['pos']
            comp = analyzer['compound']

            if neg > pos:
                negative_list.append(news) #appending the news that satisfies this condition
                negative += 1 #increasing the count by 1
            elif pos > neg:
                positive_list.append(news) #appending the news that satisfies this condition
                positive += 1 #increasing the count by 1
            elif pos == neg:
                neutral_list.append(news) #appending the news that satisfies this condition
                neutral += 1 #increasing the count by 1 

        positive = percentage(positive, len(df)) #percentage is the function defined above
        negative = percentage(negative, len(df))
        neutral = percentage(neutral, len(df))
        
        if positive>negative:
            global_polarity = 1
            tw_pol = 'Overall Positive'
        else:
            tw_pol = 'Overall_Negative'
            global_polarity = 0

        # Converting lists to pandas dataframe
        # news_list = pd.DataFrame(news_list)
        # neutral_list = pd.DataFrame(neutral_list)
        # negative_list = pd.DataFrame(negative_list)
        # positive_list = pd.DataFrame(positive_list)
        # using len(length) function for counting
        print("Positive Sentiment:", '%.2f' % len(positive_list), end='\n')
        print("Neutral Sentiment:", '%.2f' % len(neutral_list), end='\n')
        print("Negative Sentiment:", '%.2f' % len(negative_list), end='\n')
        
        return news_list, global_polarity, tw_pol, positive, neutral, negative, positive_list, negative_list, neutral_list
               


    def recommending(self,df, global_polarity, today_stock, mean):
        quote = self._quote
        if today_stock.iloc[-1]['Close'] < mean:
            if global_polarity > 0:
                idea = "RISE"
                decision = "BUY"
                print()
                print(
                    "##############################################################################")
                print("According to the ML Predictions and Sentiment Analysis of Tweets, a",
                      idea, "in", quote, "stock is expected => ", decision)
            elif global_polarity <= 0:
                idea = "FALL"
                decision = "SELL"
                print()
                print(
                    "##############################################################################")
                print("According to the ML Predictions and Sentiment Analysis of Tweets, a",
                      idea, "in", quote, "stock is expected => ", decision)
        else:
            idea = "FALL"
            decision = "SELL"
            print()
            print(
                "##############################################################################")
            print("According to the ML Predictions and Sentiment Analysis of Tweets, a",
                  idea, "in", quote, "stock is expected => ", decision)
        return idea, decision

    