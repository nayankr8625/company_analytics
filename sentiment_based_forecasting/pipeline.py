from sentiment_based_forecasting.data_processing import download_tickers
from sentiment_based_forecasting.ml_models import MLModels
from services import measure_time,logger

import matplotlib.pyplot as plt
import pandas as pd
import os

class PipelineTasks:

    def __init__(self,quote):

        self._quote = quote
        self._model = MLModels(data=self.data_generation(),quote=self._quote)

        self.DATE = pd.Timestamp.today().strftime('%Y-%m-%d')
        
    @measure_time
    def data_generation(self):
        data = download_tickers(tickers=self._quote).get_historical_yf()

        return data
    
    @measure_time
    def esg_data_generation(self):
        logger.info(f'ESG RATING  COLLECTION OF {self._quote} TASK STARTED')
        df,json_data = download_tickers(self._quote).scrape_company_esg_data()
        # CSV file path
        file_path = 'esg_data/collected_esg_data.csv'

        # Check if the file already exists
        if os.path.exists(file_path):
            # Append the DataFrame to the existing file
            df.to_csv(file_path, mode='a', header=False, index=False)
        else:
            # Create a new file and write the DataFrame to it
            df.to_csv(file_path, index=False)
        return df,json_data,file_path
    
    @measure_time
    def news_data_generation(self):
        logger.info(f'NEWS COLLECTION for {self._quote} TASK STARTED')
        df = download_tickers(self._quote).news_api_stock_news()
        return df
    
    @measure_time
    def arima_model(self):
        logger.info('DOING FORECATSING BY USING ARIMA MODEL')
        arima_pred, error_arima, test, predictions = self._model.ARIMA_ALGO()
        fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
        plt.plot(test, label='Actual Price')
        plt.plot(predictions, label='Predicted Price')
        plt.legend(loc=4)
        image_path = f'images/ARIMA forecasting {self._quote} on {self.DATE} .png'
        plt.savefig(image_path)
        plt.close(fig)

        return {
            'arima_pred' : arima_pred,
            'error_arima' : error_arima,
            'path': image_path
        }
    
    @measure_time
    def lstm_model(self):
        logger.info('DOING FORECATSING BY USING LSTM MODEL')
        lstm_pred, error_lstm , real_stock_price , predicted_stock_price = self._model.LSTM_ALGO()
        fig = plt.figure(figsize=(7.2, 4.8), dpi=65)

        plt.plot(real_stock_price, label='Actual Price')
        plt.plot(predicted_stock_price, label='Predicted Price')

        plt.legend(loc=4)
        image_path = f'images/LSTM forecasting {self._quote} on {self.DATE} .png'
        plt.savefig(image_path)
        plt.close(fig)

        return {
            'lstm_pred' : lstm_pred,
            'error_lstm' : error_lstm,
            'path': image_path
        }
    
    @measure_time
    def regressor_model(self):
        logger.info('DOING FORECATSING BY USING REGRESSOR MODEL')
        df, lr_pred, forecast_set, mean, error_lr, y_test, y_test_pred = self._model.LIN_REG_ALGO()
        fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
        plt.plot(y_test, label='Actual Price')
        plt.plot(y_test_pred, label='Predicted Price')

        plt.legend(loc=4)
        image_path = f'images/Regressor forecasting {self._quote} on {self.DATE} .png'
        plt.savefig(image_path)
        plt.close(fig)

        return {
            'lr_pred' : lr_pred,
            'error_lr' : error_lr,
            'path': image_path
        }
    
    # def collect_news_task(self):
    #     logger.info(f'NEWS COLLECTION for {self._quote} TASK STARTED')

    #     news_data = self._model.collect_news()

    #     return news_data

    @measure_time
    def sentiment_analyze_task(self):
        logger.info(f'SENTIMENT ANALYSIS OF {self._quote} NEWS TASK STARTED')

        news_list, global_polarity, tw_pol, positive, neutral, negative,\
        positive_list, negative_list, neutral_list = self._model.sentiment_analysis()

        #Creating PieCart
        labels = ['Positive ['+str(round(positive))+'%]' , 'Neutral ['+str(round(neutral))+'%]','Negative ['+str(round(negative))+'%]']
        sizes = [positive, neutral, negative]
        colors = ['yellowgreen', 'blue','red']
        patches, texts = plt.pie(sizes,colors=colors, startangle=90)
        plt.style.use('default')
        plt.legend(labels)
        plt.title("Sentiment Analysis Result for stock= "+self._quote+"" )
        plt.axis('equal')
        # plt.savefig('SentAnalysis.png')
        image_path = f'images/sentiment_analysis of stock {self._quote} on date {self.DATE}.png'
        plt.savefig(image_path)
        plt.close()

        return {
            'positive_news': positive_list,
            'negative_news': negative_list,
            'neutral_news': neutral_list,
            'global_polarity': global_polarity,
            'path': image_path
        }
    
    @measure_time
    def download_company_data(self):

        quote = self._quote
        download_stock_data = self.data_generation()
        download_esg_data = self.esg_data_generation()[0]
        download_news_data = self.news_data_generation()

        # Define the folder path and the file name
        stock_data_folder_path = 'company_data/stock_data'
        esg_data_folder_path  = 'company_data/esg_data'
        news_data_folder_path = 'company_data/news_data'


        # Check if the folder exists, create it if necessary
        if not os.path.exists(stock_data_folder_path):
            os.makedirs(stock_data_folder_path)
        download_stock_data.to_csv(os.path.join(stock_data_folder_path, f'{self.DATE}_stock_data_of_{quote}.csv'), index=False)


        if not os.path.exists(esg_data_folder_path):
            os.makedirs(esg_data_folder_path)
        download_esg_data.to_csv(os.path.join(esg_data_folder_path, f'{self.DATE}_esg_data_of_{quote}.csv'), index=False)



        if not os.path.exists(news_data_folder_path):
            os.makedirs(news_data_folder_path)
        download_news_data.to_csv(os.path.join(news_data_folder_path, f'{self.DATE}_news_data_of_{quote}.csv'), index=False)

        return {
            'stock_data_path': f'{stock_data_folder_path}{self.DATE}_stock_data_of_{quote}.csv',
            'esg_data_path': f'{esg_data_folder_path}{self.DATE}_esg_data_of_{quote}.csv',
            'news_data_path': f'{news_data_folder_path}{self.DATE}_news_data_of_{quote}.csv'
        }





        












    





        