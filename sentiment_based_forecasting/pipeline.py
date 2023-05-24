from sentiment_based_forecasting.data_processing import download_tickers
from sentiment_based_forecasting.ml_models import MLModels

import matplotlib.pyplot as plt
import pandas as pd

class PipelineTasks:

    def __init__(self,quote):

        self._quote = quote
        self._model = MLModels(data=self.data_generation(),quote=self._quote)

        self.DATE = pd.Timestamp.today().strftime('%Y-%m-%d')
        

    def data_generation(self):
        data = download_tickers(tickers=self._quote).get_historical_yf()

        return data
    
    def arima_model(self):
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

    def lstm_model(self):
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

    def regressor_model(self):
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


        