import yfinance as yf
from datetime import datetime
from alpha_vantage.timeseries import TimeSeries

import pandas as pd
import numpy as np

class download_tickers:

    def __init__(self,tickers):
        self._tickers = tickers

    def get_historical_yf(self):
        quote = self._tickers
        end = datetime.now()
        start = datetime(end.year-2, end.month, end.day)
        data = yf.Ticker(quote).history(period='5y',interval='1d')
        data['Date']=data.index.date
        # data['Date']=pd.to_datetime(data['Date'],format='%Y-%M-%d')
        data['STOCK'] = quote
        df = pd.DataFrame(data=data)
        df = df[['Date', 'STOCK','Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']]
        df.reset_index(drop=True,inplace=True)
        return df
    
    def get_historical_other_api(self):
        pass
 

