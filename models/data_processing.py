import yfinance as yf
from datetime import datetime
from alpha_vantage.timeseries import TimeSeries

import pandas as pd
import numpy as np

class download_tickers:

    def __init__(self,tickers):
        self._tickers = tickers

    def get_historical(self):
        quote = self._tickers
        end = datetime.now()
        start = datetime(end.year-2, end.month, end.day)
        data = yf.download(quote, start=start, end=end)
        df = pd.DataFrame(data=data)
        if (df.empty):
            ts = TimeSeries(key='GEFTUH2PBNLRCZAL', output_format='pandas')
            data, meta_data = ts.get_daily_adjusted(
                symbol='NSE:'+quote, outputsize='full')
            # Format df
            # Last 2 yrs rows => 502, in ascending order => ::-1
            data = data.head(503).iloc[::-1]
            data = data.reset_index()
            # Keep Required cols only
            df = pd.DataFrame()
            df['Date'] = data['date']
            df['Open'] = data['1. open']
            df['High'] = data['2. high']
            df['Low'] = data['3. low']
            df['Close'] = data['4. close']
            df['Adj Close'] = data['5. adjusted close']
            df['Volume'] = data['6. volume']
            # df.to_csv(''+quote+'.csv', index=False)
        return df


    

