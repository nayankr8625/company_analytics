from sentiment_based_forecasting.data_processing import download_tickers
from sentiment_based_forecasting.ml_models import MLModels
from sentiment_based_forecasting.pipeline import PipelineTasks

import uvicorn
import pandas as pd
from pandas import DataFrame
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import RedirectResponse, JSONResponse, HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import nltk


nltk.download('punkt')
nltk.download('vader_lexicon')


app = FastAPI()

# Configure CORS
origins = [
    "http://localhost",
    "http://localhost:3000",  # Replace with the actual URL of your React app
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def root():
    # Redirect to the /docs endpoint
    return RedirectResponse(url="/docs")


def show_data(quote):

    response = PipelineTasks(quote=quote).data_generation()
    response = response.tail(30)
    html_data = response.to_html(index=False)
    return html_data

@app.get("/show_data",description="Show last 30 days data")
def open_html_response(quote:str):
    # HTML content received from your endpoint
    html_content = show_data(quote=quote)

    # JavaScript code to open the HTML content in a new tab
    js_code = """
    <script>
        function openInNewTab(htmlContent) {{
            var blob = new Blob([htmlContent], {{type: 'text/html'}});
            var url = URL.createObjectURL(blob);
            var link = document.createElement('a');
            link.href = url;
            link.target = '_blank';
            link.click();
            URL.revokeObjectURL(url);
        }}
        openInNewTab(`{}`);
    </script>
    """.format(html_content.replace("`", r"\`"))  # Escape backticks in the HTML content

    # Return the JavaScript code as the response
    response = Response(content=js_code, media_type="text/html")
    return response


@app.get("/show_esg_data",description="Show ESG ratings of company")
def show_esg_data(quote:str):

    df,response,file_path = PipelineTasks(quote=quote).esg_data_generation()

    if response:
        return JSONResponse({
            'esg_ratings':response,
            'csv file path':file_path      
    })

def show_news_data(quote):

    response = PipelineTasks(quote=quote).news_data_generation()
    html_data = response.to_html(index=False)
    return html_data

@app.get("/show_news_data",description="Show latest news of particular stock symbol")
def open_html_response(quote:str):
    # HTML content received from your endpoint
    html_content = show_news_data(quote=quote)

    # JavaScript code to open the HTML content in a new tab
    js_code = """
    <script>
        function openInNewTab(htmlContent) {{
            var blob = new Blob([htmlContent], {{type: 'text/html'}});
            var url = URL.createObjectURL(blob);
            var link = document.createElement('a');
            link.href = url;
            link.target = '_blank';
            link.click();
            URL.revokeObjectURL(url);
        }}
        openInNewTab(`{}`);
    </script>
    """.format(html_content.replace("`", r"\`"))  # Escape backticks in the HTML content

    # Return the JavaScript code as the response
    response = Response(content=js_code, media_type="text/html")
    return response

    

@app.post("/lstm_forecast", description="Do LSTM forecasting and Predict next day stock price")
async def lstm_forecasting(quote : str):

    response = PipelineTasks(quote=quote).lstm_model()
    if response:
        return JSONResponse({
            "Tomorrow's  AAPL  Closing Price Prediction by LSTM:" : float(response['lstm_pred']),
            "LSTM RMSE": float(response['error_lstm']),
            "image_path": response['path']
                })
    else:
        raise HTTPException(status_code=404, detail="Item not found")
    

@app.post("/arima_foreacst", description="Do ARIMA forecasting and Predict next day stock price")
async def arima_forecasting(quote : str):

    response = PipelineTasks(quote=quote).arima_model()
    if response:
        return JSONResponse({
            "Tomorrow's  AAPL  Closing Price Prediction by ARIMA" : float(response['arima_pred']),
            "Arima_RMSE": float(response['error_arima']),
            "image_path": response['path']
                })
    else:
        raise HTTPException(status_code=404, detail="Item not found")
    
@app.get("/regressor_forecast", description="Do Regressor forecasting and Predict next day stock price")
async def regressor_forecasting(quote : str):

    response = PipelineTasks(quote=quote).regressor_model()
    if response:
        return JSONResponse({
            "Tomorrow's  AAPL  Closing Price Prediction by Regressor Model" : float(response['lr_pred']),
            "Arima_RMSE": float(response['error_lr']),
            "image_path": response['path']
                })
    else:
        raise HTTPException(status_code=404, detail="Item not found")
    


@app.post("/sentiment_analyzer", description="Show sentiment analysis of the stock symbol according to latest news")
async def sentiment_analyzer(quote : str):
    df = PipelineTasks(quote=quote).data_generation()
    response = PipelineTasks(quote=quote).sentiment_analyze_task(df=df)
    if response:
        return response
    else:
        raise HTTPException(status_code=404, detail="Item not found")
    

@app.post("/download_company_data", description="Download Company latest/current :- stock,ESG-RATING,news data")
async def data_downloader(quote : str):

    response = PipelineTasks(quote=quote).download_company_data()
    if response:
        return response
    else:
        raise HTTPException(status_code=404, detail="Item not found")
    

@app.post("/dummy_api", description="Dummy API")
async def dummy():

    return 'HELLO HWHWHWHWH'

      



