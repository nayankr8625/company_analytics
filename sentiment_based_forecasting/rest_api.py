from sentiment_based_forecasting.data_processing import download_tickers
from sentiment_based_forecasting.ml_models import MLModels
from sentiment_based_forecasting.pipeline import PipelineTasks

import uvicorn
import pandas as pd
from pandas import DataFrame
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse, HTMLResponse, Response
from pydantic import BaseModel


app = FastAPI()


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
    
@app.post("/regressor_forecast", description="Do Regressor forecasting and Predict next day stock price")
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
    



