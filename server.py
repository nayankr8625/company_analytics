from sentiment_based_forecasting.rest_api import app
import uvicorn

uvicorn.run(app,host='127.0.0.1',port=8080)