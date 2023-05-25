from sentiment_based_forecasting.rest_api import app
import uvicorn

# import logging

# # Disable logging from Selenium
# logging.disable(logging.CRITICAL)



uvicorn.run(app,host='127.0.0.1',port=8080)