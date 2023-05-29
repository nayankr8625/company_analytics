from sentiment_based_forecasting.rest_api import app
from sentiment_based_forecasting import rest_api
import uvicorn

# import logging

# # Disable logging from Selenium
# logging.disable(logging.CRITICAL)


if __name__ == '__main__':
    uvicorn.run("sentiment_based_forecasting.rest_api:app", host='127.0.0.1', port=8000, reload=True)