import React, { useState } from 'react';

function App() {
  const [quote, setQuote] = useState('');
  const [data, setData] = useState('');

  const handleInputChange = (event) => {
    setQuote(event.target.value);
  };

  const showData = () => {
    fetch(`/show_data?quote=${quote}`)
      .then((response) => response.text())
      .then((htmlData) => setData(htmlData))
      .catch((error) => console.log(error));
  };

  const showESGData = () => {
    fetch(`/show_esg_data?quote=${quote}`)
      .then((response) => response.json())
      .then((jsonData) => {
        // Handle the ESG data response
        console.log(jsonData);
      })
      .catch((error) => console.log(error));
  };

  const showNewsData = () => {
    fetch(`/show_news_data?quote=${quote}`)
      .then((response) => response.text())
      .then((htmlData) => setData(htmlData))
      .catch((error) => console.log(error));
  };

  const lstmForecast = () => {
    fetch('/lstm_forecast', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ quote }),
    })
      .then((response) => response.json())
      .then((jsonData) => {
        // Handle the LSTM forecast response
        console.log(jsonData);
      })
      .catch((error) => console.log(error));
  };

  const arimaForecast = () => {
    fetch('/arima_forecasting', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ quote }),
    })
      .then((response) => response.json())
      .then((jsonData) => {
        // Handle the ARIMA forecast response
        console.log(jsonData);
      })
      .catch((error) => console.log(error));
  };

  const regressorForecast = () => {
    fetch('/regressor_forecasting', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ quote }),
    })
      .then((response) => response.json())
      .then((jsonData) => {
        // Handle the Regressor forecast response
        console.log(jsonData);
      })
      .catch((error) => console.log(error));
  };

  const sentimentAnalyzer = () => {
    fetch('/sentiment_analyzer', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ quote }),
    })
      .then((response) => response.json())
      .then((jsonData) => {
        // Handle the sentiment analysis response
        console.log(jsonData);
      })
      .catch((error) => console.log(error));
  };

  return (
    <div>
      <h1>API Usage</h1>
      <div>
        <label htmlFor="quote">Quote: </label>
        <input type="text" id="quote" value={quote} onChange={handleInputChange} />
        <button onClick={showData}>Show Data</button>
        <button onClick={showESGData}>Show ESG Data</button>
        <button onClick={showNewsData}>Show News Data</button>
        <button onClick={lstmForecast}>LSTM Forecast</button>
        <button onClick={arimaForecast}>ARIMA</button>
        </div>
        </div>)}
