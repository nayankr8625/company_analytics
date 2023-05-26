import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [quote, setQuote] = useState('');
  const [data, setData] = useState('');

  const handleInputChange = (event) => {
    setQuote(event.target.value);
  };

  const showData = () => {
    const url = `http://localhost:8000/show_data?quote=${quote}`;
    console.log('Request URL:', url);

    axios
      .get(url)
      .then((response) => {
        console.log('Response:', response.data);
        setData(response.data);
      })
      .catch((error) => {
        console.error('Error:', error);
      });
  };

  const showESGData = () => {
    const url = `http://localhost:8000/show_esg_data?quote=${quote}`;
    console.log('Request URL:', url);
  
    axios
      .get(url)
      .then((response) => {
        console.log('Response:', response.data);
        const { esg_ratings, csv_file_path } = response.data;
        let dataString = `ESG Ratings: ${JSON.stringify(esg_ratings)}`;
        if (csv_file_path) {
          dataString += `\nCSV File Path: ${csv_file_path}`;
        }
        setData(dataString);
      })
      .catch((error) => {
        console.error('Error:', error);
      });
  };

  const showNewsData = () => {
    const url = `http://localhost:8000/show_news_data?quote=${quote}`;
    console.log('Request URL:', url);

    axios
      .get(url)
      .then((response) => {
        console.log('Response:', response.data);
        setData(response.data);
      })
      .catch((error) => {
        console.error('Error:', error);
      });
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
      </div>
      <div>
        <h2>Response:</h2>
        <pre>{data}</pre>
      </div>
    </div>
  );
}

export default App;
