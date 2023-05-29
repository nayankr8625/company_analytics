import yfinance as yf
from datetime import datetime
from alpha_vantage.timeseries import TimeSeries

import requests
import newspaper
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import time

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver import Chrome

from services import logger

import pandas as pd
import openai



class download_tickers:

    def __init__(self,tickers):
        self._tickers = tickers

    def get_historical_yf(self):
        try:
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
            logger.debug(f'DOWNLOADED {self._tickers} STOCK DATA')
            return df
        except:
            logger.debug('Enter a stock Symbol')
    
    def get_historical_other_api(self):
        pass


    def scrape_company_esg_data(self):

        # creating web driver 
        service = Service('chrome_driver/chromedriver.exe')
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        driver = Chrome(service=service, options=options)
        stock_symbol = self._tickers
        driver.set_page_load_timeout(10)

        # Opening webpage
        logger.debug(f'Opening the sustainablity rating web page')
        driver.get("https://www.sustainalytics.com/esg-rating")
        time.sleep(1)
        
        # waiting for the page to load
        # Searchig for stock symbol 
        tickers = driver.find_element(By.ID, "searchInput")

        # Enter Your Stock Symbol

        for i in stock_symbol:

            tickers.send_keys(i)
            time.sleep(0.5)
        
        time.sleep(2)
        logger.debug(f'Opening the {self._tickers} link')
        driver.find_element(By.XPATH, "//a[@class='search-link js-fix-path']").click()
        # In case of an error, try changing the
        # XPath used here.

        src = driver.page_source

        # Extract the HTML content
        html_content = src
        

        # Parse the HTML with BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')

        # Find the company name
        company_name = soup.find('div', class_='company-name').h2.text.strip()

        # Find the industry group
        industry_group = soup.find('strong', class_='industry-group').text.strip()

        # Find the country/region
        country = soup.find('strong', class_='country').text.strip()

        # Find the identifier
        identifier = soup.find('strong', class_='identifier').text.strip()

        # Find the company description
        company_description = soup.find('div', class_='company-description-text').find('span').text.strip()

        # Find the full-time employees
        full_time_employees = soup.find('div', class_='company-description-details').strong.text.strip()

        # Find the ESG risk rating
        esg_risk_rating = soup.find('h3', text='ESG Risk Rating').find_next('span', class_='').text.strip()
        esg_rating_assessment = soup.find('div', class_='risk-rating-assessment').span.text.strip()

        # Find the industry group ranking
        industry_group_ranking = soup.find('p', class_='intro-ranking-heading', text='Industry Group (1st = lowest risk)')
        industry_group_position = industry_group_ranking.find_next('strong', class_='industry-group-position').text.strip()
        industry_group_positions_total = industry_group_ranking.find_next('span', class_='industry-group-positions-total').text.strip()
        industry_group_ranking_formatted = f"{industry_group_position}/{industry_group_positions_total}"

        # Find the universe ranking
        universe_ranking = soup.find('p', class_='intro-ranking-heading', text='Universe')
        universe_position = universe_ranking.find_next('strong', class_='universe-position').text.strip()
        universe_positions_total = universe_ranking.find_next('span', class_='universe-positions-total').text.strip()
        universe_ranking_formatted = f"{universe_position}/{universe_positions_total}"

        # Create a dictionary with the scraped data
        data = {
            'Company Name': [company_name],
            'Industry Group': [industry_group],
            'Country/Region': [country],
            'Identifier': [identifier],
            'Company Description': [company_description],
            'Full Time Employees': [full_time_employees],
            'ESG Risk Rating': [esg_risk_rating],
            'ESG Risk Rating Assessment': [esg_rating_assessment],
            'Industry Group Ranking': [industry_group_ranking_formatted],
            'Universe Ranking': [universe_ranking_formatted]
        }

        json_data = {
            'Company Name': company_name,
            'Industry Group': industry_group,
            'Country/Region': country,
            'Identifier': identifier,
            'Company Description': company_description,
            'Full Time Employees': full_time_employees,
            'ESG Risk Rating': esg_risk_rating,
            'ESG Risk Rating Assessment': esg_rating_assessment,
            'Industry Group Ranking': industry_group_ranking_formatted,
            'Universe Ranking': universe_ranking_formatted
        }

        # Create a Pandas DataFrame from the dictionary
        df = pd.DataFrame(data)
        logger.debug(f'ESG DATA EXTRACTION OF {self._tickers} COMPLETED')

        # Return the DataFrame
        return df,json_data
    

    def get_indepth_news(self,url):
        article = newspaper.Article(url)
        article.download()
        article.parse()
        content = article.text
        return content

    def news_api_stock_news(self):
        api_key='212e34fad4074c8cb491adc643451684'
        stock_symbol = self._tickers
        # Calculate the date 5 days ago from today
        date_5_days_ago = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")

        url = f"https://newsapi.org/v2/everything?q={stock_symbol}&from={date_5_days_ago}&apiKey={api_key}"

        try:
            response = requests.get(url)
            data = response.json()

            # Check if the API call was successful
            if response.status_code == 200:
                articles = data["articles"]

                # Create an empty list to store the news data
                news_list = []

                # Define relevant keywords
                keywords = [
                    "Stock market",
                    stock_symbol,
                    "Financial markets",
                    "Investing",
                    "Stocks",
                    "Company earnings",
                    "Market trends",
                    "Economic indicators",
                    "Stock analysis",
                    "Corporate news",
                    "Market performance",
                    "Stock prices",
                    "Market volatility",
                    "Market outlook",
                    "Industry updates",
                    "Market movers"
                ]  # Add more relevant keywords here

                # Iterate over the articles and filter relevant news
                for article in articles:
                    title = article["title"]
                    source = article["source"]["name"]
                    published_date = article["publishedAt"]
                    url = article["url"]

                    # Get the detailed news content
                    try:
                        content = self.get_indepth_news(url)
                    except newspaper.ArticleException as e:
                        print(f"Error fetching article: {e}")
                        content = title

                    # Check if any keyword is present in the title or content
                    if any(keyword.lower() in title.lower() or keyword.lower() in content.lower() for keyword in keywords):
                        news_list.append({
                            "Title": title,
                            "Source": source,
                            "Published Date": published_date,
                            "URL": url,
                            "Content": content,
                        })

                # Create a DataFrame from the list of dictionaries
                news_df = pd.DataFrame(news_list)
                logger.debug(f'DOWNLOADING NEWS FOR {self._tickers} COMPLETED')
                return news_df

            else:
                print(f"Error: {data['message']}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            return None
 



