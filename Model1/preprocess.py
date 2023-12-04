import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import requests
import datetime
import time
from transformers import pipeline
from pygooglenews import GoogleNews

specific_model = pipeline("text-classification", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

# https://twitter.com/search?q=reliance%20stock%20until%3A2023-09-24%20since%3A2023-09-17&prefetchTimestamp=1698217492645&f=media
ROWS = 940
MONTHS = {
    'Jan': '01',
    'Feb': '02',
    'Mar': '03',
    'Apr': '04',
    'May': '05',
    'Jun': '06',
    'Jul': '07',
    'Aug': '08',
    'Sep': '09',
    'Oct': '10',
    'Nov': '11',
    'Dec': '12'
}

gn = GoogleNews()

def score(model_resuts):
    ret = []
    for result in model_resuts:
        if result['label'] == 'negative':
            ret.append(str(1-result['score']))
        elif result['label'] == 'neutral':
            ret.append(str(0.5 - (result['score']/ 2)))
        else:
            ret.append(str(result['score']))
    return " ".join(ret)

def get_news(search, since, until, cnt):
    stories = []
    result = gn.search(search, from_=since.strftime('%Y-%m-%d'), to_=until.strftime('%Y-%m-%d'))
    newsitem = result['entries']

    for item in newsitem:
        stories.append(item.title)
    print(f"#{cnt}: {len(newsitem)} entries found")
    return score(specific_model(stories))

def getStock():
    dataset = pd.read_csv('reliance_weekly.csv')
    price = dataset.iloc[1:, 2]
    date = dataset.iloc[1:, 1]
    return list(price), list(date)

def newsonDate(since, until, search, cnt):
    URL = f'https://timesofindia.indiatimes.com/topic/{search}?dateFilter={until},{since}'
    response = requests.get(URL)
    print(f"#{cnt}: {URL}")
    soup = BeautifulSoup(response.content, 'html.parser')
    divs = soup.find_all('div', attrs={ 'class': 'fHv_i o58kM' })
    # print(f"#{cnt}: {len(divs)} news found")
    return score(specific_model([div.text for div in divs]))

def main():
    price, dates = getStock()
    price = list(reversed(price[:ROWS]))
    dates = list(reversed(dates[:ROWS]))
    date = []
    for d in dates:
        [month, dt, year] = d.replace(',', '').split(' ')
        date.append(datetime.date(int(year),int(MONTHS[month]),int(dt)))
    print(f"{len(price)} datasets generated. Getting news!")
    pd.DataFrame([[price[i], get_news('Reliance-Industries', date[i-1], date[i], i)] for i in range(1, len(price))]).to_csv('weekly_data.csv')

if __name__ == "__main__":
    main()