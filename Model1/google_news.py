import pandas as pd
from pygooglenews import GoogleNews
import datetime

gn = GoogleNews()

def get_news(search):
    stories = []
    start_date = datetime.date(2021,3,1)
    end_date = datetime.date(2021,3,5)
    delta = datetime.timedelta(days=1)
    date_list = pd.date_range(start_date, end_date).tolist()
    # print(date_list)

    # result = gn.search(search, from_='2021-03-01', to_='2021-03-02')
    # for item in result['entries']:
    #     story = {
    #         'title': item.title,
    #         'link': item.link,
    #         'published': item.published
    #     }
    #     stories.append(story)
    for date in date_list[:-1]:
        result = gn.search(search, from_=date.strftime('%Y-%m-%d'), to_=(date+delta).strftime('%Y-%m-%d'))
        newsitem = result['entries']

        for item in newsitem:
            story = {
                'title':item.title,
                'link':item.link,
                'published':item.published
            }
            stories.append(story)

    return stories

print(get_news('reliance'))