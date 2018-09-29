import tweepy
import pandas as pd
import os

os.chdir('/Users/sambeet/Downloads/icwsm-2016-data-release/')

data1 = pd.read_table('alcohol-labeled-tweets-icwsm-2016-Q1.tsv',dtype={'id': object,'q1': np.int8})
data2 = pd.read_table('alcohol-labeled-tweets-icwsm-2016-Q2.tsv',dtype={'id': object,'q1': np.int8})
data3 = pd.read_table('alcohol-labeled-tweets-icwsm-2016-Q3.tsv',dtype={'id': object,'q1': np.int8})
data1.columns = ['tweet_id','q']
data2.columns = ['tweet_id','q']
data3.columns = ['tweet_id','q']

q1 = [i for i in data1.tweet_id]
q2 = [i for i in data2.tweet_id]
q3 = [i for i in data3.tweet_id]

tweet_IDs = list(set().union(*[q1,q2,q3]))

CONSUMER_KEY = ''
CONSUMER_SECRET = ''
OAUTH_TOKEN = ''
OAUTH_TOKEN_SECRET = ''

auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

full_tweets = pd.DataFrame(columns=['tweet_id','text'])
tweet_count = len(tweet_IDs)

for i in range((tweet_count/100) + 1):
    print i
    end_loc = min((i + 1) * 100, tweet_count)
    tweets = api.statuses_lookup(id_=tweet_IDs[(i*100):end_loc])
    for tweet in tweets:
        full_tweets = full_tweets.append(pd.Series([tweet.id_str, tweet.text.encode('utf-8').strip()], index=['tweet_id','text']),ignore_index=True)

full_tweets.to_csv('tweets_dataset.csv',index=False)
