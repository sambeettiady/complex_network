import tweepy
import csv
import pandas as pd
import os
import sys
import jsonpickle
import json

os.chdir('/Users/sambeet/Desktop/cn/')

consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''

#Switching to application authentication
auth = tweepy.AppAuthHandler(consumer_key, consumer_secret)

#Setting up new api wrapper, using authentication only
api = tweepy.API(auth, wait_on_rate_limit=True,wait_on_rate_limit_notify=True)
 
#Error handling
if (not api):
    print ("Problem Connecting to API")

api.rate_limit_status()['resources']['search']

#Maximum number of tweets we want to collect 
maxTweets = 1000000

#The twitter Search API allows up to 100 tweets per query
tweetsPerQry = 100

tweetCount = 0
searchQuery = '#drunk OR #drank OR #imdrunk'
#Open a text file to save the tweets to
with open('dataset1_drunk_2.json', 'w') as f:
    #Tell the Cursor method that we want to use the Search API (api.search)
    #Also tell Cursor our query, and the maximum number of tweets to return
    for tweet in tweepy.Cursor(api.search,q=searchQuery).items(maxTweets):         
        #Write the JSON format to the text file, and add one to the number of tweets we've collected
        f.write(jsonpickle.encode(tweet._json, unpicklable=False) + '\n')
        tweetCount += 1
    #Display how many tweets we have collected
    print("Downloaded {0} tweets".format(tweetCount))

tweetCount = 0
searchQuery = '#notdrunk OR #imnotdrunk OR #sober'
#Open a text file to save the tweets to
with open('dataset1_sober_2.json', 'w') as f:
    #Tell the Cursor method that we want to use the Search API (api.search)
    #Also tell Cursor our query, and the maximum number of tweets to return
    for tweet in tweepy.Cursor(api.search,q=searchQuery).items(maxTweets):         
        #Write the JSON format to the text file, and add one to the number of tweets we've collected
        f.write(jsonpickle.encode(tweet._json, unpicklable=False) + '\n')
        tweetCount += 1
    #Display how many tweets we have collected
    print("Downloaded {0} tweets".format(tweetCount))

drunk_userids = []
for line in open('dataset1_drunk_2.json', 'r'):
    drunk_userids.append(json.loads(line)['user']['screen_name'])

drunk_userids = list(set(drunk_userids))
already_scraped = os.listdir('user_tweets/')
already_scraped = [screen_name.split('.')[0] for screen_name in already_scraped]

for screen_name in drunk_userids:
    print screen_name
    if screen_name not in already_scraped:
        try:
            print 'scraping tweets!' 
            new_tweets = api.user_timeline(screen_name = screen_name,count=20, tweet_mode="extended")
            with open('user_tweets/' + screen_name + '.txt','w') as ut:
                for tweet in new_tweets:
                    ut.write(tweet.full_text.encode('utf-8') + '\n')
        except:
            print 'Continuing after error!'
            continue

with open('dataset1_drunk_2.csv','w') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(['tweet_id','text','screen_name','drunk'])
    for line in open('dataset1_drunk.json', 'r'):
        x = json.loads(line)
        y = [x['id_str'],x['text'].encode('utf-8'),x['user']['screen_name'],1]
        csv_writer.writerow(y)
    for line in open('dataset1_drunk_2.json', 'r'):
        x = json.loads(line)
        y = [x['id_str'],x['text'].encode('utf-8'),x['user']['screen_name'],1]
        csv_writer.writerow(y)

with open('dataset1_sober_2.csv','w') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(['tweet_id','text','screen_name','drunk'])
    for line in open('dataset1_sober.json', 'r'):
        x = json.loads(line)
        y = [x['id_str'],x['text'].encode('utf-8'),x['user']['screen_name'],0]
        csv_writer.writerow(y)
    for line in open('dataset1_sober_2.json', 'r'):
        x = json.loads(line)
        y = [x['id_str'],x['text'].encode('utf-8'),x['user']['screen_name'],0]
        csv_writer.writerow(y)

x = pd.read_csv('dataset1_drunk_2.csv',dtype={'tweet_id':object})
y = pd.read_csv('dataset1_sober_2.csv',dtype={'tweet_id':object})

z = pd.concat([x,y],axis=0,ignore_index=True)
z.to_csv('dataset1_2.csv',index=False)
