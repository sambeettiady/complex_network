df = pd.DataFrame(columns=['user','tweet'])

x = pd.read_csv('/Users/sambeet/Downloads/cn_cleaned_dataset (1).csv')
x = pd.read_csv('cn_cleaned_dataset_2.csv')
x.drop_duplicates(subset='cleaned_text',inplace=True)
drunk_userids = list(np.unique(x.screen_name[x.drunk == 1]))

for user in drunk_userids:
    try:
        with open('user_tweets_1/' + user + '.txt','r') as ut:
            content = ut.readlines()
            content = [x.strip() for x in content] 
        temp = pd.DataFrame({'user':[user]*len(content),'tweet':content})
        df = pd.concat([df,temp],axis=0)
    except:
        print 'Error!'
        continue

df.to_csv('dq2.csv',index=False)

import pandas as pd
import emot
import re

dataset1 = df.copy()
dataset1.columns = ['text','user']
#Detect emoticons - presence and counts
detect_emojis_and_emoticons = lambda x: len(emot.emoji(x)) + len(emot.emoticons(x))
dataset1['count_of_emoticons'] = dataset1.text.apply(detect_emojis_and_emoticons)

#Detect and clean non-unicode characters
clean_non_unicode = lambda x: x.replace(r'[^\x00-\x7F]+',' ').strip().lower()
dataset1['cleaned_text'] = dataset1.text.apply(clean_non_unicode)

#Detect and clean links
clean_links = lambda x: re.sub(r'http\S+', ' ', x).strip()
dataset1['cleaned_text'] = dataset1.cleaned_text.apply(clean_links)

#Clean hashtags
hashtags = ['#drunk','#imdrunk','#drank','#sober','#notdrunk','#imnotdrunk']
hashtags_regex = re.compile('|'.join(map(re.escape,hashtags)))
clean_hashtags = lambda x: hashtags_regex.sub(' ',x)
dataset1['cleaned_text'] = dataset1.cleaned_text.apply(clean_hashtags)

#Remove punctuations and extra whitespaces
clean_punctuations = lambda x: re.sub(r'[^\w\s]',' ',x)
dataset1['cleaned_text'] = dataset1.cleaned_text.apply(clean_punctuations)
pattern = re.compile(r'\s+')
clean_ws = lambda x: re.sub(pattern,' ',x.strip())
dataset1['cleaned_text'] = dataset1.cleaned_text.apply(clean_ws)

#Detect and clean tweets less than 6 words in length
num_words = lambda x: len(re.findall(r'\w+', x))
dataset1['tweet_length'] = dataset1.cleaned_text.apply(num_words)
dataset1 = dataset1[dataset1.tweet_length >= 6]

#Save file
dataset1.to_csv('drunk_tweets_cleaned.csv',index = False)
