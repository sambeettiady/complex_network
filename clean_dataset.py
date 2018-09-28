import pandas as pd
import emot
import re

dataset1 = pd.read_csv('dataset1_2.csv',dtype={'tweet_id':object})

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
dataset1.to_csv('cn_cleaned_dataset_2.csv',index = False)
