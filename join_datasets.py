import pandas as pd
import numpy as np
import os

os.chdir('/home/sambeet/data/cn project - drunk texting/')

#Read datasets
q1 = pd.read_table('alcohol-labeled-tweets-icwsm-2016-Q1.tsv',dtype={'id': object,'Q1': np.int8})
q1.columns = ['tweet_id','q1']

q2 = pd.read_table('alcohol-labeled-tweets-icwsm-2016-Q2.tsv',dtype={'id': object,'Q2': np.int8})
q2.columns = ['tweet_id','q2']

q3 = pd.read_table('alcohol-labeled-tweets-icwsm-2016-Q3.tsv',dtype={'id': object,'Q3': np.int8})
q3.columns = ['tweet_id','q3']

tweets_data = pd.read_csv('tweets_dataset.csv',dtype={'tweet_id': object,'text': object})

#Inner Join datasets
q1_data = q1.merge(right=tweets_data,on='tweet_id',how='inner')
q2_data = q2.merge(right=tweets_data,on='tweet_id',how='inner')
q3_data = q3.merge(right=tweets_data,on='tweet_id',how='inner')

q1_data.to_csv('q1_data.csv',index=False)
q2_data.to_csv('q2_data.csv',index=False)
q3_data.to_csv('q3_data.csv',index=False)
