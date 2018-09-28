import pandas as pd
import gensim
import spacy
import textblob
import enchant
from sklearn.feature_extraction.text import CountVectorizer
from gensim.parsing.preprocessing import STOPWORDS
from sklearn.model_selection import train_test_split

os.chdir('/Users/sambeet/Desktop/cn/')

dataset1 = pd.read_csv('dataset_q1.csv',dtype={'tweet_id':object})
dataset1.drop_duplicates(subset='cleaned_text',inplace=True)
dataset1 = dataset1[dataset1.drunk == 1]
dataset1.columns
dataset1.drop(['tweet_id','screen_name','drunk'],axis=1,inplace=True)

dataset2 = pd.read_csv('dq2_processed.csv',dtype={'tweet_id':object})
dataset2.drop_duplicates(subset='cleaned_text',inplace=True)
dataset2.columns
dataset2.drop(['user'],axis=1,inplace=True)

dataset2 = pd.concat([dataset2,dataset1],axis=0)
dataset2.drop_duplicates(subset='cleaned_text',inplace=True)
mylist =['#drunk','#imdrunk','#drank']
pattern = '|'.join(mylist)
dataset2['drunk'] = dataset2.text.str.contains(pattern,case=False).fillna(False).astype('int')
dataset2.to_csv('dataset_q2.csv',index=True)
