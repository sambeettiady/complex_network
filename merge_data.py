import pandas as pd
from sklearn.model_selection import train_test_split

os.chdir('/Users/sambeet/Desktop/cn/')

dataset1 = pd.read_csv('cn_cleaned_dataset_2.csv',dtype={'tweet_id':object})
dataset1.drop_duplicates(subset='cleaned_text',inplace=True)
dataset1.columns

dataset1_tanny = pd.read_csv('cn_cleaned_dataset_tanny.csv',dtype={'tweet_id':object})
dataset1_tanny.drop_duplicates(subset='cleaned_text',inplace=True)
dataset1_tanny.columns
dataset1_tanny.drop(['Unnamed: 7', 'Unnamed: 8'],axis=1,inplace=True)

dataset_q1 = pd.concat([dataset1,dataset1_tanny],axis=0)
dataset_q1.shape
dataset_q1.columns

dataset_q1.to_csv('dataset_q1.csv',index=False)
