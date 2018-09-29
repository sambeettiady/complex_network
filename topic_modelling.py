import pandas as pd
import numpy as np
import gensim
import gensim.parsing.preprocessing as gpp
import os
os.chdir('/home/sambeet/data/cn project - drunk texting')

dataset = pd.read_csv('dq2_processed.csv')
dataset.shape
dataset.drop_duplicates(subset='cleaned_text',inplace=True)
dataset.shape

threshold = 20
len(np.unique(dataset.user[dataset.tweet_length >= threshold]))
len((dataset.user[dataset.tweet_length >= threshold]))

corpus = list(dataset.cleaned_text[dataset.tweet_length >= threshold])
len(corpus)

def tokenize(text):
    return [token for token in gensim.utils.simple_preprocess(gpp.strip_non_alphanum(gpp.strip_punctuation(gpp.strip_multiple_whitespaces(gensim.utils.deaccent(text))))) if token not in list(gpp.STOPWORDS) +['rt']  and len(token) >= 4]

sentence_stream = [tokenize(article) for article in corpus]

#LDA
dictionary = gensim.corpora.Dictionary(sentence_stream)

dictionary.filter_extremes(no_below=3, no_above=0.9)
len(dictionary.token2id)
dictionary.compactify()

corpus_lda = [dictionary.doc2bow(text) for text in sentence_stream]
np.random.seed(37)
lda = gensim.models.ldamodel.LdaModel(corpus=corpus_lda, id2word=dictionary, num_topics=10,eta=0.01,update_every=1,passes=20)
lda.show_topics(num_topics=10,num_words=20)
lda.save('drunk_lda_model')

#Predict topic vectors
dataset['cleaned_text'] = dataset.cleaned_text.astype(str)
user_tweets_concatenated = dataset[['user','cleaned_text']].groupby('user')['cleaned_text'].apply(' '.join).reset_index()

bow_vector = [dictionary.doc2bow(tokenize(text)) for text in user_tweets_concatenated.cleaned_text]
lda.minimum_probability = 0.0
lda_vector = lda[bow_vector]
lda_vector = list(lda_vector)
lda_vector = [[topic[1] for topic in td] for td in lda_vector]
td_df = pd.DataFrame(lda_vector,columns=['topic_1','topic_2','topic_3','topic_4','topic_5','topic_6','topic_7','topic_8','topic_9','topic_10'])
td_df = pd.concat([user_tweets_concatenated[['user']],td_df],axis=1)
td_df.to_csv('drunk_user_topic_probabilities.csv',index=False)