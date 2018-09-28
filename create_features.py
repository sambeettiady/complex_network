import pandas as pd
import gensim
import spacy
import textblob
import enchant
from sklearn.feature_extraction.text import CountVectorizer
from gensim.parsing.preprocessing import STOPWORDS
from sklearn.model_selection import train_test_split

os.chdir('/Users/sambeet/Desktop/cn/')

dataset1 = pd.read_csv('dataset_q2.csv',dtype={'tweet_id':object})
dataset1['cleaned_text'] = dataset1.cleaned_text.astype(str)
dataset1.drop_duplicates(subset='cleaned_text',inplace=True)

#Collocations for Topic Model
def tokenize(text):
    return [token for token in gensim.utils.simple_preprocess(gensim.parsing.preprocessing.strip_non_alphanum(gensim.parsing.preprocessing.strip_punctuation(gensim.parsing.preprocessing.strip_multiple_whitespaces(text.decode('utf-8','ignore'))))) if token not in STOPWORDS and token != "rt"]

sentence_stream = [tokenize(tweet) for tweet in dataset1.cleaned_text]

phrases = gensim.models.phrases.Phrases(sentence_stream)
bigram = gensim.models.phrases.Phraser(phrases)

bigrams = list(bigram[sentence_stream])

#LDA features
lda_words = pd.read_csv('lda_words_new.csv',header=None)
lda_words.columns = ['lda_words']
lda_words = list(lda_words.lda_words)
corpus = [' '.join(tweet) for tweet in bigrams]
vectorizer = CountVectorizer(vocabulary=lda_words)
lda_features = vectorizer.fit_transform(corpus).toarray()

#POS ratio
nlp = spacy.load('en')

def count_noun(x):
    doc = nlp(x.decode('utf-8','ignore'))
    count = 0
    for token in doc:
        if token.pos_ in ['NOUN','PROPN']:
            count = count + 1
    return count

def count_adverb(x):
    doc = nlp(x.decode('utf-8','ignore'))
    count = 0
    for token in doc:
        if token.pos_ == 'ADV':
            count = count + 1
    return count

def count_adjectives(x):
    doc = nlp(x.decode('utf-8','ignore'))
    count = 0
    for token in doc:
        if token.pos_ == 'ADJ':
            count = count + 1
    return count

dataset1['count_of_nouns'] = dataset1.cleaned_text.apply(count_noun)
dataset1['count_of_adverbs'] = dataset1.cleaned_text.apply(count_adverb)
dataset1['count_of_adjectives'] = dataset1.cleaned_text.apply(count_adjectives)
dataset1['prop_of_nouns'] = dataset1.count_of_nouns/dataset1.tweet_length
dataset1['prop_of_adverbs'] = dataset1.count_of_adverbs/dataset1.tweet_length
dataset1['prop_of_adjectives'] = dataset1.count_of_adjectives/dataset1.tweet_length

dataset1.drop(['count_of_nouns','count_of_adverbs','count_of_adjectives'],inplace=True,axis=1)
dataset1.to_csv('dq1_int.csv',index=False)

#Sentiment Ratio
def get_positive_sentiment_ratio(x):
    t = textblob.TextBlob(x.decode('utf-8','ignore'))
    count = 0.
    for token in t.sentiment_assessments[2]:
        if token[1] > 0:
            count = count + 1
    return count

def get_negative_sentiment_ratio(x):
    t = textblob.TextBlob(x.decode('utf-8','ignore'))
    count = 0.
    for token in t.sentiment_assessments[2]:
        if token[1] < 0:
            count = count + 1
    return count

dataset1['num_pos_words'] = dataset1.cleaned_text.apply(get_positive_sentiment_ratio)
dataset1['num_neg_words'] = dataset1.cleaned_text.apply(get_negative_sentiment_ratio)

dataset1['prop_of_pos_sentiment'] = dataset1.num_pos_words/dataset1.tweet_length
dataset1['prop_of_neg_sentiment'] = dataset1.num_neg_words/dataset1.tweet_length

dataset1.drop(['num_pos_words','num_neg_words'],inplace=True,axis=1)

#Number of Named Entity Mentions
def count_named_entities(x):
    doc = nlp(x.decode('utf-8','ignore'))
    return len(doc.ents)

dataset1['count_of_entities'] = dataset1.cleaned_text.apply(count_named_entities)

#Number of Capitalised Characters
def count_capitalised_characters(x):
    count = 0
    for char in x:
        if char != char.lower():
            count = count + 1
    return count

dataset1['count_of_capitalised_chars'] = dataset1.cleaned_text.apply(count_capitalised_characters)

#Number of Spelling Mistakes
spell_checker = enchant.Dict("en_US")
def count_spelling_mistakes(x):
    bow = tokenize(x)
    count = 0
    for word in bow:
        if spell_checker.check(word) != True:
            count = count + 1
    return count

dataset1['count_of_spelling_mistakes'] = dataset1.cleaned_text.apply(count_spelling_mistakes)

#Presence of Repeated Characters
def check_repeated_characters(x):
    text = x.decode('utf-8','ignore')
    character_set = [letter for letter in set(text) if letter.isalpha()]
    for letter in character_set:
        if letter*3 in text:
            return 1
    return 0

dataset1['repeated_characters_present'] = dataset1.cleaned_text.apply(check_repeated_characters)

dataset1.to_csv('q1_features.csv',index=False)
#Discourse Connectors - Done manually in the paper and hence ignored for now
#Build Bag of words representation
corpus = dataset1.cleaned_text
bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=3,decode_error='ignore',stop_words='english',strip_accents='unicode')
bigram_features = bigram_vectorizer.fit_transform(corpus).toarray()

#Labels
y = dataset1.drunk.values

#Stylistic features
other_stylistic_features = dataset1[['count_of_emoticons','prop_of_nouns', 'prop_of_adverbs',
       'prop_of_adjectives', 'prop_of_pos_sentiment',
       'prop_of_neg_sentiment', 'count_of_entities',
       'count_of_capitalised_chars', 'count_of_spelling_mistakes',
       'repeated_characters_present']].values

stylistic_features = np.concatenate([lda_features,other_stylistic_features],axis = 1)

#All features
all_features = np.concatenate([bigram_features,stylistic_features],axis = 1)

#Split into test and train
all_features_train, all_features_test, bigram_features_train, bigram_features_test, stylistic_features_train, stylistic_features_test, y_train, y_test = train_test_split(all_features,bigram_features,stylistic_features,y,test_size=0.2,random_state = 37)

np.savetxt('labels_train.txt', y_train, fmt='%d')
np.savetxt('all_features_train.txt', all_features_train, fmt='%d')
np.savetxt('bigram_features_train.txt', bigram_features_train, fmt='%d')
np.savetxt('stylistic_features_train.txt', stylistic_features_train, fmt='%d')

np.savetxt('labels_test.txt', y_test, fmt='%d')
np.savetxt('all_features_test.txt', all_features_test, fmt='%d')
np.savetxt('bigram_features_test.txt', bigram_features_test, fmt='%d')
np.savetxt('stylistic_features_test.txt', stylistic_features_test, fmt='%d')
