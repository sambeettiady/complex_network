import pandas as pd
import numpy as np
import os
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPooling1D
from keras.models import Model
from keras import initializations, regularizers, constraints, optimizers, layers, optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score, roc_auc_score, roc_curve, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Model,load_model
from keras import callbacks
import keras.backend as K
import keras.utils.visualize_util as kvis

#Change working directory and execute helper fuctions
os.chdir('/home/sambeet/data/cn project - drunk texting/')
execfile('helper_functions.py')

#Read datasets
q1 = pd.read_csv('../q2_features.csv',dtype={'tweet_id': object})
q1.drop_duplicates(subset='cleaned_text',inplace=True)
q1 = q1[q1.tweet_length >=6]
q1.shape
#Separate labels and tweets
labels = q1.drunk
#Clean hashtags
hashtags = ['#drunk','#imdrunk','#drank','#sober','#notdrunk','#imnotdrunk']
hashtags_regex = re.compile('|'.join(map(re.escape,hashtags)))
clean_hashtags = lambda x: hashtags_regex.sub(' ',x)
q1['text'] = q1.text.astype(str)
q1['text'] = q1.text.apply(lambda x: x.lower())
features = q1.text.apply(clean_hashtags)

#Split into training and testing
x_train, x_test, y_train, y_test = train_test_split(features,labels,test_size = 0.2,random_state = 37)

#Check for NULL values
x_train.isnull().any(),x_test.isnull().any()

#Character level tokeniser and sequence generator
char_tokeniser = Tokenizer(char_level=True,filters=None)
char_tokeniser.fit_on_texts(x_train)
max_features = len(char_tokeniser.word_index)
training = char_tokeniser.texts_to_sequences(x_train)
testing = char_tokeniser.texts_to_sequences(x_test)

#Checking for distribution of character length of tweets
train_len = [len(tweet) for tweet in training]
test_len = [len(tweet) for tweet in testing]
length_distribution = sns.distplot(train_len)
fig = length_distribution.get_figure()
fig.savefig('length_distribution.jpg') 
sns.distplot(test_len)

#Padding each tweet so that we have a length of 160 characters
maxlen = 160
X_t = pad_sequences(training, maxlen=maxlen)
#X_t = X_t.reshape((3040,160,1))
X_te = pad_sequences(testing, maxlen=maxlen)
#X_te = X_te.reshape((761,160,1))

#####LSTM Model#####
#Define input
inp = Input(shape=(maxlen,),dtype='int32')

#Add embedding layer
embed_size = 128
embed = Embedding(max_features, embed_size)(inp)

#Add LSTM Layer with Global Max Pooling and Dropout
x = LSTM(128,return_sequences=True)(embed)
x = GlobalMaxPooling1D()(x)
x = Dropout(0.5)(x)

#Add Dense Layer with dropout
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)

#Final sigmoid layer for classification
x = Dense(1, activation="sigmoid")(x)

#Compile model
model = Model(input=inp, output=x)
model.compile(loss='binary_crossentropy',optimizer=optimizers.Adam(1e-3),metrics=['accuracy'])

#Check model summary
model.summary()
#kvis.plot_model(model,show_shapes=True,to_file='model_arch.png')

#Define callbacks
model_checkpoint = callbacks.ModelCheckpoint(filepath = 'logs/char_lstm_2.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
#tensorboard = callbacks.TensorBoard(log_dir='logs', histogram_freq=0, batch_size=2, write_graph=True, write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
csv_logger = callbacks.CSVLogger('logs/training.log')

#Fit Model
history = model.fit(x=X_t,y=y_train,nb_epoch=10,verbose=1,validation_data=(X_te,y_test),shuffle=True,batch_size=1024,callbacks = [model_checkpoint,csv_logger])
#history = model.fit(x=X_t,y=y_train,nb_epoch=26,initial_epoch=25,verbose=1,validation_data=(X_te,y_test),shuffle=True,batch_size=32,callbacks = [model_checkpoint,csv_logger])

#Save model weights
model.save('lstm_q1.hd5')
model.load_weights('logs/char_lstm_2.17-0.31.hdf5')#,custom_objects={'dice_loss':dice_loss,'dice_coef':dice_coef})

#Generate and show Training history plot
generate_and_save_training_history_plot(history=history,filename='Dataset1_2 - LSTM2 - Training History')

#Check Accuracy, Precision, Recall, AUC and ROC Curve on Testing Dataset
classfication_report(testing_data=X_te,model=model)
