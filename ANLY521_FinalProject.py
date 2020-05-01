import pandas as pd, numpy as np, re, time
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc



# Loading data from json file
data = pd.read_json('Sarcasm_Headlines_Dataset.json', lines = True)

'''
Exploratory data analysis
'''
#Check NA
data.isnull().any(axis = 0)
#data.head()

#Check Label Distribution
import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(data['is_sarcastic'])

#WordCloud of Sarcastic and Non-Sarcastic records
from wordcloud import WordCloud, STOPWORDS
wordcloud_sar = WordCloud(background_color='black', stopwords = STOPWORDS,
                max_words = 100, max_font_size = 100, 
                random_state = 15, width=800, height=400)
plt.figure(figsize=(16, 12))
wordcloud_sar.generate(str(data.loc[data['is_sarcastic'] == 1, 'headline']))
plt.imshow(wordcloud_sar)

wordcloud_nonsar = WordCloud(background_color='white', stopwords = STOPWORDS,
                max_words = 100, max_font_size = 100, 
                random_state = 15, width=800, height=400)
plt.figure(figsize=(16, 12))
wordcloud_nonsar.generate(str(data.loc[data['is_sarcastic'] == 0, 'headline']))
plt.imshow(wordcloud_nonsar)

'''
1. Classic Machine Learning Models
    try 4 models: Linear SVM, Gaussian Naive Bayes, Logistic Regression, Random Forest
'''
#Pre-Processing Text
data['headline'] = data['headline'].apply(lambda s : re.sub('[^a-zA-Z]', ' ', s))
# getting features and labels
features = data['headline']
labels = data['is_sarcastic']
# Stemming our data
ps = PorterStemmer()
features = features.apply(lambda x: x.split())
features = features.apply(lambda x : ' '.join([ps.stem(word) for word in x]))

# TF-IDF Vectorize
# vectorizing the data with maximum of 5000 features
from sklearn.feature_extraction.text import TfidfVectorizer
tv = TfidfVectorizer(max_features = 5000)
features = list(features)
features = tv.fit_transform(features).toarray()

#Split Data
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = .25, random_state = 0)

'''
1.1 Linear SVM
'''
lsvc = LinearSVC()
# training the model
lsvc.fit(features_train, labels_train)
lsvc_pred_train = lsvc .predict(features_train)
lsvc_pred = lsvc .predict(features_test)

#metrics
print('lsvc training data accuracy: ',lsvc.score(features_train, labels_train)) 
print('lsvc testing data accuracy: ',lsvc.score(features_test, labels_test))  

print('lsvc training Confusion matrix\n',confusion_matrix(labels_train,lsvc_pred_train))
print('lsvc testing Confusion matrix\n',confusion_matrix(labels_test,lsvc_pred))

print('lsvc training Classification_report\n',classification_report(labels_train,lsvc_pred_train))
print('lsvc testing Classification_report\n',classification_report(labels_test,lsvc_pred))

'''
1.2 Gaussian Naive Bayes
'''
gnb = GaussianNB()
gnb.fit(features_train, labels_train)
gnb_pred_train = gnb .predict(features_train)
gnb_pred = gnb .predict(features_test)

#metrics
print('gnb training data accuracy: ',gnb.score(features_train, labels_train)) 
print('gnb testing data accuracy: ',gnb.score(features_test, labels_test))  

print('gnb training Confusion matrix\n',confusion_matrix(labels_train,gnb_pred_train))
print('gnb testing Confusion matrix\n',confusion_matrix(labels_test,gnb_pred))

print('gnb training Classification_report\n',classification_report(labels_train,gnb_pred_train))
print('gnb testing Classification_report\n',classification_report(labels_test,gnb_pred))

'''
1.3 Logistic Regression
'''
lr = LogisticRegression()
lr.fit(features_train, labels_train)
lr_pred_train = lr .predict(features_train)
lr_pred = lr .predict(features_test)

#metrics
print('lr training data accuracy: ',lr.score(features_train, labels_train)) 
print('lr testing data accuracy: ',lr.score(features_test, labels_test))  

print('lr training Confusion matrix\n',confusion_matrix(labels_train,lr_pred_train))
print('lr testing Confusion matrix\n',confusion_matrix(labels_test,lr_pred))

print('lr training Classification_report\n',classification_report(labels_train,lr_pred_train))
print('lr testing Classification_report\n',classification_report(labels_test,lr_pred))

'''
1.4 Random Forest
'''
rfc = RandomForestClassifier(n_estimators = 10, random_state = 0)
rfc.fit(features_train, labels_train)
rfc_pred_train = rfc .predict(features_train)
rfc_pred = rfc .predict(features_test)

#metrics
print('rfc training data accuracy: ',rfc.score(features_train, labels_train)) 
print('rfc testing data accuracy: ',rfc.score(features_test, labels_test))  

print('rfc training Confusion matrix\n',confusion_matrix(labels_train,rfc_pred_train))
print('rfc testing Confusion matrix\n',confusion_matrix(labels_test,rfc_pred))

print('rfc training Classification_report\n',classification_report(labels_train,rfc_pred_train))
print('rfc testing Classification_report\n',classification_report(labels_test,rfc_pred))

'''
2. LSTM with Different Word Embedding
'''
#Plese read the README before running this part

'''
2.1 LSTM with GloVe
'''
#You need glove.6B.200d.txt, this file has to be downloaded via this link http://nlp.stanford.edu/data/glove.6B.zip After you unzip it, please find glove.6B.200d.txt and put it in your directory.
import pandas as pd
df = pd.read_json("Sarcasm_Headlines_Dataset.json", lines=True)
#df.head()

df = df.drop(['article_link'], axis=1)
df['len'] = df['headline'].apply(lambda x: len(x.split(" ")))
#df.head()

#Pre-Processing Text
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Flatten, Bidirectional, GlobalMaxPool1D
from tensorflow.keras.models import Model, Sequential
import tensorflow as tf

max_features = 10000
maxlen = 25
embedding_size = 200

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(df['headline']))
X = tokenizer.texts_to_sequences(df['headline'])
X = pad_sequences(X, maxlen = maxlen)
y = df['is_sarcastic']

#Glove Embeddings
EMBEDDING_FILE = 'glove.6B.200d.txt'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE,encoding='utf8') if len(o)>100)

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))

embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embedding_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

#train and fit model    
y = np.asarray(y)
model = Sequential()
model.add(Embedding(max_features, embedding_size, weights = [embedding_matrix]))
model.add(Bidirectional(LSTM(128, return_sequences = True)))
model.add(GlobalMaxPool1D())
model.add(Dense(40, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(20, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', 
              metrics=['accuracy', tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])

batch_size = 100
epochs = 5
history = model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=0.2)

#Plot of Performance
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(15,5))
fig.suptitle("Performance of Glove Vectors")
ax1.plot(history.history['accuracy'])
ax1.plot(history.history['val_accuracy'])
vline_cut = np.where(history.history['val_accuracy'] == np.max(history.history['val_accuracy']))[0][0]
ax1.axvline(x=vline_cut, color='k', linestyle='--')
ax1.set_title("Model Accuracy")
ax1.legend(['train', 'test'])

ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
vline_cut = np.where(history.history['val_loss'] == np.min(history.history['val_loss']))[0][0]
ax2.axvline(x=vline_cut, color='k', linestyle='--')
ax2.set_title("Model Loss")
ax2.legend(['train', 'test'])
plt.show()

'''
2.2 LSTM with FastText
'''
#Pretrained FastText word embedding file wiki-news-300d-1M.vec has to be downloaded from: https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip
#Please do not forget to unzip the downloaded file, and place it under your directory

#FastTest Embedding
EMBEDDING_FILE = 'wiki-news-300d-1M.vec'
def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE,encoding='utf8'))
embed_size = 300
word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    
#Train and fit model
max_features = 10000
embedding_size = 300

model2 = Sequential()
model2.add(Embedding(max_features, embedding_size))
model2.add(Bidirectional(LSTM(128, return_sequences = True)))
model2.add(GlobalMaxPool1D())
model2.add(Dense(40, activation="relu"))
model2.add(Dropout(0.5))
model2.add(Dense(20, activation="relu"))
model2.add(Dropout(0.5))
model2.add(Dense(1, activation="sigmoid"))
model2.compile(loss='binary_crossentropy', optimizer='adam', 
              metrics=['accuracy', tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])

batch_size = 100
epochs = 5
history2 = model2.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=0.2)

#Plot of Performance
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(15,5))
fig.suptitle("Performance of Model with Fasttext embeddings")
ax1.plot(history2.history['accuracy'])
ax1.plot(history2.history['val_accuracy'])
vline_cut = np.where(history2.history['val_accuracy'] == np.max(history2.history['val_accuracy']))[0][0]
ax1.axvline(x=vline_cut, color='k', linestyle='--')
ax1.set_title("Model Accuracy")
ax1.legend(['train', 'test'])

ax2.plot(history2.history['loss'])
ax2.plot(history2.history['val_loss'])
vline_cut = np.where(history2.history['val_loss'] == np.min(history2.history['val_loss']))[0][0]
ax2.axvline(x=vline_cut, color='k', linestyle='--')
ax2.set_title("Model Loss")
ax2.legend(['train', 'test'])
plt.show()

