# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 16:29:24 2021

@author: amit saini

Spam Detection Project
Dataset taken from UCI website: https://archive.ics.uci.edu/ml/datasets/sms+spam+collection

"""

#Import the libraries
import pandas as pd
import re
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

#Load the data
messages=pd.read_csv('SMSSpamCollection', sep='\t',names=['Label','Message'])

#Data Preprocessing
ps=PorterStemmer()
wordnet=WordNetLemmatizer()
corpus_stem=[]
corpus_lemm=[]

for i in range(0,len(messages)):
    review=re.sub('^a-zA-Z', ' ', messages['Message'][i])
    review=review.lower()
    review=review.split()
    review=[ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus_stem.append(review)
    
for i in range(0,len(messages)):
    review=re.sub('^a-zA-Z', ' ', messages['Message'][i])
    review=review.lower()
    review=review.split()
    review=[wordnet.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus_lemm.append(review)
    
#Creating Bag of Words
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=2500)
X1=cv.fit_transform(corpus_stem).toarray()

#Creating TF-IDV
from sklearn.feature_extraction.text import TfidfVectorizer
tf=TfidfVectorizer(max_features=2500)
X2=tf.fit_transform(corpus_stem).toarray()

#One hot encoding
y=pd.get_dummies(messages['Label'],drop_first=True)

#Train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X1, y,test_size=0.2, random_state=0)

#Training model used is Naive bayes classifier
from sklearn.naive_bayes import MultinomialNB
spam_detect_model=MultinomialNB().fit(X_train,y_train)
y_pred=spam_detect_model.predict(X_test)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_mat=confusion_matrix(y_test, y_pred)

#Accuracy
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test, y_pred)

#In case of Stemming
#98.924% accuracy with bag of words method
#98.296% accuracy with TF-IDF method

#In case of Lemmatization
#98.565% accuracy with bag of words method
#98.386% accuracy with TF-IDF method

#So in this case Stemming with bag of words method is giving the highest accuracy

