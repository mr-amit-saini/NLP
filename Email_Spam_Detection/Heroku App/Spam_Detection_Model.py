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
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pickle

#Load the data
messages=pd.read_csv('SMSSpamCollection', sep='\t',names=['Label','Message'])

#Data Preprocessing
wordnet=WordNetLemmatizer()
corpus_lemm=[]
    
for i in range(0,len(messages)):
    review=re.sub('^a-zA-Z', ' ', messages['Message'][i])
    review=review.lower()
    review=review.split()
    review=[wordnet.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus_lemm.append(review)
    
#Creating TF-IDV
from sklearn.feature_extraction.text import TfidfVectorizer
tf=TfidfVectorizer(max_features=2500)
X=tf.fit_transform(corpus_lemm).toarray()

#Generate words vector file to deploy in heroku platform
pickle.dump(tf, open('vectorized_data.pkl', 'wb'))


#One hot encoding
y=pd.get_dummies(messages['Label'],drop_first=True)

#Train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y,test_size=0.33, random_state=0)

#Training model used is Naive bayes classifier
from sklearn.naive_bayes import MultinomialNB
spam_detect_model=MultinomialNB().fit(X_train,y_train)
spam_detect_model.score(X_test,y_test)

#Generate model file to deploy in heroku platform
pickle.dump(spam_detect_model, open('nlp_model.pkl', 'wb'))



