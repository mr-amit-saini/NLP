# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 20:22:50 2021

@author: amits
"""

from flask import Flask, render_template, request
import pickle

#Load the model from disk
model = pickle.load(open('nlp_model.pkl', 'rb'))
tfidf=pickle.load(open('vectorized_data.pkl','rb'))

#Create Flask App
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        data_vector = tfidf.transform(data).toarray()
        my_prediction = model.predict(data_vector)
    return render_template('result.html', prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)