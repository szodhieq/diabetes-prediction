# -*- coding: utf-8 -*-
"""
Created on Mon May 16 12:36:23 2022

@author: YASHIM GABRIEL
"""

import numpy as np 
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('RandomForest887.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    prg = int(request.form['pregnancies'])
    glu = int(request.form['glucose'])
    bp = int(request.form['bp'])
    skin = int(request.form['skin'])
    ins = int(request.form['insulin'])
    bmi = float(request.form['bmi'])
    dpf = float(request.form['dpf'])
    age = int(request.form['age'])
    
    final_features = np.array([[prg,glu,bp,skin,ins,bmi,dpf,age]])
    prediction = model.predict(final_features)
    
    output = prediction[0]
    
    if output == 1:
        output = 'Diabetic'
    else:
        output = 'Not Diabetic'
     
    return render_template('results.html', prediction_text=output)
     

if __name__ == "__main__":
    app.run(debug=True)