#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 14:40:40 2021

@author: josedelcour
"""

import pickle
from flask import Flask
from flask import request
from flask import jsonify

model_dv = 'dv.bin'
model_lr = 'model1.bin'

with open(model_dv,'rb') as f_in1:
    dv = pickle.load(f_in1)
    
with open(model_lr,'rb') as f_in2:
    lr = pickle.load(f_in2)
    
customer = {"contract": "two_year", "tenure": 12, "monthlycharges": 10}


def predict(cust):

    X = dv.transform(cust)
    y_pred = lr.predict_proba(X)[0,1]
    return y_pred

print(predict(customer))

"""
app = Flask("churn")
@app.route('/predict',methods=['POST'])
def predict():
    customer = request.get_json()
    X = dv.transform(customer)
    y_pred = lr.predict_proba(X)[0,1]
    churn = y_pred >=0.5
    result = {
        "churn probability = ": float(y_pred),
        "churn":bool(churn)
        }
    return jsonify(result)



if __name__ == "__main__" :
    app.run(debug=True, host = '0.0.0.0', port=9696)
"""
## Q1:version pipenv: 2021.5.29
## Q2: 121f78d6564000dc5e968394f45aac87981fcaaf2be40cfcd8f07b2baa1e1829
## Q3: 0.11549580587832914
## Q4: {'churn': True, 'churn probability = ': 0.9988892771007961}
## Q6: 0.32940789808151005
## Q5: 32a5625aad35

#import requests    
#url = "http://0.0.0.0:9696/predict"
#customer = {"contract": "two_year", "tenure": 1, "monthlycharges": 10}
#requests.post(url, json=customer).json()
#{'churn': True, 'churn probability = ': 0.9988892771007961}


