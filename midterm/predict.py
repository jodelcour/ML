#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 18:35:06 2021

@author: josedelcour
"""
import pickle
from flask import Flask
from flask import request
from flask import jsonify
import pandas as pd
import xgboost as xgb

model_file='titanic-model.bin'
with open(model_file,'rb') as fin:
    model = pickle.load(fin)

def ptransform(pdic):
    outdic={'pclass':0, 'age':0, 'fare':0, 'family':0, 'x0_female':0, 'x0_male':0,
            'x1_C':0,'x1_Q':0, 'x1_S':0, 'x2_Master':0, 'x2_Miss':0, 'x2_Mr':0,
            'x2_Mrs':0, 'x2_Officer':0,'x2_Royalty':0}
    outdic['pclass']=pdic['pclass']; outdic['age']=pdic['age']
    outdic['fare']=pdic['fare']; outdic['family']=pdic['family']
    if pdic['sex']=='male':
        outdic['x0_male']=1
    else:
        outdic['x0_female']=1
    if pdic['embarked']=='Southampton':
       outdic['x1_S']=1
    elif pdic['embarked']=='Cherbourg':
        outdic['c1_C']=1
    else:
        outdic['c1_Q']=1
    
#{Cherbourg, Queenstown and Southhampton}
    if pdic['title']=='Miss':
        outdic['x2_Miss']=1
    elif pdic['title']=='Master':
        outdic['x2_Master']=1
    elif pdic['title']=='Mr':
        outdic['x2_Mr']=1
    elif pdic['title']=='Mrs':
        outdic['x2_Mrs']=1
    elif pdic['Officer']=='Officer':
        outdic['x2_Officer']=1    
    else:
        outdic['Royalty']=1
    df_record = pd.DataFrame([outdic])
    xgb_record = xgb.DMatrix(df_record)
    return xgb_record
#{'Miss', 'Master', 'Mr', 'Mrs', 'Officer', 'Royalty'}

app = Flask('titanic')    
@app.route('/predict',methods = ['POST'])

def predict():
    passenger = request.get_json()
    xgbrec = ptransform(passenger)    
    y_pred = model.predict(xgbrec)
    survived=(y_pred>=0.5)
    result = {'Probability of survival':float(y_pred), 
              'survived':bool(survived)}
    
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port ='9696')    
#------------------------------------------------------------------------------    
passenger = {"pclass": 1,
             "age": 28,
             "fare": 50,
             "family": 0,
             "sex": "male",
             "embarked": "Southampton",
             "title": "Mr"}
#------------------------------------------------------------------------------