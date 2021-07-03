# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 17:39:20 2021

@author: Kaushik
"""


import joblib

model=joblib.load('diabetes_model')

array_input=[[6,148,72,35,0,33.6,0.627,50]]

print(model.predict(array_input))