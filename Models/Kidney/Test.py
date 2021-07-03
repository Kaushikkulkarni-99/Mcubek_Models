# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 12:01:34 2021

@author: Kaushik
"""


import joblib

input=[[71,60,0,0,97,27,0.9,15.2,42,0,0,1]]

result=joblib.load('kidney_model')

result.predict(input)

