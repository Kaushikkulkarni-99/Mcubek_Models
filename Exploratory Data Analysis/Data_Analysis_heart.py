# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 11:40:53 2020

@author: Kaushik
"""

import matplotlib

import pandas as pd

from pandas_profiling import ProfileReport 

df=pd.read_csv('Machine_learning_Disease_prediction/heart.csv')

print(df)
#Generating the Reports

profile=ProfileReport(df)

profile.to_file(output_file="Machine_learning_Disease_prediction/Exploratory Data Analysis/data_analysis_heart.html")




