# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 11:37:57 2021

@author: Kaushik
"""


import joblib
import numpy as np

sc_test=joblib.load("Heart_Standard_Scalar")
model_test=joblib.load("heart_model")

array_positive=[[63,1,3,145,233,0,150,0,2.3,0,1]]
array_negative=[[67,1,2,152,212,0,150,0,0.8,1,3]]

array_positive[0][3]=np.log(array_positive[0][3])
array_positive[0][4]=np.log(array_positive[0][4])
array_negative[0][3]=np.log(array_negative[0][3])
array_negative[0][4]=np.log(array_negative[0][4])


array_positive=sc_test.transform(array_positive)
array_negative=sc_test.transform(array_negative)

print(model_test.predict(array_positive))
print(model_test.predict(array_negative))






