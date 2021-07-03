import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate

#Data Preprocessing 
data=pd.read_csv("diabetes.csv")
X=data.iloc[:,:8]
y=data[["Outcome"]]


#Model Fitting
from sklearn.naive_bayes import GaussianNB
logreg=GaussianNB()
logreg.fit(X,y)

#Kfold cross validation
cv_results_diabetes_nb = cross_validate(logreg,X,y, cv=10)