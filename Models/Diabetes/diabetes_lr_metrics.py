import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split


#Data Preprocessing 
data=pd.read_csv("diabetes.csv")
X=data.iloc[:,:8]
y=data[["Outcome"]]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.10,random_state=0)

#Model Fitting
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(X_train,y_train)

y_pred=logreg.predict(X_test)

from sklearn.metrics import confusion_matrix,classification_report
confusion_lr_matrix=confusion_matrix(y_test,y_pred)
classification_lr_diabetes_report=classification_report(y_test,y_pred)

#Kfold cross validation
cv_results = cross_validate(logreg,X,y, cv=10)
print(cv_results)