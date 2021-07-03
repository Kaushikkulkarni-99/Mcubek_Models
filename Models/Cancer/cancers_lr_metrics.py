import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split

#Data Preprocessing 
data=pd.read_csv("cancers.csv")

data.drop(["Unnamed: 32"],axis="columns",inplace=True)
data.drop(["id"],axis="columns",inplace=True)

#Converting the Categorical Variables
a=pd.get_dummies(data["diagnosis"])

#Concatinating the categorical data columns to the dataset 
cancer=pd.concat([data,a],axis="columns")

#Removing Dummy Variable Trap and duplicate Diagnosis column
cancer.drop(["diagnosis","B"],axis="columns",inplace=True) 

cancer.rename(columns={"M":"Malignant/Benign"},inplace=True)

#get the dependnt column
y=cancer[["Malignant/Benign"]]

#Independent column
X=cancer.drop(["Malignant/Benign"],axis="columns")

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)

#READY TO TRAIN AND PREDICT
#Classification models
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)

from sklearn.metrics import confusion_matrix,classification_report
confusion_lr_matrix=confusion_matrix(y_test,y_pred)
classification_lr_cancer_report=classification_report(y_test,y_pred)

#K fold cross validation
cv_results = cross_validate(logreg,X,y,cv=10)
print(cv_results)


#Inorder to load the model use m=joblib.load("filename")




