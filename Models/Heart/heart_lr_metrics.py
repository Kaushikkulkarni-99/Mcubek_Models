import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.linear_model import LogisticRegression
#warnings.filterwarnings("ignore", category=DeprecationWarning) 
from sklearn.preprocessing import StandardScaler
import random
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split

data = pd.read_csv("heart.csv")
data["trestbps"]=np.log(data["trestbps"])

data=data.drop(["fbs"],axis=1)
data=data.drop(["ca"],axis=1)
data["chol"]=np.log(data["chol"])
target=data["target"]
print(data.shape[1])

np.random.shuffle(data.values)
data=data.drop(["target"],axis=1)
print(data.columns)
sc= StandardScaler()
data=sc.fit_transform(data)

X_train,X_test,y_train,y_test=train_test_split(data,target,test_size=0.20,random_state=0)

lr=LogisticRegression()
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)

from sklearn.metrics import confusion_matrix,classification_report
confusion_lr_matrix=confusion_matrix(y_test,y_pred)
classification_lr_heart_report=classification_report(y_test,y_pred)

cv_results = cross_validate(lr, data,target, cv=10)
print(cv_results)
cv_results_heart=cv_results


#While Giving the values for the prediction we need to covert the input to standard scalar 
#sconvert= StandardScaler()
#converting_factor=scconvert.fit_transform("Two Dimensional Input")
