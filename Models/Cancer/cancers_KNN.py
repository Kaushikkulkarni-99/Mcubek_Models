import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt

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

'''
knn_scores= []
for k in range(1,21):
    knn_classifier=KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train,y_train)
    knn_scores.append(knn_classifier.score(X_test,y_test))

plt.plot([k for k in range(1,21)],knn_scores,color='red')
for i in range(1,21):
    plt.text(i,knn_scores[i-1],(i,knn_scores[i-1]))

plt.xticks([i for i in range(1,21)])
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Scores")
plt.title("K Neighbors Classifiers scores for different K values")
'''

knn_classifier=KNeighborsClassifier(n_neighbors=8)
knn_classifier.fit(X_train,y_train)


#K fold cross validation
cv_results_knn = cross_validate(knn_classifier,X,y,cv=10)





