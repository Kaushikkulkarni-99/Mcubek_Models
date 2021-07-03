import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

#Data Preprocessing 
data=pd.read_csv("diabetes.csv")
X=data.iloc[:,:8]
y=data[["Outcome"]]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.22,random_state=0)

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

knn_classifier=KNeighborsClassifier(n_neighbors=10)
knn_classifier.fit(X_train,y_train)


#Kfold cross validation
cv_results_diabetes_knn = cross_validate(knn_classifier,X,y, cv=10)