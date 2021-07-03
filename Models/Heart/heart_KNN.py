import pandas as pd
import numpy as np
import warnings
from matplotlib import pyplot as plt
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
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

X_train,X_test,y_train,y_test=train_test_split(data,target,test_size=0.33,random_state=0)

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
cv_results = cross_validate(knn_classifier, data,target, cv=10)
print(cv_results)
cv_results_heart_KNN=cv_results



#While Giving the values for the prediction we need to covert the input to standard scalar 
#sconvert= StandardScaler()
#converting_factor=scconvert.fit_transform("Two Dimensional Input")
