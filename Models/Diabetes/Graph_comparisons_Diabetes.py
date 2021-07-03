# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 14:01:19 2021

@author: Kaushik
"""


from diabetes import cv_results as lr
from diabetes_KNN import cv_results_diabetes_knn as knn
from diabetes_Naive_Bayes import cv_results_diabetes_nb as nb
from matplotlib import pyplot as plt
import numpy as np

plt.style.use('seaborn-dark')

Y_axis_diabetes_logistic = lr['test_score']
Y_axis_diabetes_knn = knn['test_score']
Y_axis_diabetes_nb = nb['test_score']

X_axis = [i for i in range(1,len(Y_axis_diabetes_knn)+1)]

plt.plot(X_axis,Y_axis_diabetes_logistic,label="Logistic_Regression")
plt.plot(X_axis,Y_axis_diabetes_knn,label="K Nearest Neighbors")
plt.plot(X_axis,Y_axis_diabetes_nb,label="Naive_bayes")

plt.ylim(0.5,1.1)
plt.xlabel('K Folds')
plt.ylabel('Accuracies')
plt.title('Accuracies of Diabetes Machine Learning Model')
plt.legend(loc = "lower right")
plt.tight_layout()
plt.show()

mean_accuracies_diabetes={"logistic_regression":np.mean(Y_axis_diabetes_logistic),"KNN":np.mean(Y_axis_diabetes_knn),"Naive_Bayes":np.mean(Y_axis_diabetes_nb)}
