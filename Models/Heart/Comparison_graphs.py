# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 12:44:51 2021

@author: Kaushik
"""


from heart import cv_results as lr
from heart_KNN import cv_results_heart_KNN as knn
from heart_Naive_Bayes import cv_results_heart_Naive_bayes as nb
from matplotlib import pyplot as plt
import numpy as np

plt.style.use('seaborn-dark')

Y_axis_heart_logistic = lr['test_score']
Y_axis_heart_knn = knn['test_score']
Y_axis_heart_nb = nb['test_score']

X_axis = [i for i in range(1,len(Y_axis_heart_knn)+1)]

plt.plot(X_axis,Y_axis_heart_logistic,label="Logistic_Regression")
plt.plot(X_axis,Y_axis_heart_knn,label="K Nearest Neighbors")
plt.plot(X_axis,Y_axis_heart_nb,label="Naive_bayes")

plt.ylim(0.5,1.1)
plt.xlabel('K Folds')
plt.ylabel('Accuracies')
plt.title('Accuracies of Heart Machine Learning Model')
plt.legend(loc = "lower right")
plt.tight_layout()
plt.show()

mean_accuracies_heart={"logistic_regression":np.mean(Y_axis_heart_logistic),"KNN":np.mean(Y_axis_heart_knn),"Naive_Bayes":np.mean(Y_axis_heart_nb)}





