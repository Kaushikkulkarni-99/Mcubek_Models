# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 14:32:31 2021

@author: Kaushik
"""


from cancers import cv_results as lr
from cancers_KNN import cv_results_knn as knn
from cancers_Naive_Bayes import cv_results_nb as nb
from matplotlib import pyplot as plt
import numpy as np

plt.style.use('seaborn-dark')

Y_axis_cancer_logistic = lr['test_score']
Y_axis_cancer_knn = knn['test_score']
Y_axis_cancer_nb = nb['test_score']

X_axis = [i for i in range(1,len(Y_axis_cancer_knn)+1)]

plt.plot(X_axis,Y_axis_cancer_logistic,label="Logistic_Regression")
plt.plot(X_axis,Y_axis_cancer_knn,label="K Nearest Neighbors")
plt.plot(X_axis,Y_axis_cancer_nb,label="Naive_bayes")

plt.ylim(0.5,1.1)
plt.xlabel('K Folds')
plt.ylabel('Accuracies')
plt.title('Accuracies of Cancer Machine Learning Model')
plt.legend(loc = "lower right")
plt.tight_layout()
plt.show()

mean_accuracies_cancer={"logistic_regression":np.mean(Y_axis_cancer_logistic),"KNN":np.mean(Y_axis_cancer_knn),"Naive_Bayes":np.mean(Y_axis_cancer_nb)}
