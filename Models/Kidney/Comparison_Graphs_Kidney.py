# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 13:18:10 2021

@author: Kaushik
"""


from kidney import a11 as lr
from kidney_KNN import a11_kidney_KNN as knn
from kidney_Naive_Bayes import a11_kidney_nb as nb
from matplotlib import pyplot as plt
import numpy as np

plt.style.use('seaborn-dark')

Y_axis_kidney_logistic = lr['test_score']
Y_axis_kidney_knn = knn['test_score']
Y_axis_kidney_nb = nb['test_score']

X_axis = [i for i in range(1,len(Y_axis_kidney_knn)+1)]

plt.plot(X_axis,Y_axis_kidney_logistic,label="Logistic_Regression")
plt.plot(X_axis,Y_axis_kidney_knn,label="K Nearest Neighbors")
plt.plot(X_axis,Y_axis_kidney_nb,label="Naive_bayes")

plt.ylim(0.5,1.1)
plt.xlabel('K Folds')
plt.ylabel('Accuracies')
plt.title('Accuracies of Kidney Machine Learning Model')
plt.legend(loc = "lower right")
plt.tight_layout()
plt.show()

mean_accuracies_kidney={"logistic_regression":np.mean(Y_axis_kidney_logistic),"KNN":np.mean(Y_axis_kidney_knn),"Naive_Bayes":np.mean(Y_axis_kidney_nb)}
