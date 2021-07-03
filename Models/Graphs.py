# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 14:15:06 2021

@author: Kaushik
"""

from Heart import heart
from Kidney import kidney
from Diabetes import diabetes
from Cancer import cancers
from matplotlib import pyplot as plt

plt.style.use('seaborn-dark')

Y_axis_heart = heart.cv_results['test_score']
Y_axis_kidney = kidney.a11['test_score']
Y_axis_cancer = cancers.cv_results['test_score']
Y_axis_diabetes = diabetes.cv_results['test_score']

X_axis = [i for i in range(1,len(Y_axis_heart)+1)]

plt.plot(X_axis,Y_axis_heart,label="Heart_model")
plt.plot(X_axis,Y_axis_kidney,label="Kidney_model")
plt.plot(X_axis,Y_axis_cancer,label="Cancer_model")
plt.plot(X_axis,Y_axis_diabetes,label="Diabetes_model")

plt.ylim(0.5,1.1)
plt.xlabel('K Folds')
plt.ylabel('Accuracies')
plt.title('Accuracies of Machine Learning Models')
plt.legend(loc = "lower right")
plt.tight_layout()
plt.show()
plt.savefig("Accuracies.png")







