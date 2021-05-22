
import pandas as pd

import matplotlib

from pandas_profiling import ProfileReport 

df=pd.read_csv('Machine_learning_Disease_prediction/cancer.csv')

print(df)
#Generating the Reports

profile=ProfileReport(df)

profile.to_file(output_file="Machine_learning_Disease_prediction/Exploratory Data Analysis/data_analysis_cancer.html")