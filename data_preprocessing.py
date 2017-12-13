import sys
import pandas as pd

data_raw = pd.read_csv('train.csv')
#data validation
data_val = pd.read_csv('test.csv')
#creating a deep copy
data1 = data_raw.copy(deep=True)
#however passing by reference is convenient, because we can clean both datasets at once
data_cleaner = [data1, data_val]

#preview data
# print (data_raw.info())
# print(data_raw.head())
# data_raw.tail()
# print(data_raw.sample(10))


