import sys
import pandas as pd
from sklearn.preprocessing import LabelEncoder

data_raw = pd.read_csv('train.csv')
#data validation
data_val = pd.read_csv('test.csv')
#creating a deep copy
data1 = data_raw.copy(deep=True)
#however passing by reference is convenient, because we can clean both datasets at once
#here basically concatenating the train and test dataset
data_cleaner = [data1, data_val]

#preview data
# print (data_raw.info())
# print(data_raw.head())
# data_raw.tail()
# print(data_raw.sample(10))

# print('Train columns with null values:\n', data1.isnull().sum())
# print("-"*10)
#
# print('Test/Validation columns with null values:\n', data_val.isnull().sum())
# print("-"*10)
#
# data_raw.describe(include = 'all')



#DATA CLEANING

#STEP1 CORRECTING that is here manualy values checked not required in this case

#STEP2 COMPLETING filling the none values
for dataset in data_cleaner:
    dataset['Age'].fillna(value=dataset['Age'].median(), inplace=True)
    # complete embarked with mode
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)

    # complete missing age with median
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace=True)

#delete the cabin feature/column and others previously stated to exclude in train dataset
drop_column = ['PassengerId','Cabin', 'Ticket']

data1.drop(drop_column, axis=1 , inplace=True)

#to check if data contains any null values or not
# print(data1.isnull().sum())
# print(data_val.isnull().sum())

#STEP3 CREATE: Feature Engineering for train and test/validation dataset
for dataset in data_cleaner:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset['IsAlone'] = 1
    dataset['IsAlone'].loc[dataset['FamilySize']>1] = 0

    dataset['Title'] = dataset['Name'].str.split(',', expand=True)[1].str.split('.', expand=True)[0]

    # Continuous variable bins; qcut vs cut: https://stackoverflow.com/questions/30211923/what-is-the-difference-between-pandas-qcut-and-pandas-cut
    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)

    # Age Bins/Buckets using cut or value bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.cut.html
    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)

    # print(dataset['AgeBin'].head().value_counts())

#CLEANING TITLE NAMES
start_min = 10
title_names = (data1['Title'].value_counts() < start_min)

# apply and lambda functions are quick and dirty code to find and replace with fewer lines of code:
# https://community.modeanalytics.com/python/tutorial/pandas-groupby-and-python-lambda-functions/
data1['Title'] = data1['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)

# preview data again
# data1.info()
# data_val.info()
# print(data1.sample(10))

# print(data1.sample(1))
#STEP 4 CONVERTING THE ENGLISH KEYWORDS TO MATHEMATICAL INPUTS
label = LabelEncoder()
for dataset in data_cleaner:
    #this append it to the end of the row, so both exists
    dataset['Embarked_Code'] = label.fit_transform(dataset['Embarked'])
    dataset['Title_Code'] = label.fit_transform(dataset['Title'])
    dataset['AgeBin_Code'] = label.fit_transform(dataset['AgeBin'])
    dataset['FareBin_Code'] = label.fit_transform(dataset['FareBin'])
    dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])

#define y variable aka target/outcome
Target = ['Survived']
#define x variables for original features aka feature selection
data1_x = ['Sex','Pclass', 'Embarked', 'Title','SibSp', 'Parch', 'Age', 'Fare',
           'FamilySize', 'IsAlone'] #pretty name/values for charts

data1_x_calc = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code','SibSp', 'Parch', 'Age', 'Fare'] #coded for algorithm calculation
data1_xy = Target + data1_x

#define x variables for original w/bin features to remove continuous variables
data1_x_bin = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code', 'FamilySize', 'AgeBin_Code', 'FareBin_Code']
data1_xy_bin = Target + data1_x_bin
# print('Bin X Y: ', data1_xy_bin, '\n')

#define x and y variables for dummy features original
data1_dummy = pd.get_dummies(data1[data1_x])
# print(data1_dummy.sample(5))
data1_x_dummy = data1_dummy.columns.tolist()
print(data1_x_dummy)
data1_xy_dummy = Target + data1_x_dummy
# print('Dummy X Y: ', data1_xy_dummy, '\n')



# print(data1_dummy.head())





