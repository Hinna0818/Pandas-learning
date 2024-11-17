import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
import warnings
from sklearn.tree import DecisionTreeRegressor

## prepare the datasets for practice!
titanic_train = pd.read_csv("../titanic/train.csv")
titanic_test = pd.read_csv("../titanic/test.csv")

titanic_train.head(10)
titanic_test.head(10)

## concat two data
titanic_train['data_type'] = 'train'
titanic_test['data_type'] = 'test'
df = pd.concat([titanic_train, titanic_test])
df.head(10)

print("the combined data shape is: ", df.shape)
df['data_type'].value_counts()

## check for duplicated records
df.duplicated().sum()
df.info()

## show basic info for the data
df.describe()

## check the na
df.isna()
df = df.dropna(axis = 0) ## drop missing values

## subset variables
PassengerId = df.PassengerId
PassengerId.head()

## subset some columns for further analysis
df_outcome = df.Survived
df_features = ['Age', 'Parch', 'Fare']
df_subset = df[df_features]
df_subset.head(10)
df_subset.describe()

## prediction
titanic_model = DecisionTreeRegressor(random_state=1)
titanic_model.fit(df_features, df_outcome)