import numpy as np
import pandas as pd

### preparation for pandas
## create a dataframe
df1 = pd.DataFrame({'YES': [50 , 21], "NO": [131, 2]})

## build a index for row
df2 = pd.DataFrame({'Name': ['hinna', 'henan'], 'scores': [98, 88]}, index = ['row1', 'row2'])
df2

## create a series columns
df3 = pd.Series([10, 30, 20], index = ['row1', 'row2', 'row3'], name = 'df3')
df3

### loading data in pandas
## load titanic from kaggle
train = pd.read_csv("../titanic/train.csv")
test = pd.read_csv("../titanic/test.csv")

train['type'] = 'train'
test['type'] = 'test'

train.shape
test.shape

train.head(10)
test.head(10)

### rows and columns and splicing
## select a specific column in df
train.Pclass
train['Pclass']   # the same as train.Pclass
train['Pclass'][0]

## use iloc and loc for splicing
# iloc is based on location
train.iloc[1]     # select the second rows in train
train.iloc[:, 1]  # select data from the second column
train.iloc[:3, 0] # select data from the first column whose rows are from 1 to 3
train.iloc[[1, 4, 8], 0] # select data from column1/4/8 and row1
train.iloc[-5: ]  # select data from the last 5 rows

# loc is based on selection
train.loc[0, 'Survived']
train.loc[:, ['Survived', 'Age', 'Sex']]

## conditional selection
df = pd.concat([train, test])
df['Survived'] == 0

# select data whose survived are 0
df[df['Survived'] == 0]

# multi-conditional selection
df[(df['Survived'] == 0) & (df['Pclass'] == 1)]

condition = (df['Survived'] == 1) | (df['Pclass'] == 3)
df[condition]

# isin for checking
df['Survived'].isin([1, 3])
df[df['Survived'].isin([1, 3])]

# notnull for checking data
df['Pclass'].notnull()
df[df['Pclass'].notnull()]

# drop na
df_Pclass_notna = df.dropna(subset=['Pclass'])
df_Pclass_notna.head()
df_Pclass_notna.shape

## assigning data
df['type'] = 'same'
df['type']
df['index_backwards'] = range(len(df), 0, -1)
df['index_backwards']

## summary function
df[['type']].describe()
df[['Age']].describe()

df[['Age']].mean()
df['Pclass'].unique()
df['Pclass'].count()
df['Pclass'].value_counts()   

## Maps function
df_Age_mean = df['Age'].mean()
df['Age'].map(lambda p: p - df_Age_mean) ## not change the previous data

# or use 'apply' to personalize the function
def remean_Age(row):
    row.Age = row.Age - df_Age_mean
    return row
df.apply(remean_Age, axis='columns')

## Group by analysis
df.groupby('Survived')['Age'].mean()
df.groupby('Sex')['Survived'].mean()

df.groupby('Survived')['Age'].min()

## combined with groupby
df.groupby('Survived').apply(lambda df: df['Sex'].iloc[0])



