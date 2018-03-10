# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#from IPython.display import image
#image(url="https://static1.squarespace.com/static/5006453fe4b09ef2252ba068/5095eabce4b06cb305058603/5095eabce4b02d37bef4c24c/1352002236895/100_anniversary_titanic_sinking_by_esai8mellows-d4xbme8.jpg")

import pandas as pd

train = pd.read_csv("C:\\input\\train.csv")
test = pd.read_csv("C:\\input\\test.csv")

train.head(40)
test.head(10)
train.shape
train.info()
test.info()

train.isnull().sum()
test.isnull().sum()

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()

def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True,figsize=(10,5))

bar_chart('Sex')
bar_chart('Pclass')
bar_chart("SibSp")
bar_chart("Parch")
bar_chart("Embarked")

Image(url= "https://static1.squarespace.com/static/5006453fe4b09ef2252ba068/t/5090b249e4b047ba54dfd258/1351660113175/TItanic-Survival-Infographic.jpg?format=1500w")

train.head(10)

train_test_data = [train,test]

for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract('([A-Za-z]+)\.',expand=False)

train['Title'].value_counts()

test['Title'].value_counts()

title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)

train.head(10)
test.head(10)

bar_chart('Title')
train.drop('Name',axls = 1, inplace = True)
test.drop('Name',axls=1 , inplace = True)

train.head(10)
test.head(10)
