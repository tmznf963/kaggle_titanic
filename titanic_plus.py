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

#성별에 따라 , 클래스, 등급에 따라 죽었는지 살았는지 구별하는 함수
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


#Feature engineering 도메인 데이터
#Image(url= "https://static1.squarespace.com/static/5006453fe4b09ef2252ba068/t/5090b249e4b047ba54dfd258/1351660113175/TItanic-Survival-Infographic.jpg?format=1500w")
train.head(10)

train_test_data = [train,test]

for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract('([A-Za-z]+)\.',expand=False)

train['Title'].value_counts()

test['Title'].value_counts()

#이름
title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }

for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)

train.head(10)
test.head(10)

bar_chart('Title')
#delete unnecessary feature from dataset
train.drop('Name',axis = 1, inplace = True)
test.drop('Name',axis=1 , inplace = True)

train.head(10)
test.head(10)

#성별 
sex_mapping = {"male": 0,"feamle" : 1}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)
    
bar_chart('Sex')

train.head(100)

train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace=True)
test["Age"].fillna(test.groupby("Title")["Age"].transform("median"), inplace=True)

#그래프 표현 해보기
facet = sns.FacetGrid(train, hue="Survived",aspect = 4)
facet.map(sns.kdeplot,'Age', shade= True)
facet.set(xlim=(0,train['Age'].max()))
facet.add_legend()
plt.show()

facet = sns.FacetGrid(train, hue="Survived",aspect = 4)
facet.map(sns.kdeplot,'Age', shade= True)
facet.set(xlim=(0,train['Age'].max()))
facet.add_legend()
plt.xlim(0,20)#0~20살

facet = sns.FacetGrid(train, hue="Survived",aspect = 4)
facet.map(sns.kdeplot,'Age', shade= True)
facet.set(xlim=(0,train['Age'].max()))
facet.add_legend()
plt.xlim(20,30)#20~30살

train.info()
test.info()

#Binning 카테고리별로 나누기
#Age 숫자로 바꾸기
for dataset in train_test_data:
    dataset.loc[dataset['Age'] <=16,'Age']=0,
    dataset.loc[(dataset['Age'] >16) & (dataset['Age']<=26),'Age']=1,
    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2,
    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 3,
    dataset.loc[ dataset['Age'] > 62, 'Age'] = 4

train.head()

bar_chart('Age')

#Embarked 선탑장
Pclass1 = train[train['Pclass']==1]['Embarked'].value_counts()
Pclass2 = train[train['Pclass']==2]['Embarked'].value_counts()
Pclass3 = train[train['Pclass']==3]['Embarked'].value_counts()
df = pd.DataFrame([Pclass1,Pclass2,Pclass3])
df.index = ['1st class','2nd class','3rd class']
df.plot(kind='bar',stacked=True,figsize=(10,5))

#Embarked에 값이 없으면 S로 채워 넣어라
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

train.head(10)

#머신러닝 클래스 파이어를 위해 텍스트를 숫자로 바꿔주기
embarked_mapping = {"S" : 0 , "C" : 1 ,"Q":2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)

#티켓가격 missing 구역을 채워넣기
train["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace=True)
test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace=True)
train.head(30)

facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Fare',shade= True)
facet.set(xlim=(0, train['Fare'].max()))
facet.add_legend()
plt.show()

#달러를 카테고리화 하기
for dataset in train_test_data:
    dataset.loc[ dataset['Fare'] <= 17, 'Fare'] = 0,
    dataset.loc[(dataset['Fare'] > 17) & (dataset['Fare'] <= 30), 'Fare'] = 1,
    dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 100), 'Fare'] = 2,
    dataset.loc[ dataset['Fare'] > 100, 'Fare'] = 3

train.head(10)

#Cabin 방
train.Cabin.value_counts()

#첫번째 방 받아오기
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].str[:1]

Pclass1 = train[train['Pclass']==1]['Cabin'].value_counts()
Pclass2 = train[train['Pclass']==2]['Cabin'].value_counts()
Pclass3 = train[train['Pclass']==3]['Cabin'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class', '3rd class']
df.plot(kind='bar',stacked=True, figsize=(10,5))

#숫자화 하기 머신러닝 클래스 파이어를 위해 , 숫자를 소숫점 사용한 이유 : 범위 오차를 줄이기 위해
cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)

train["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
test["Cabin"].fillna(test.groupby("Pclass")["Cabin"].transform("median"), inplace=True)

#가족크기 , 혼자 탔는지 다수와 탔는지
train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
test["FamilySize"] = test["SibSp"] + test["Parch"] + 1

facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'FamilySize',shade= True)
facet.set(xlim=(0, train['FamilySize'].max()))
facet.add_legend()
plt.xlim(0)

family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}
for dataset in train_test_data:
    dataset['FamilySize'] = dataset['FamilySize'].map(family_mapping)

train.head(20)

#불필요한 속성 빼기
features_drop = ['Ticket', 'SibSp', 'Parch']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)
train = train.drop(['PassengerId'], axis=1)

train_data = train.drop('Survived', axis=1)
target = train['Survived']

train_data.shape, target.shape

#모든 feature들이 숫자를 이룸
train_data.head(10)