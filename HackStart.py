# -*- coding: utf-8 -*-

# -- Sheet --

# # Hackstart Competition
# 
# 
# ## Problem statement 1
# The first problem statement consists of two parts:
# 1. Using data analytics **create a framework** to match the startups of Tamil Nadu sector-wise.
# 2. Given such data of Startups With Startup TN which in years will be scaled exponentially, **propose an architecture and implementable solution** that can be used to:
#     - Maintain a digital snapshot of the Startup Database
#     - Track the progress of Startup TN startups
# 
# ## Problem statement 2
# 1. Participants are required to identify and submit the names of the top 5 startups with the highest growth in terms of funding.
# 2. Find the top growing sector from the given dataset, Also elaborate on how you managed it?
# 3. Predict the total funding amount for the startups given in the prediction dataset


# importing all the required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# load data 
datauc= pd.read_excel('/data/notebook_files/DPIIT+Startups+-+8.2.2023+(1).xls') 
datauc

#add the cleaned data to original dataset
data = datauc.copy()
data


data['Sector'] = data['Sector'].replace([' '], 'unasssigned')
data

# to check for null or inconsistent values
data.info()
# or data.describe()

# data checkup
data.columns

y = data['Sector'].value_counts()
y

data.sort_values(by=['Sector'])

#create unique list of names
UniqueNames = data.Sector.unique()
print(UniqueNames)

#create a data frame dictionary to store your data frames
DataFrameDict = {elem : pd.DataFrame() for elem in UniqueNames}

for key in DataFrameDict.keys():
    DataFrameDict[key] = data[:][data.Sector == key]

#ask = input('Sector you want to access')

#DataFrameDict[str(ask)]


# Problem statement 1b 

y = data['Stage'].value_counts()
y

sns.countplot(data=data, x="Stage")

fig = px.pie(data, names='Stage', title='Stage',color_discrete_sequence=px.colors.sequential.RdBu)
fig.show()

# Changing the stages to integers value
data_copy = data.copy()
data_copy['Stage'] = data_copy['Stage'].replace(['Ideation','Validation','Early Traction','Scaling'], [1,2,3,4])
data_copy

# Sorted data copy 
data_copy.sort_values(by=['Stage'])

#create unique list of names
UniqueNames = data.Stage.unique()
print(UniqueNames)

#create a data frame dictionary to store your data frames
DataFrameDict = {elem : pd.DataFrame() for elem in UniqueNames}

for key in DataFrameDict.keys():
    DataFrameDict[key] = data[:][data.Stage == key]

#ask1 = input('Stage you want to access')

#DataFrameDict[str(ask1)]

# Problem statement 2
# load data 
data2uc= pd.read_excel('/data/notebook_files/investments_VC.xlsx') 
data2uc

# cleaning the data 
data_clean = data2uc[data2uc["status"]=="operating"]
data_clean = data_clean[data_clean[' funding_total_usd '] != ' -   ']
data_clean[' funding_total_usd '] = pd.to_numeric(data_clean[" funding_total_usd "])
data_clean

data_final = data_clean.assign(Arbitrary_growth = (data_clean[' funding_total_usd '] / data_clean['funding_rounds'])-data_clean['debt_financing'])
data_final

data_analysed = data_final.sort_values(by=['Arbitrary_growth'], ascending=False)
data_analysed

# top 5 in the above table
# problem statement 2a
data_analysed.head(5)

#create unique list of names
UniqueNames = data_final.category_list.unique()
print(UniqueNames)

#create a data frame dictionary to store your data frames
DataFrameDict = {elem : pd.DataFrame() for elem in UniqueNames}

for key in DataFrameDict.keys():
    DataFrameDict[key] = data_final[:][data_final.category_list == key]

#ask = input('Sector you want to access')

#DataFrameDict[str(ask)]

data2b = data_final.groupby("category_list").Arbitrary_growth.sum()
data2bfinal = data2b.sort_values(ascending=False)
Solution = data2bfinal.head(5)
# Solution is the dataframe containing top 5 growing categories
Solution

# problem statement 2c
# load data 
data2uc= pd.read_excel('/data/notebook_files/prediction_data.xlsx') 
data2uc

# cleaning the data 
data_clean_train = data_clean[[' funding_total_usd ','funding_rounds']]
data_clean_test = data2uc[data2uc["status"]== "operating" ]
data_clean_test = data_clean_test[data_clean_test[' funding_total_usd '] != ' -   ']
data_clean_test[' funding_total_usd '] = pd.to_numeric(data_clean_test[" funding_total_usd "])
data_clean_test_f = data_clean_test[[' funding_total_usd ','funding_rounds']]
data_clean_test_f

X_test = data_clean_test_f.drop(columns=[' funding_total_usd '])
y_test = data_clean_test_f[' funding_total_usd ']
X_train = data_clean_train.drop(columns=[' funding_total_usd '])
y_train = data_clean_train[' funding_total_usd ']
y_train = y_train.astype('int')
model = DecisionTreeClassifier()
model.fit(X_train, y_train)


predictions = model.predict(X_test.values)
score = accuracy_score(y_test,predictions)
score

