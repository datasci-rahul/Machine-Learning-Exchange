# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 11:48:31 2022 DAY 1

@author: csera
"""

# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset
dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:,:-1].values #Matrix Of Features and Dependent Variable Vector
y = dataset.iloc[:,3].values

# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean') #'verbose' parameter was deprecated
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

# Encode Categorical Data

# Encoding the Independent Variable

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
LabelEncoder_X = LabelEncoder()
X[:,0] = LabelEncoder_X.fit_transform(X[:,0])

from sklearn.compose import ColumnTransformer 
ct = ColumnTransformer([("Country", OneHotEncoder(), [0])], remainder = 'passthrough')
X = ct.fit_transform(X)   #X = np.array(ct.fit_transform(X), dtype=np.float)
# Encoding Y data
LabelEncoder_y = LabelEncoder()
y = LabelEncoder_y.fit_transform(y)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split  #from sklearn.cross_validation import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))