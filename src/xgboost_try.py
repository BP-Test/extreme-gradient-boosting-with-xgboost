# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 16:26:02 2021

@author: tatsuro.0315@keio.jp
"""

import sys
sys.dont_write_bytecode = True
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import cv
import matplotlib.pyplot as plt
import seaborn as sns

#Change directory
os.chdir('C:/Users/tatsu/Documents/project_workspace/02_DataCamp/extreme-gradient-boosting-with-xgboost')

# Load dataset
df_ames_price = pd.read_csv('data/ames_housing_trimmed_processed.csv')
df_ames_price.head()

# Simple EDA on the dataset
df_ames_price.shape
df_ames_price.dtypes
df_ames_price.columns

# Split into training and test
X,y = df_ames_price.iloc[:,:-1], df_ames_price.iloc[:,-1]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=34)

# Create DMatrices
D_train = xgb.DMatrix(X_train, label=y_train)
D_test = xgb.DMatrix(X_test, label=y_test)
# Get DMatrices Details
D_train.get_base_margin

# Set Parameters and cross-validation scheme
params = {'booster':'gblinear','objective':'reg:linear'}
model1 = xgb.cv(dtrain=D_train,num_boost_round=3, nfold=4,params=params, seed=123)

# Using linear base : Learning API Only
model1_train = xgb.train(params=params, dtrain=D_train, num_boost_round=3, nfold=4, random_state=123)
xgb.XGBRegressor()

