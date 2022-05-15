import os
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import chart_studio.plotly as py
import cufflinks as cf
import plotly.express as ex
%matplotlib inline
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode = True
cf.go_offline()
os.getcwd()
housing_data = pd.read_csv('../data/ames_housing_trimmed_processed.csv')
housing_data.head()
X, y = housing_data.iloc[:,:-1], housing_data.iloc[:,-1]
X.head()
y.head()

d_train = xgb.DMatrix(data=X,label=y)
params = {"objective":"reg:linear"}
max_depth_vals = [2, 5, 10, 20]
best_rmse = []
for current_val in max_depth_vals:
    params["max_depth"] = current_val
    cval_results = xgb.cv(params=params, dtrain=d_train,seed=123)
    best_rmse.append(cval_results["test-rmse-mean"].tail().values[-1])

max_depth_results = pd.DataFrame(list(zip(max_depth_vals, best_rmse)),columns=['max_depth','best_rmse'])
max_depth_results.iplot(
    x='max_depth',y='best_rmse', 
    title='max_depth vs RMSE')