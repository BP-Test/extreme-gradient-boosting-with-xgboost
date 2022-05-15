import os
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import chart_studio.plotly as py
import cufflinks as cf
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import GridSearchCV
%matplotlib inline
# Make Plotly work in your Jupyter Notebook
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
# Use Plotly locally
cf.go_offline()

######################################################################################
# Load Dataset
os.getcwd()
os.listdir('../data')
housing_data = pd.read_csv('../data/ames_housing_trimmed_processed.csv')
housing_data.head()

# Split into features and labels
X,y = housing_data.iloc[:,:-1], housing_data.iloc[:,-1]


# Create your housing DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

gbm_param_grid = {
    'colsample_bytree':[0.3,0.7],
    'n_estimators':[50], #Number of gradient boosted trees. Equivalent to number of boosting rounds.
    'max_depth': [2,5]
}

gbm = xgb.XGBRegressor()
grid_mse = GridSearchCV(estimator=gbm,
                        param_grid=gbm_param_grid,
                        scoring='neg_mean_squared_error',
                        cv=4,
                        verbose=1)
# Scoring value can be found here : sklearn.metrics.SCORERS.keys()
grid_mse.fit(X,y)
print('The best parameters are as follows',grid_mse.best_params_)
print('The best RMSE is as follows',np.sqrt(np.abs(grid_mse.best_score_)))