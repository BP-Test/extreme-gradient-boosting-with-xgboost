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

params = {'objective':'reg:linear','max_depth':3}

# Try different values for colsample_by_tree (Parameter to control %feature used by tree, larger value can bring overfitting problem)
colsample_by_tree_vals = [0.1,0.5,0.8,1]

best_rmse = []
for val in colsample_by_tree_vals:
    params['colsample_bytree'] = val
    cv_result_data = xgb.cv(params=params,
                            dtrain=housing_dmatrix,
                            seed=123,
                            nfold=2,
                            num_boost_round=10,
                            early_stopping_rounds=5,
                            metrics='rmse',
                            as_pandas=True
                            )
    best_rmse.append(cv_result_data['test-rmse-mean'].tail().values[-1])

result_dataframe = pd.DataFrame(
    list(
        zip(colsample_by_tree_vals, best_rmse)),
        columns=['colsample_by_tree','best_rmse']
    )
display(result_dataframe)


def show_basic_information(data):
    return display(data), display(data.info()), display(data.describe())

show_basic_information(result_dataframe)