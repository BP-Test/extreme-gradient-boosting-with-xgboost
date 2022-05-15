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
housing_data.info(verbose=True)

sns_plot = sns.pairplot(housing_data)
sns_plot.savefig("../figure/pairplot.png")


for column in housing_data.columns:
    plt.hist(x=housing_data[column])
    plt.title('Histogram of {}'.format(column))
    plt.show()

for column in housing_data.columns:
    sns.histplot(data=housing_data, x=housing_data[column]).set(title='Histogram of {}'.format(column))
    plt.show()


