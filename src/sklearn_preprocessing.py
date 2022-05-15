import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


# Load Dataset
df = pd.read_csv('../data/ames_unprocessed_data.csv')
df.head()
df.shape
df.dtypes
df.columns.tolist()
df['LotArea'].mean()
np.sqrt(df['SalePrice'].var())

# Check NULL Availability
df.isnull().sum()

# 5 Columns (MSZoning, MSZoning, Neighborhood, BldgType, HouseStyle) can be factorized.
for i in df.columns:
    print(i,df[f'{i}'].unique())

# Fill missing values with 0
df.LotFrontage = df.LotFrontage.fillna(0)

# Create a boolean mask for categorical columns
categorical_mask = (df.dtypes== object)

# Get list of categorical column names
categorical_columns = df.columns[categorical_mask].tolist()

# Print the head of the categorical columns
print(df[categorical_columns].head())

# Create LabelEncoder object: le
le = LabelEncoder()
# Apply LabelEncoder to categorical columns
df[categorical_columns] = df[categorical_columns].apply(lambda x: le.fit_transform(x))

# Print the head of the LabelEncoded categorical columns
print(df[categorical_columns].head())

# Create OneHotEncoder: ohe, categorical_feature is now deprecated.
ohe = OneHotEncoder(categorical_features=categorical_mask, sparse=False)

# Apply OneHotEncoder to categorical columns - output is no longer a dataframe: df_encoded
df_encoded = ohe.fit_transform(df)

# Print first 5 rows of the resulting dataset - again, this will no longer be a pandas dataframe
print(df_encoded[:5, :])

# Print the shape of the original DataFrame
print(df.shape)

# Print the shape of the transformed array
print(df_encoded.shape)