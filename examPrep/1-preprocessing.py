"""
In this file, I'll list all possible cases that can occur when preprocessing data.
"""
# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



# Loading data with pandas
dataset = pd.read_csv("./datasets/houses_portaland_simple.csv")



# Shuffle samples
dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)



# The first thing to do is to check what kind of columns I'm dealing with
print(dataset.describe())



# Usually, in the csv there are column description (best case)
selectedFeatures = ['Size', 'Bedroom']
targetFeature = ['Price']
X = dataset[selectedFeatures].values
y = dataset[selectedFeatures].values



# If there aren't column descriptions, we have to index the columns manually by integer indices
dataset = dataset.values
X = dataset[:, 0:2] # Remember, if I want to select from n to m, I have to specify n-1 to m
y = dataset[:, 2]



# And what if the order of the columns are not consequently, and I have to stack them horizontally?
X1 = dataset[:, 0]
X2 = dataset[:, 1]
X = np.column_stack((X1, X2)) # Remember the double parenthesis



# Let's see how to remove duplicates
newDataset = pd.read_csv("datasets/ds_salaries new.csv")
newDataset = newDataset[['work_year', 'experience_level', 'salary_in_usd']]
print(newDataset.describe(include="all"))
newDataset = newDataset.drop_duplicates()



# If I want to delete all the rows that contain NaN, I do this
newDataset.dropna() # Use the subset=["column_name"] if you want to specify only a column where searching for NaN



# If I want instead substitute the NaN values with the most frequent, I use this
newDataset = newDataset.fillna(newDataset['experience_level'].value_counts().index[0])



# Then, I want to transform the categorical features into numerical features
newDataset = pd.get_dummies(newDataset, prefix='work_year')
newDataset.drop(columns=["work_year"]) # Remember to eliminate the original feature
print(newDataset.describe(include="all"))



# What if I want to remove outliers?
q1 = newDataset["salary_in_usd"].quantile(0.25)
q3 = newDataset["salary_in_usd"].quantile(0.75)
newDataset = newDataset[(newDataset["salary_in_usd"] > q1) | (newDataset["salary_in_usd"] < q3)]
# In general, if I want to filter data I have to do df[condition]
newDataset = newDataset.values
X = newDataset[:, 1:]
y = newDataset[:, 0]



# Now I want to split data into train and test set
trainPercentage = 0.8
trainSamples = round(X.shape[0]*trainPercentage)
X_train = X[:trainSamples, :]
y_train = y[:trainSamples]
X_test = X[trainSamples:, :]
y_test = y[trainSamples:]
# With scikit-learn
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=trainPercentage, shuffle=True)



# Now I want to perform a z-score normalization
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
X_train = (X_train - mean)/std
X_test = (X_test - mean)/std
# If we want to use scikit-learn
scaler = StandardScaler() # Or RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)