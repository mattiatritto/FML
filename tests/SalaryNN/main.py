import pandas as pd
import numpy as np

# Settings
filepath = "ds_salaries new.csv"
train_proportion = 0.8

# PRE PROCESSING STEPS
dataset = pd.read_csv(filepath)

# Drop duplicated rows
dataset = dataset.drop_duplicates()

# Remove outliers
q1 = dataset['salary_in_usd'].quantile(0.25)
q3 = dataset['salary_in_usd'].quantile(0.75)
iqr = q3 - q1
outliers = dataset[(dataset['salary_in_usd'] < q1 - 1.5 * iqr) | (dataset['salary_in_usd'] > q3 + 1.5 * iqr)]
dataset = dataset.drop(outliers.index)

# Encoding categorical features
dataset = pd.get_dummies(dataset, prefix='work_year')
dataset = pd.get_dummies(dataset, prefix='experience_level')
dataset = pd.get_dummies(dataset, prefix='employment_type')
dataset = pd.get_dummies(dataset, prefix='job_title')
dataset = pd.get_dummies(dataset, prefix='employee_residence')
dataset = pd.get_dummies(dataset, prefix='remote_ratio')
dataset = pd.get_dummies(dataset, prefix='company_location')
dataset = pd.get_dummies(dataset, prefix='company_size')

# Splitting data into training and test set
y = dataset["salary_in_usd"].values
X = dataset.drop(columns="salary_in_usd", axis=1).values
train_index = round(X.shape[0]*train_proportion)
X_train = X[:train_index, :]
y_train = X[:train_index]
X_test = X[train_index:, :]
y_test = X[train_index:]
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std