import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from LogisticRegression import LogisticRegression
from Evaluation import Evaluation



# Modify here all the settings
path_dataset = '../datasets/diabetes.csv'
selected_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
target_feature = 'Outcome'
random_state = 42
train_percentage = 0.8
n_steps = 500
learning_rate = 1e-1
batch_size = 32



# Read the dataset of house prices
dataset = pd.read_csv(path_dataset)
# Print dataset stats (not necessary)
print(dataset.describe())
# Shuffling all samples to avoid group bias
dataset = dataset.sample(frac=1, random_state=random_state).reset_index(drop=True)

# Select only some features
X = dataset[selected_features].values
# This is to perform a polynomial regression
# X_squared = X**2
# X = np.column_stack((X_squared))
# Select target values
y = dataset[target_feature].values



# We split the dataset into training and test
train_index =round(len(X) * train_percentage)
X_train = X[:train_index]
y_train = y[:train_index]
X_test = X[train_index:]
y_test = y[train_index:]



# Compute the mean and standard deviation ONLY ON TRAINING SAMPLES, and we apply Z-score normalization
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std



# Add a bias column
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]



# Create a model with specified characteristics
model = LogisticRegression(n_features=X_train.shape[1], n_steps=n_steps, learning_rate=learning_rate)



# FULL BATCH MODE
# ------------------------------------------------------------
cost_history_fb, theta_history_fb = model.fit_fb(X_train, y_train)
# Print thetas and final costs
print('\n\nFULL BATCH MODE RESULTS: ')
print(f'''Thetas: {*model.theta,}''')
print(f'''Final train cost:  {cost_history_fb[-1]:.3f}''')
# Plot the cost history wrt all the epochs
plt.plot(cost_history_fb, 'g--')
plt.show()
# Evaluate the model
eval_fb = Evaluation(model)
print(eval_fb.compute_performance(X_test, y_test))



# MINI BATCH MODE
# ------------------------------------------------------------
# Fit with mini batch mode
cost_history_mb, theta_history_mb = model.fit_mb(X_train, y_train, batch_size=batch_size)
# Print thetas and final costs
print('\n\nMINI BATCH MODE RESULTS: ')
print(f'''Thetas: {*model.theta,}''')
print(f'''Final train cost:  {cost_history_mb[-1]:.3f}''')
# Plot the cost history wrt all the epochs
plt.plot(cost_history_mb, 'g--')
plt.show()
# Evaluate the model
eval_mb = Evaluation(model)
print(eval_mb.compute_performance(X_test, y_test))



# STOCASTIC GRADIENT DESCENT
# ------------------------------------------------------------
# Fit with Stocastic Gradient Descent
cost_history_sgd, theta_history_sgd = model.fit_mb(X_train, y_train, batch_size=1)
# Print thetas and final costs
print('\n\nSTOCASTIC GRADIENT DESCENT RESULTS: ')
print(f'''Thetas: {*model.theta,}''')
print(f'''Final train cost:  {cost_history_fb[-1]:.3f}''')
# Plot the cost history wrt all the epochs
plt.plot(cost_history_sgd, 'g--')
plt.show()
# Evaluate the model
eval_sgd = Evaluation(model)
print(eval_sgd.compute_performance(X_test, y_test))