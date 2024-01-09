import pandas as pd
import matplotlib.pyplot as plt
from logistic_regression import LogisticRegression
import utilities
import numpy as np
plt.style.use(['ggplot'])

# Read the dataset
diabetes = pd.read_csv('diabetes.csv')

# Print dataset stats
print(diabetes.describe())
print(diabetes.columns)

# Shuffling all samples to avoid group bias
diabetes = diabetes.sample(frac=1).reset_index(drop=True)

selected_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                     'BMI', 'DiabetesPedigreeFunction', 'Age']
x = diabetes[selected_features].values

# Select target value
y = diabetes['Outcome'].values
# Perform train-test split
train_ratio = 0.8
train_size = int(len(x) * train_ratio)

X_train = x[:train_size]
y_train = y[:train_size]

X_test = x[train_size:]
y_test = y[train_size:]

# Compute mean and standard deviation ONLY ON TRAINING SAMPLES
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

# Apply mean and std (standard deviation) computed on the training sample to training set and test set
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Add a column of ones for the bias term
X_train = np.column_stack((np.ones(X_train.shape[0]), X_train))
X_test = np.column_stack((np.ones(X_test.shape[0]), X_test))

# Create an instance of LogisticRegression
logistic_fb = LogisticRegression(learning_rate=1e-1, n_steps=500, n_features=X_train.shape[1])

# Train the model using full batch gradient descent
cost_history, theta_history = logistic_fb.fit_full_batch(X_train, y_train)

# Make predictions on the test data
# Logistic regression full batch

y_pred_fb = logistic_fb.predict(X_test)

# compute metrics for classification
metrics = utilities.compute_classification_metrics(y_test, y_pred_fb)

#print metrics
print("Classification Metrics Full batch:")
print("Accuracy:", metrics["Accuracy"])
print("Precision:", metrics["Precision"])
print("Recall:", metrics["Recall"])
print("F1 Score:", metrics["F1 Score"])

utilities.plot_theta_gd(X_train, y_train, logistic_fb, cost_history, theta_history)

# Plot the change in individual theta values over iterations
plt.figure(figsize=(12, 6))
for i in range(logistic_fb.theta.shape[0]):
    plt.plot(range(logistic_fb.n_steps), theta_history[:, i], label=f'fit_full_batch Theta {i + 1}')

plt.xlabel('Iteration')
plt.ylabel('Theta Value')
plt.legend()
plt.title('Change in Theta Values over Iterations in full_batch')
plt.grid(True)
plt.show()

#Logistic with mini batch
# Create an instance of LogisticRegression
logistic_mb = LogisticRegression(learning_rate=1e-1, n_steps=500, n_features=X_train.shape[1])

# Train the model using mini batch gradient descent
cost_history, theta_history = logistic_mb.fit_mini_batch(X_train, y_train, batch_size=32)

# Make predictions on the test data

y_pred_mb = logistic_mb.predict(X_test)

# compute metrics for classification
metrics = utilities.compute_classification_metrics(y_test, y_pred_mb)

#print metrics
print("Classification Metrics Mini batch:")
print("Accuracy:", metrics["Accuracy"])
print("Precision:", metrics["Precision"])
print("Recall:", metrics["Recall"])
print("F1 Score:", metrics["F1 Score"])

utilities.plot_theta_gd(X_train, y_train, logistic_mb, cost_history, theta_history)
# Plot the change in individual theta values over iterations
plt.figure(figsize=(12, 6))
for i in range(logistic_mb.theta.shape[0]):
    plt.plot(range(logistic_mb.n_steps), theta_history[:, i], label=f'fit_mini_batch Theta {i + 1}')

plt.xlabel('Iteration')
plt.ylabel('Theta Value')
plt.legend()
plt.title('Change in Theta Values over Iterations mini_batch')
plt.grid(True)
plt.show()

#Logistic with SGD
# Create an instance of LogisticRegression
logistic_sgd = LogisticRegression(learning_rate=0.002, n_steps=500, n_features=X_train.shape[1])

# Train the model using mini batch gradient descent
cost_history, theta_history = logistic_sgd.fit_mini_batch(X_train, y_train, batch_size=32)

# Make predictions on the test data

y_pred_sgd = logistic_sgd.predict(X_test)

# compute metrics for classification
metrics = utilities.compute_classification_metrics(y_test, y_pred_sgd)

#print metrics
print("Classification Metrics SGD:")
print("Accuracy:", metrics["Accuracy"])
print("Precision:", metrics["Precision"])
print("Recall:", metrics["Recall"])
print("F1 Score:", metrics["F1 Score"])

utilities.plot_theta_gd(X_train, y_train, logistic_sgd, cost_history, theta_history)

# Plot the change in individual theta values over iterations
plt.figure(figsize=(12, 6))
for i in range(logistic_sgd.theta.shape[0]):
    plt.plot(range(logistic_sgd.n_steps), theta_history[:, i], label=f'fit_sgd Theta {i + 1}')

plt.xlabel('Iteration')
plt.ylabel('Theta Value')
plt.legend()
plt.title('Change in Theta Values over Iterations in SGD')
plt.grid(True)
plt.show()