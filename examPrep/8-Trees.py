import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree

dataset = pd.read_csv("datasets/diabetes.csv")
X = dataset.drop(labels=["Outcome"], axis=1).values
y = dataset["Outcome"].values
train_proportion = 0.8
train_index = round(X.shape[0] * train_proportion)

X_train = X[0:train_index, :]
y_train = y[0:train_index]
X_test = X[train_index:, :]
y_test = y[train_index:]

mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
X_train = (X_train - mean)/std
X_test = (X_test - mean)/std

# This is for tree classifier
model = DecisionTreeClassifier(max_depth=10, min_samples_leaf=10)
#model = RandomForestClassifier(max_depth=10, min_samples_leaf=10, n_estimators=100, criterion='entropy')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Accuracy score: {accuracy_score(y_test, y_pred)}")
plt.figure(figsize=(12, 8))
plot_tree(model, feature_names=dataset.columns, filled=True)
plt.show()