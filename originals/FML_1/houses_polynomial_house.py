import operator
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np

from FML_1.evaluation import Evaluation
from linear_regression import LinearRegression
import matplotlib.pyplot as plt

# read the dataset of houses prices
houses = pd.read_csv('datasets/houses.csv')

# shuffling all samples to avoid group bias
houses = houses.sample(frac=1).reset_index(drop=True)

# select only some features, also you can try with other features
x = houses[['GrLivArea', 'LotArea']].values

# select target value
y = houses['SalePrice'].values

x_square = x**2
x_cubic = x**3
x_4 = x**4
x = np.column_stack((x,))

# poly = PolynomialFeatures(2, include_bias=False)
# x = poly.fit_transform(x)

# in order to perform hold-out splitting 80/20 identify max train index value
train_index = round(len(x) * 0.8)

# split dataset into training and test
X_train = x[:train_index]
y_train = y[:train_index]

X_test = x[train_index:]
y_test = y[train_index:]

# compute mean and standard deviation ONLY ON TRAINING SAMPLES
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

# apply mean and std (standard deviation) compute on training sample to training set and to test set
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# add bias column
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

# create a regressor with specific characteristics
linear = LinearRegression(n_features=X_train.shape[1], n_steps=1000, learning_rate=0.05)

cost_history, theta_history = linear.fit(X_train, y_train)

print(f'''Thetas: {*linear.theta,}''')
print(f'''Final train cost/MSE:  {cost_history[-1]:.3f}''')

pred = linear.predict(X_test)

sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(X_test[:, 1], pred), key=sort_axis)
x_poly, y_poly_pred = zip(*sorted_zip)

plt.plot(X_test[:, 1], y_test, 'r.', label='Training data')
plt.plot(x_poly, y_poly_pred, 'b--', label='Current hypothesis')
plt.legend()
plt.show()
plt.show()


eval = Evaluation(linear)

print(eval.compute_performance(X_test, y_test))