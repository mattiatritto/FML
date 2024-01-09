import operator
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np
from linear_regression import LinearRegression
import matplotlib.pyplot as plt

# read the dataset of houses prices
salary = pd.read_csv('datasets/Position_Salaries_base.csv')

plt.plot(salary.Level, salary.Salary, 'r.')
plt.show()

# another way to test the correlation
print(salary.corr())

salary = salary.values

# compute mean and standard deviation ONLY ON TRAINING SAMPLES
mean = salary.mean(axis=0)
std = salary.std(axis=0)

# apply mean and std (standard deviation) compute on training sample to training set and to test set
salary = (salary - mean) / std


x = salary[:, 0]
y = salary[:, 1]


x_squared = x**2
x_cubic = x**3
# x_4 = x**4
x = np.column_stack((x,x_squared,x_cubic))

# poly = PolynomialFeatures(1)
# x = poly.fit_transform(x.reshape(-1, 1))
# add bias column
x = np.c_[np.ones(x.shape[0]), x]

# create a regressor with specific characteristics
linear = LinearRegression(n_features=x.shape[1], n_steps=1000, learning_rate=0.01)

cost_history, theta_history = linear.fit(x, y)

print(f'''Thetas: {*linear.theta,}''')
print(f'''Final train cost/MSE:  {cost_history[-1]:.3f}''')

pred = linear.predict(x)

sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(x[:, 1], pred), key=sort_axis)
x_poly, y_poly_pred = zip(*sorted_zip)

plt.plot(x[:, 1], y, 'r.', label='Training data')
plt.plot(x_poly, pred, 'b--', label='Current hypothesis')
plt.legend()
plt.show()
plt.show()
