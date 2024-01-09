import pandas as pd
import numpy as np
from linear_regression import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import io
import base64
from IPython.display import HTML


# read the dataset of houses prices
houses = pd.read_csv('datasets/houses_portaland_simple.csv')

# print dataset stats
print(houses.describe())
houses.drop('Bedroom', axis=1, inplace=True)
houses = houses.values

# compute mean and standard deviation ONLY ON TRAINING SAMPLES
mean = houses.mean(axis=0)
std = houses.std(axis=0)

# apply mean and std (standard deviation) compute on training sample to training set and to test set
houses = (houses - mean) / std

# in order to perform hold-out splitting 80/20 identify max train index value
train_index = round(len(houses) * 0.8)

x = houses[:, 0]
y = houses[:, 1]

# add bias column
X_train = np.c_[np.ones(x.shape[0]), x]

# create a regressor with specific characteristics
linear = LinearRegression(n_features=X_train.shape[1], n_steps=500, learning_rate=0.01)
linear.theta = np.array([1, -0.9])

lineX = np.linspace(X_train[:, 1].min(), X_train[:, 1].max(), 100)
liney = [linear.theta[0] + linear.theta[1]*xx for xx in lineX]

# fit (try different strategies) your trained regressor
cost_history, cost_history_plot, theta_history = linear.fit(X_train, y, 10)

#Grid over which we will calculate J
theta0_vals = np.linspace(-2, 2, 100)
theta1_vals = np.linspace(-2, 3, 100)

#initialize J_vals to a matrix of 0's
J_vals = np.zeros((theta0_vals.size, theta1_vals.size))

#Fill out J_vals
for t1, element in enumerate(theta0_vals):
    for t2, element2 in enumerate(theta1_vals):
        thetaT = np.zeros(shape=(2, 1))
        thetaT[0][0] = element
        thetaT[1][0] = element2
        h = X_train.dot(thetaT.flatten())
        j = (h - y)
        J = j.dot(j) / 2 / (len(X_train))
        J_vals[t1, t2] = J

#Contour plot
J_vals = J_vals.T

A, B = np.meshgrid(theta0_vals, theta1_vals)
C = J_vals

# Set the plot up,
fig = plt.figure(figsize=(12, 5))

plt.subplot(121)
plt.plot(X_train[:, 1], y, 'ro', label='Training data')
plt.title('Housing Price Prediction')
plt.axis([X_train[:, 1].min() - X_train[:, 1].std(), X_train[:, 1].max() + X_train[:, 1].std(),
          y.min() - y.std(), y.max() + y.std()])
plt.grid(axis='both')
plt.xlabel("Size of house in ft^2 (X1) ")
plt.ylabel("Price in $1000s (Y)")
plt.legend(loc='lower right')

line, = plt.plot([], [], 'b-', label='Current Hypothesis')
annotation = plt.text(-2, 3, '', fontsize=20, color='green')
annotation.set_animated(True)

plt.subplot(122)
cp = plt.contourf(A, B, C)
plt.colorbar(cp)
plt.title('Filled Contours Plot')
plt.xlabel('theta 0')
plt.ylabel('theta 1')
track, = plt.plot([], [], 'r-')
point, = plt.plot([], [], 'ro')

plt.tight_layout()
plt.close()


# Generate the animation data,
def init():
    line.set_data([], [])
    track.set_data([], [])
    point.set_data([], [])
    annotation.set_text('')
    return line, track, point, annotation


# animation function.  This is called sequentially
def animate(i):
    fit1_X = np.linspace(X_train[:, 1].min() - X_train[:, 1].std(), X_train[:, 1].max() + X_train[:, 1].std(), 1000)
    fit1_y = theta_history[i][0] + theta_history[i][1] * fit1_X

    fit2_X = theta_history.T[0][:i]
    fit2_y = theta_history.T[1][:i]

    track.set_data(fit2_X, fit2_y)
    line.set_data(fit1_X, fit1_y)
    point.set_data(theta_history.T[0][i], theta_history.T[1][i])

    annotation.set_text('Cost = %.4f' % (cost_history[i]))
    return line, track, point, annotation


anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=500, interval=0, blit=True)

anim.save('animation.gif', writer='imagemagick', fps=30)

filename = 'animation.gif'

video = io.open(filename, 'r+b').read()
encoded = base64.b64encode(video)
HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))