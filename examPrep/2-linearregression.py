import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt



class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000, numFeatures=1):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.theta = np.random.randn(numFeatures)

    def fit(self, X, y, batch_size=-1):

        m = X.shape[0]

        if batch_size == -1:
            batch_size = m

        cost_history = np.zeros(self.epochs)
        theta_history = np.zeros((self.epochs, self.theta.shape[0]))

        for epoch in range(self.epochs):
            for start in range(0, m, batch_size):
                X_batch = X[start:start+batch_size, :]
                y_batch = y[start:start+batch_size]

                preds = self.predict(X_batch)
                error = preds-y_batch

                self.theta = self.theta - (self.learning_rate *(1/batch_size)*np.dot(X_batch.T, error))

            theta_history[epoch, :] = self.theta.T
            global_error = self.predict(X) - y
            cost_history[epoch] = 1 / (2*m) * np.dot(global_error.T, global_error)

        return cost_history, theta_history

    def fit_normal_equations(self, X, y):
        self.theta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))
        return self.theta

    def predict(self, X):
        return np.dot(X, self.theta)



class Evaluation:
    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y
        self.preds = model.predict(X)

    def mae(self):
        return np.average(np.abs(self.preds - self.y))

    def mse(self):
        return np.average((self.preds-self.y)**2)

    def rmse(self):
        return np.average(np.sqrt((self.preds-self.y)**2))

    def mape(self):
        return np.average(np.abs(self.preds-self.y)/self.y)*100

    def r2(self):
        tss = np.sum((self.y - self.y.mean()) ** 2)
        rss = np.sum((self.preds - self.y) ** 2)
        return 1 - (rss / tss)



dataset = pd.read_csv("datasets/houses_portaland_simple.csv")
dataset = dataset.values
X = dataset[:, 0:2]
X = np.column_stack((X, np.ones(X.shape[0])))
y = dataset[:, 2]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LinearRegression(0.01, 1000, X.shape[1])
cost_history, theta_history = model.fit(X_train, y_train)
plt.plot(cost_history)
plt.show()

print(f"Theta found: {theta_history[-1, :]} ")
eval = Evaluation(model, X_test, y_test)
print(f"R2 score: {eval.r2()}")