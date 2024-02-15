import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class NeuralNetwork:
    def __init__(self, layers, epochs=2000, alpha=0.01, lmd=0.1):
        self.layers = layers
        self.n_layers = len(layers)
        self.epochs = epochs
        self.alpha = alpha
        self.lmd = lmd

        self.w = {}
        self.b = {}
        self.loss = []

    def fit(self, X, y):
        self.init_parameters()

        for i in range(self.epochs):
            values = self.forward_propagation(X)
            grads = self.back_propagation(values, X, y)
            self.update(grads)
            cost = self.compute_cost(values, y)
            self.loss.append(cost)

    def predict(self, X):
        values = self.forward_propagation(X)
        return np.round(values["A" + str(self.n_layers - 1)])

    def plot_loss(self):
        plt.plot(self.loss)
        plt.xlabel("Epochs")
        plt.ylabel("Loss values")
        plt.title("Loss curve")
        plt.show()

    def init_parameters(self):
        for i in range(1, self.n_layers):
            self.w[i] = np.random.randn(self.layers[i], self.layers[i-1])
            self.b[i] = np.ones((self.layers[i], 1))

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    def sigmoid_derivative(self, X):
        return X * (1-X)

    def compute_cost(self, values, y):
        pred = values["A"+str(self.n_layers-1)]
        cost = -np.average(y.T*np.log(pred) + (1-y.T)*(1-pred))

        reg = 0
        for i in range(1, self.n_layers):
            reg = reg + np.sum(np.square(self.w[i]))
        reg = reg * self.lmd / (2 * len(y))
        return cost + reg

    def compute_cost_derivative(self, preds, y):
        return np.divide(y.T, preds) + np.divide(1-y, 1-preds)

    def update(self, upd):
        for i in range(1, self.n_layers):
            self.w[i] -= self.alpha*upd["W"+str(i)]
            self.b[i] -= self.alpha*upd["B" + str(i)]

        return
    def forward_propagation(self, X):

        values = {}
        for i in range(1, self.n_layers):
            if i == 1:
                values["Z" + str(i)] = np.dot(self.w[i], X.T) +self.b[i]
            else:
                values["Z" + str(i)] = np.dot(self.w[i], values["A" + str(i-1)]) + self.b[i]
            values["A" + str(i)] = self.sigmoid(values["Z"+str(i)])
        return values

    def back_propagation(self, values, X, y):
        m = len(y)
        upd = {}
        dZ = None

        for i in range(self.n_layers-1, 0, -1):
            if i == self.n_layers-1:
                dA = self.compute_cost_derivative(values["A" + str(i)], y)
            else:
                dA = np.dot(self.w[i+1].T, dZ)

            dZ = np.multiply(dA, self.sigmoid_derivative(values["A"+str(i)]))

            if i == 1:
                upd["W"+str(i)] = (1 / m)* (np.dot(dZ, X) + self.lmd*self.w[i])
            else:
                upd["W" + str(i)] = (1 / m) * (np.dot(dZ, values["A" + str(i - 1)].T) + self.lmd * self.w[i])

            upd["B"+str(i)] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        return upd


dataset = pd.read_csv("datasets/diabetes.csv")
dataset = dataset.values
X = dataset[:, 0:8]
y = dataset[:, 8]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

model = NeuralNetwork(layers=[X.shape[1], 5, 5, 1])
model.fit(X_train, y_train)
model.predict(X_test)
model.plot_loss()