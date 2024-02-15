import numpy as np
import pandas as pd
from matplotlib import pyplot as plt



class LogisticRegression:
    def __init__(self, learning_rate, epochs, num_features, threshold=0.5):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.num_features = num_features
        self.theta = np.random.randn(num_features)
        self.threshold = threshold

    def sigmoid(self, X):
        return 1 / (1+ np.exp(-np.dot(X, self.theta.T)))

    def fit(self, X, y, batch_size=-1):

        m = X.shape[0]

        if (batch_size == -1):
            batch_size = m

        cost_history = np.zeros(self.epochs)
        theta_history = np.zeros((self.epochs, self.num_features))

        for epoch in range(0, self.epochs):
            for start in range(0, m, batch_size):
                X_batch = X[start:start+batch_size, :]
                y_batch = y[start:start+batch_size]
                preds_batch = self.sigmoid(X_batch)
                error_batch = preds_batch - y_batch
                self.theta = self.theta - self.learning_rate*(1 / batch_size)*np.dot(X.T, error_batch)

            theta_history[epoch, :] = self.theta.T
            preds = self.sigmoid(X)
            error = preds-y
            cost_history[epoch] = -np.average(np.dot(y, np.log(preds)) + np.dot(1-y, np.log(1-preds)))

        return cost_history, theta_history

    def predict(self, X):
        #return np.round(self.sigmoid(X))
        preds = self.sigmoid(X)

        for index in range(0, len(preds)):
            if preds[index] >= self.threshold:
                preds[index] = 1
            else:
                preds[index] = 0

        return preds



class Evaluation:
    def __init__(self, model,  X, y):
        self.model = model
        self.y = y
        preds = model.predict(X)
        self.TP = np.sum(np.logical_and(preds, y))
        self.TN = np.sum(np.logical_and(np.logical_not(preds), np.logical_not(y)))
        self.FP = np.sum((preds - y) == 1)
        self.FN = np.sum((preds-y) == -1)

    def accuracy(self):
        return (self.TP + self.TN)/(self.TP + self.TN + self.FP + self.FN)

    def precision(self):
        return (self.TP) / (self.TP + self.FP)

    def recall(self):
        return  self.TP / (self.TP + self.FN)

    def f1_score(self):
        return 2 * self.TP / (2 * self.TP + self.FP + self.FN)

    def tpr(self):
        return self.TP / (self.TP + self.FN)

    def fpr(self):
        return self.FP / (self.FP + self.TN)


dataset = pd.read_csv("datasets/diabetes.csv")
dataset = dataset.values
X = dataset[:, 0:8]
y = dataset[:, 8]
train_index = round(X.shape[0] * 0.8)
X_train = X[0:train_index, :]
X_test = X[train_index:, :]
y_train = y[0:train_index]
y_test = y[train_index:]
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
X_train = (X_train - mean)/std
X_test = (X_test - mean)/std
X_train = np.column_stack((X_train, np.ones(X_train.shape[0])))
X_test = np.column_stack((X_test, np.ones(X_test.shape[0])))



model = LogisticRegression(learning_rate=0.1, epochs=1000, num_features=X_train.shape[1], threshold=0.5)
cost_history, theta_history = model.fit(X_train, y_train)
plt.plot(cost_history)
plt.show()



eval = Evaluation(model, X_test, y_test)
print(f'Accuracy on test set: {eval.accuracy()}')



# This is to construct a ROC curve

tpr = np.zeros(10)
fpr = np.zeros(10)

for threshold in range(0, 10):
    model = LogisticRegression(learning_rate=0.1, epochs=1000, num_features=X_train.shape[1], threshold=threshold/10)
    eval = Evaluation(model, X_test, y_test)
    fpr[threshold] = eval.fpr()
    tpr[threshold] = eval.tpr()

plt.plot(fpr, tpr)
plt.show()