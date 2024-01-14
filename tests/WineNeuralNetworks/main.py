import numpy as np
import pandas as pd
from FML.tests.WineNeuralNetworks.RegressionMetrics import RegressionMetrics
from FML.tests.WineNeuralNetworks.RegressionNeuralNetwork import RegressionNeuralNetwork

# Settings
k = 5
test_size = 1/5



# Pre processing steps
dataset = pd.read_csv("Wine.csv")
dataset = dataset.values
X = dataset[:, 0:11]
y = dataset[:, 11]
testSamples = round(X.shape[0]*test_size)
average = np.mean(X, 0)
std = np.std(X, 0)
X = (X - average) / std



# Grid search on these hyperparameters values
lambdas = [0.5, 1, 2, 3]
alphas = [0.001, 0.01, 0.1, 1]
neuronsFirstHiddenLayers = [2, 3, 4, 5]
neuronsSecondHiddenLayers = [2, 3, 4, 5]
k_folds = [0, 1, 2, 3, 4]

rmseOld = 9999999999
rmseNew = None
bestModel = None
bestLmd = None
bestAlpha = None
bestNeuronsFirstHiddenLayer = None
bestNeuronsSecondHiddenLayer = None

for lmd in lambdas:
    for alpha in alphas:
        for neuronsFirstHiddenLayer in neuronsFirstHiddenLayers:
            for neuronsSecondHiddenLayer in neuronsSecondHiddenLayers:
                for k_fold in k_folds:

                    start = k_fold*testSamples
                    end = start+testSamples

                    X_train_prima = X[:start, :]
                    X_train_dopo = X[end:, :]
                    X_train = np.vstack((X_train_prima, X_train_dopo))
                    X_test = X[start:end, :]

                    y_train_prima = y[:start]
                    y_train_dopo = y[end:]
                    y_train = np.concatenate((y_train_prima, y_train_dopo))
                    y_test = y[start:end]

                    nn = RegressionNeuralNetwork([X_train.shape[1], neuronsFirstHiddenLayer, neuronsSecondHiddenLayer, 1], epochs=50, alpha=alpha, lmd=lmd)
                    nn.fit(X_train, y_train)
                    metrics = RegressionMetrics(nn)
                    dictionary = metrics.compute_performance(X_test, y_test)

                    rmseNew = rmseNew + dictionary["rmse"]

                rmseNew = rmseNew / k

                if (rmseNew < rmseOld):
                    rmseOld = rmseNew
                    bestModel = nn
                    bestLmd = lmd
                    bestAlpha = alpha
                    bestNeuronsFirstHiddenLayer = neuronsFirstHiddenLayer
                    bestNeuronsSecondHiddenLayer = neuronsSecondHiddenLayer



print(f"Best model has RMSE = {rmseNew}")
print(f"Best model has lambda = {bestLmd}")
print(f"Best model has alpha = {bestAlpha}")
print(f"Best model has neuronsFirstHiddenLayer = {bestNeuronsFirstHiddenLayer}")
print(f"Best model has neuronsSecondHiddenLayer = {bestNeuronsSecondHiddenLayer}")

sampleTest = np.array([6.0, 0.31, 0.47, 3.6, 0.067, 18.0, 42.0, 0.99549, 3.39, 0.66, 11.0])
bestModel.plot_loss()
pred = bestModel.predict(sampleTest)
print(f"Prediction on the sample data = {pred}")