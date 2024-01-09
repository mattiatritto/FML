import numpy as np

np.random.seed(123)

class LinearRegression:
    """
    Class to perform learning for a linear regression. This class has all methods to be trained with different strategies
    and one method to produce a full prediction based on input samples. Moreover, this one is equipped by one method to
    measure performance and another method to build learning curves
    """
    def __init__(self, learning_rate=1e-2, n_steps=2000, n_features=1):
        """
        :param learning_rate: learning rate value
        :param n_steps: number of epochs around gd
        :param n_features: number of features involved in regression
        :param lmd: regularization factor

        lmd_ is an array useful when is necessary compute theta's update with regularization factor
        """
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.theta = np.random.rand(n_features)

    def fit(self, X, y, batch):
        """
        apply gradient descent in full batch mode, without regularization, to training samples and return evolution
        history of train and validation cost.
        :param X: training samples with bias
        :param y: training target values
        :param batch: batch size value
        :return: history of evolution about cost and theta during training steps and, cost during validation phase
        """
        m = len(X)

        if batch == -1:
            batch = m

        cost_history = []
        cost_history_plot = []
        theta_history = []
        for step in range(0, self.n_steps):
            for start in range(0, m, batch):
                stop = start + batch
                preds = np.dot(X[start:stop], self.theta)

                error = preds - y[start:stop]

                self.theta = self.theta - (self.learning_rate * (1/batch) * np.dot(X[start:stop].T, error))
                theta_history.append(self.theta.T)
                cost_history_plot.append(1/(2 * batch) * np.dot(error.T, error))
            # compute global error after exploit each sample
            ge = np.dot(X, self.theta) - y
            cost_history.append(1/(2 * m) * np.dot(ge.T, ge))

        return np.array(cost_history), np.array(cost_history_plot), np.array(theta_history)

    def predict(self, X):
        """
        perform a complete prediction about X samples
        :param X: test sample with shape (m, n_features)
        :return: prediction wrt X sample. The shape of return array is (m,)
        """
        return np.dot(X, self.theta)