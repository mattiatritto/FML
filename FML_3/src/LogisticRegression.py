import numpy as np
np.random.seed(123)

class LogisticRegression:

    def __init__(self, learning_rate=1e-2, n_steps=2000, n_features=1):
        """
        Constructor method
        :param learning_rate: learning rate for the gradient descent (hyperparameter)
        :param n_steps: number of epochs for gradient descent (hyperparameter)
        :param n_features: number of features to train the model
        """
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.theta = np.random.rand(n_features)



    def _sigmoid(self, z):
        """
        Internal method used to calculate the sigmoid function.
        :param z: variable for the sigmoid function
        :return: value from 0 to 1
        """
        return 1 / (1 + np.exp(-z))



    def fit_fb(self, X, y):
        """
        Apply Gradent Descent in full batch mode to train the model.
        :param X: training samples with bias
        :param y: training target values
        :return: history of evolution about cost and theta during training steps
        """

        m = len(X)
        cost_history = np.zeros(self.n_steps)
        theta_history = np.zeros((self.n_steps, self.theta.shape[0]))

        for step in range(0, self.n_steps):
            z = np.dot(X, self.theta)
            preds = self._sigmoid(z)
            error = preds - y
            self.theta = self.theta - (self.learning_rate * (1/m) * np.dot(X.T, error))
            theta_history[step, :] = self.theta.T
            cost_history[step] = -1 / m * (np.dot(y, np.log(preds)) + np.dot(1 - y, np.log(1 - preds)))
        return cost_history, theta_history



    def fit_mb(self, X, y, batch_size=10):
        """
        Apply gradient descent in mini batch mode to training samples, without regularization
        :param X: training samples with bias
        :param y: training target values
        :param batch_size: size of the batches that we're going to use
        :return: history of evolution about cost and theta during training steps

        If we put batch_size = 1 we're performing a Stocastic Gradient Descent
        If we put batch_size = -1 we're performing a Gradient Descent in full batch mode
        """
        m = len(X)

        if batch_size == -1:
            batch_size = m

        cost_history = np.zeros(self.n_steps)
        theta_history = np.zeros((self.n_steps, self.theta.shape[0]))

        for step in range(0, self.n_steps):
            total_error = np.zeros(X.shape[1])
            for start in range(0, m, batch_size):
                X_batch = X[start:start+batch_size]
                y_batch = y[start:start+batch_size]

                z = np.dot(X_batch, self.theta)
                preds = self._sigmoid(z)
                error = preds - y_batch
                total_error = total_error + np.dot(X_batch.T, error)

            self.theta = self.theta - (self.learning_rate * (1/m) * total_error)
            theta_history[step, :] = self.theta.T
            preds = self.predict(X)
            cost_history[step] = -1 / m * (np.dot(y, np.log(preds)) + np.dot(1 - y, np.log(1 - preds)))

        return cost_history, theta_history



    def fit_sgd(self, X, y):
        """
        Apply gradient descent in sgd mode to training samples, without regularization
        :param X: training samples with bias
        :param y: training target values
        :return: history of evolution about cost and theta during training steps
        """

        m = len(X)
        cost_history = np.zeros(self.n_steps)
        theta_history = np.zeros((self.n_steps, self.theta.shape[0]))

        for step in range(self.n_steps):
            random_index = np.random.randint(m)
            xi = X[random_index]
            yi = y[random_index]

            z = np.dot(xi, self.theta)
            preds = self._sigmoid(z)
            error = preds - yi
            self.theta = self.theta - self.learning_rate * xi.T.dot(error)
            z = np.dot(X, self.theta)
            predictions = self._sigmoid(z)
            theta_history[step, :] = self.theta.T
            cost_history[step] = -1 / m * (np.dot(y, np.log(preds)) + np.dot(1 - y, np.log(1 - preds)))

        return cost_history, theta_history



    def predict(self, X):
        """
        Performs a complete prediction about the X samples provided in input
        :param X: test sample with shape (m, n_features)
        :return: prediction wrt X sample. The shape of return array is (m)
        """
        return self._sigmoid(np.dot(X, self.theta))