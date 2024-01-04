import numpy as np
np.random.seed(123)
class LinearRegression:

    def __init__(self, learning_rate=1e-2, n_steps=2000, n_features=1):
        """
        Constructor method
        :param learning_rate: learning rate value (hyperparameter)
        :param n_steps: number of epochs for gradient descent (hyperparameter)
        :param n_features: number of features involved in regression
        """
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.theta = np.random.rand(n_features)



    def fit_fb(self, X, y):
        """
        Apply gradient descent in full batch mode to training samples, without regularization
        :param X: training samples with bias
        :param y: training target values
        :return: history of evolution about cost and theta during training steps
        """

        m = len(X)
        cost_history = np.zeros(self.n_steps)
        theta_history = np.zeros((self.n_steps, self.theta.shape[0]))

        for step in range(0, self.n_steps):
            preds = self.predict(X)
            error = preds - y
            self.theta = self.theta - (self.learning_rate * (1 / m) * np.dot(X.T, error))
            theta_history[step, :] = self.theta.T
            cost_history[step] = 1 / (2 * m) * np.dot(error.T, error)
        return cost_history, theta_history



    def fit_mb(self, X, y, batch_size):
        """
        Apply gradient descent in mini batch mode to training samples, without regularization
        :param X: training samples with bias
        :param y: training target values
        :param batch_size: size of the batches that we're going to use
        :return: history of evolution about cost and theta during training steps, and cost and theta during validation phase

        Note that if we want to return theta and cost during validation phase we have to uncomment the following lines.
        If we put batch_size = 1 we're performing a Stocastic Gradient Descent
        If we put batch_size = -1 we're performing a Gradient Descent in full batch mode
        """
        m = len(X)

        if batch_size == -1:
            batch_size = m

        #cost_history_validation_phase = []
        #theta_history_validation_phase = []
        cost_history = np.zeros(self.n_steps)
        theta_history = np.zeros((self.n_steps, self.theta.shape[0]))

        for step in range(0, self.n_steps):
            for start in range(0, m, batch_size):
                X_batch = X[start:start+batch_size]
                y_batch = y[start:start+batch_size]

                preds = self.predict(X_batch)
                error = preds - y_batch
                self.theta = self.theta - (self.learning_rate * (1/batch_size) * np.dot(X_batch.T, error))

                #cost_history_validation_phase.append(self.theta.T)
                #theta_history_validation_phase.append(1/(2 * batch) * np.dot(error.T, error))

            theta_history[step, :] = self.theta.T
            global_error = self.predict(X) - y
            cost_history[step] = 1/(2 * m) * np.dot(global_error.T, global_error)

        return cost_history, theta_history
        #return np.array(cost_history_validation_phase), cost_history_per_step, np.array(theta_history_validation_phase), theta_history_per_step



    def fit_normal_equations(self, X, y):
        """
        Apply normal equations to calculate thetas parameters
        :param X: training samples with bias
        :param y: training target values
        :return: thetas ready to use
        """
        self.theta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))
        return self.theta



    def predict(self, X):
        """
        Performs a complete prediction about the X samples provided in input
        :param X: test sample with shape (m, n_features)
        :return: prediction wrt X sample. The shape of return array is (m)
        """
        return np.dot(X, self.theta)