import matplotlib.pyplot as plt
import numpy as np

# Set a seed for random number generation to ensure reproducibility
np.random.seed(42)



class RegressionNeuralNetwork:
    def __init__(self, layers, epochs=700, alpha=1e-2, lmd=1):
        """
        Constructor method for initializing the hyperparameters of our Neural Network.
        :param layers: a vector, containing the number of neurons for each layer.
        :param epochs: number of iterations used to train the NN.
        :param alpha: learning rate, for GD.
        :param lmd: regularization parameter.
        """

        self.layers = layers
        self.n_layers = len(layers)
        self.epochs = epochs
        self.alpha = alpha
        self.lmd = lmd

        # Initialize weights, biases, and loss variables
        self.w = {}
        self.b = {}
        self.loss = []



    def fit(self, X_train, y_train):
        """
        The fit method is used to train and find the correct weights and biases.

        :param X_train: training data.
        :param y_train: target feature of our training data.
        """

        self.loss = []
        self._init_parameters()

        for i in range(self.epochs):
            values = self._forward_propagation(X_train)
            grads = self._backpropagation_step(values, X_train, y_train)
            self._update(grads)

            cost = self._compute_cost(values, y_train)
            self.loss.append(cost)



    def predict(self, X_test):
        """
        The predict method is used to make inference on new data.

        :param X_test: data used to make inference.
        :return: predictions.
        """

        values = self._forward_propagation(X_test)
        return values["A" + str(self.n_layers - 1)]



    def plot_loss(self):
        """
        This method is used in order to plot the loss epoch by epoch.
        """
        plt.plot(self.loss)
        plt.xlabel("Epochs")
        plt.ylabel("Loss values")
        plt.title("Loss curve")
        plt.show()



    def _init_parameters(self):
        """
        This private method is used to initialize the weights and biases.
        """
        for i in range(1, self.n_layers):
            self.w[i] = np.random.randn(self.layers[i], self.layers[i - 1])
            self.b[i] = np.ones((self.layers[i], 1))



    def _forward_propagation(self, X):
        """
        Given the dataset, this method perform the forward propagation.
        :param X: training data.
        :return: all the values used to perform the calculations.
        """
        values = {}

        # For each layer (excluding the first layer):
        for i in range(1, self.n_layers):
            if i == 1:
                # If I'm in the second layer, I'll use X for the dot product
                values["Z" + str(i)] = np.dot(self.w[i], X.T) + self.b[i]
            else:
                # If I'm in the other layers, I'll use the previous outputs a for the dot product
                values["Z" + str(i)] = np.dot(self.w[i], values["A" + str(i - 1)]) + self.b[i]

            # For the output layer, set A equal to Z, if not apply the activation function
            if i == (self.n_layers - 1):
                values["A" + str(i)] = values["Z" + str(i)]
            else:
                values["A" + str(i)] = self._sigmoid(values["Z" + str(i)])

        return values



    def _compute_cost(self, values, y):
        """
        This method is used to compute the total cost.
        :param values: The entire history of values used to perform the forward propagation, useful to extract the predictions.
        :param y: The true values.
        :return: cost.
        """

        pred = values["A" + str(self.n_layers - 1)]

        # This is the MSE
        cost = np.average((y - pred) ** 2) / 2

        # This is the regularization part
        reg_sum = 0
        for i in range(1, self.n_layers):
            reg_sum += np.sum(np.square(self.w[i]))
        L2_reg = reg_sum * self.lmd

        return cost + L2_reg



    def _compute_cost_derivative(self, values, y):
        """
        This is the derivative of the cost function used.
        :param values: predicted values.
        :param y: true values.
        :return: cost derivative.
        """
        return values - y



    def _backpropagation_step(self, values, X, y):
        """
        This is the backpropagation step, in order to update the weights and biases.

        :param values: all the history of values used in the forward propagation.
        :param X: the training set.
        :param y: true values.
        :return: the updated parameters.
        """

        m = y.shape[0]
        params_upd = {}
        dZ = None

        for i in range(self.n_layers - 1, 0, -1):
            if i == (self.n_layers - 1):
                # For the output layer, compute the derivative of the cost function
                dA = self._compute_cost_derivative(values["A" + str(i)], y)
                dZ = dA
            else:
                # For hidden layers, compute the derivative using the chain rule
                dA = np.dot(self.w[i + 1].T, dZ)
                dZ = np.multiply(dA, self._sigmoid_derivative(values["A" + str(i)]))

            if i == 1:
                # Compute the weight and bias updates for the first layer
                params_upd["W" + str(i)] = (1 / m) * (np.dot(dZ, X) + self.lmd * self.w[i])
            else:
                # Compute the weight and bias updates for subsequent layers
                params_upd["W" + str(i)] = (1 / m) * (np.dot(dZ, values["A" + str(i - 1)].T) + self.lmd * self.w[i])

            # Compute the bias updates
            params_upd["B" + str(i)] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        # Return the computed parameter updates
        return params_upd



    def _update(self, upd):
        """
        This method is used to calculate the new parameters.
        :param upd: the updates calculated in the backpropagation step.
        """
        for i in range(1, self.n_layers):
            self.w[i] -= self.alpha * upd["W" + str(i)]
            self.b[i] -= self.alpha * upd["B" + str(i)]



    def _sigmoid(self, z):
        """
        This method is the activation function.
        :param z: input of the sigmoid function.
        :return: output of the sigmoid function.
        """
        return 1 / (1 + np.exp(-z))



    def _sigmoid_derivative(self, z):
        """
        This method is the derivative of the activation function.
        :param z: input of the derivative of the sigmoid function.
        :return: output of the derivative of the sigmoid function.
        """
        return z * (1 - z)