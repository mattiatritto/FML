from sklearn.metrics.pairwise import euclidean_distances
import numpy as np



class KMedoids(object):
    def __init__(self, n_clusters=2, dist=euclidean_distances, random_state=42):
        """
        This class implements the KMedoids algorithm.

        :param n_clusters: number of clusters in which we want to group data
        :param dist: type of distance to measure distances between data points
        :param random_state: seed for random number generation
        """

        self.n_clusters = n_clusters
        self.dist = dist
        self.randomInt = np.random.RandomState(random_state).randint
        self.cluster_centers = []



    def fit(self, X):
        """
        Method used to train the KMedoids algorithm.

        :param X: training data
        """

        # Initializing the first medoid index randomly
        indices = [self.randomInt(X.shape[0])]

        # And now we define the other n_clusters-1 other indices randomly
        for _ in range(self.n_clusters-1):
            i = self.randomInt(X.shape[0])
            while i in indices:
                i = self.randomInt(X.shape[0])
            indices.append(i)

        # Setting the cluster medoids
        self.cluster_centers = X[indices, :]

        # Initial cost and labels with current medoids
        cost, y_pred = self._compute_cost(X, indices)

        # Variables to track the best configuration
        new_cost = cost
        new_y_pred = y_pred.copy()
        new_indices = indices[:]
        firstIteration = True

        while(new_cost < cost) | firstIteration:

            firstIteration = False

            # Update cost and labels with the best configuration
            cost = new_cost
            y_pred = new_y_pred
            indices = new_indices

            for k in range(self.n_clusters):
                # Find indices of data points belonging to the current cluster
                k_cluster_indices = [i for i, x in enumerate(new_y_pred == k) if x]

                # For each data point in the current cluster
                for r in k_cluster_indices:

                    # If the data point is not already a medoid
                    if r not in indices:
                        indices_tmp = indices[:]
                        indices_tmp[k] = r
                        new_cost_tmp, y_pred_tmp = self._compute_cost(X, indices_tmp)

                        if new_cost_tmp < new_cost:
                            new_cost = new_cost_tmp
                            new_y_pred = y_pred_tmp
                            new_indices = indices_tmp

            # Update the final medoids indices based on the best configuration
            self.cluster_centers = X[indices, :]



    def predict(self, X):
        """
        Method used to predict the label of each data point in X.

        :param X: data
        :return: Array containing the predicted cluster indices for each sample.
        """

        return np.argmin(self.dist(X, self.cluster_centers), axis=1)



    def _compute_cost(self, X, indices):
        """
        Private method used to compute the cost of the current medoid configuration.

        :param X: data
        :param indices: list of the current medoid indices
        :return: total cost of the current medoid configuration, predicted labels for samples
        """

        y_pred = np.argmin(self.dist(X, X[indices, :]), axis=1)

        # The total cost is calculated by summing the distances from each point to its assigned medoid
        total_cost = np.sum(
            [
                np.sum(self.dist(X[y_pred == i], X[[indices[i]], :])) for i in set(y_pred)
            ]
        )

        return total_cost, y_pred