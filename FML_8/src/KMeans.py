from sklearn.metrics.pairwise import euclidean_distances
import numpy as np



class KMeans(object):
    def __init__(self, n_clusters=2, dist=euclidean_distances, random_state=42):
        """
        This class implements the KMeans algorithm.

        :param n_clusters: number of clusters in which we want to group data
        :param dist: type of distance to measure distances between data points
        :param random_state: seed for random number generation
        """

        self.n_clusters = n_clusters
        self.dist = dist
        self.randomInt = np.random.RandomState(random_state).randint
        self.cluster_centers = []
        self.y_pred = None



    def fit(self, X):
        """
        Method used to train the KMeans algorithm.

        :param X: training data
        """

        # Initializing the first centroid index randomly
        initial_indices = [self.randomInt(X.shape[0])]

        # And now we define the other n_clusters-1 other indices randomly
        for _ in range(self.n_clusters-1):
            i = self.randomInt(X.shape[0])
            while i in initial_indices:
                i = self.randomInt(X.shape[0])
            initial_indices.append(i)

        # Setting the cluster centers
        self.cluster_centers = X[initial_indices, :]

        # Main loop for KMeans algorithm
        continue_condition = True
        while continue_condition:
            old_centroids = self.cluster_centers.copy()
            self.y_pred = self.predict(X)
            for i in set(self.y_pred):
                self.cluster_centers[i] = np.mean(X[self.y_pred == i], axis=0)

            # KMeans stops if all centroids are the same
            if (old_centroids == self.cluster_centers).all():
                continue_condition = False



    def predict(self, X):
        """
        Method used to predict the label of each data point in X.

        :param X:
        :return: Array containing the predicted cluster indices for each sample.
        """

        return np.argmin(self.dist(X, self.cluster_centers), axis=1)