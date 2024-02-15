import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import euclidean_distances, silhouette_score
from sklearn import datasets

class KMeans:
    def __init__(self, n_clusters=2, random_state=42, dist=euclidean_distances):
        self.n_clusters = n_clusters
        self.dist = dist
        self.y = None
        self.centroids = None
        self.random_state = random_state

    def fit(self, X):
        randInt = np.random.RandomState(self.random_state).randint

        # Set the first centroid randomly
        initalIndices = [randInt(X.shape[0])]
        # And then, set the other n_cluster-1 indices

        for _ in range(0, self.n_clusters-1):
            i = randInt(X.shape[0])
            while i in initalIndices:
                i = randInt(X.shape[0])
            initalIndices.append(i)

        self.centroids = X[initalIndices, :]

        continue_condition = True
        while continue_condition:
            old_centroids = self.centroids.copy()
            self.y = self.predict(X)
            for i in set(self.y):
                self.centroids[i] = np.mean(X[self.y == i], axis=0)
            if (old_centroids == self.centroids).all():
                continue_condition = False

    def predict(self, X):
        return np.argmin(self.dist(X, self.centroids), axis=1)

    # Non funziona, ma la struttura è acccettabile
    def silhouette_score(self, X):
        y = self.predict(X)

        accumulator = 0
        for index in range(0, X.shape[0]):
            cluster = y[index]
            indices_cluster = []
            for i in range(0, X.shape[0]):
                if y[i] == cluster:
                    indices_cluster.append(i)
            coh = []
            for i in indices_cluster:
                coh.append(self.dist(X[index, :], X[i, :]))

            coh = np.mean(coh)

            indices_cluster = []
            for i in range(0, X.shape[0]):
                if y[i] != cluster:
                    indices_cluster.append(i)
            sep = np.mean(self.dist(X[index, :], X[indices_cluster, :]))
            accumulator = accumulator + (coh/sep)

        return accumulator/X.shape[0]

random_state = 42
n_clusters = 3
X, y = datasets.make_blobs(
    n_samples=500,
    n_features=2,
    centers=3,
    cluster_std=2,
    center_box=(-10.0, 10.0),
    shuffle=True,
    random_state=random_state,
)

# KMeans
model = KMeans(n_clusters=n_clusters, random_state=random_state)
model.fit(X)
y_pred = model.predict(X)
print(f'Silhouette Score: {silhouette_score(X, y_pred)}')
plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=n_clusters)
plt.show()