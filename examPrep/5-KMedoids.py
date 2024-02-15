from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
import sklearn.datasets as datasets
import numpy as np

class KMedoids:
    def __init__(self, n_clusters, dist=euclidean_distances, random_state=42):
        self.n_clusters = n_clusters
        self.dist = dist
        self.random_state = random_state
        self.cluster_centers = []

    def fit(self, X):

        randInt = np.random.RandomState(self.random_state).randint
        indices = [randInt(X.shape[0])]

        for _ in range(0, self.n_clusters-1):
            newIndex = randInt(X.shape[0])
            while newIndex in indices:
                newIndex = randInt(X.shape[0])
            indices.append(newIndex)

        self.cluster_centers = X[indices, :]

        cost, y = self.compute_cost(X, indices)

        new_cost = cost
        new_y = y.copy()
        new_indices = indices[:]
        firstIteration = True

        while (new_cost < cost) | firstIteration:
            firstIteration = False

            cost = new_cost
            y = new_y
            indices = new_indices

            for k in range(0, self.n_clusters):
                k_cluster_indices = [i for i, x in enumerate(new_y == k) if x]

                for r in k_cluster_indices:

                    if r not in indices:
                        indices_tmp = indices[:]
                        indices_tmp[k] = r
                        new_cost_tmp, y_tmp = self.compute_cost(X, indices_tmp)

                        if new_cost_tmp < new_cost:
                            new_cost = new_cost_tmp
                            new_y = y_tmp
                            new_indices = indices_tmp

            self.cluster_centers = X[indices, :]

    def predict(self, X):
        return np.argmin(self.dist(X, self.cluster_centers), axis=1)


    def compute_cost(self, X, indices):
        y = np.argmin(self.dist(X, X[indices, :]), axis=1)
        cost_vector_by_cluster = [np.sum(self.dist(X[y == i], X[[indices[i]], :])) for i in set(y)]
        total_cost = np.sum(cost_vector_by_cluster)
        return total_cost, y

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

model = KMedoids(n_clusters=n_clusters, random_state=42)
model.fit(X)
y = model.predict(X)
print(f'Silhouette Score: {silhouette_score(X, y)}')
plt.scatter(X[:, 0], X[:, 1], c=y, s=n_clusters)
plt.show()