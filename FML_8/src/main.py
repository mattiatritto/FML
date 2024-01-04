import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from FML_8.src.KMeans import KMeans
from FML_8.src.KMedoids import KMedoids

# Variables to modify
random_state = 42
n_clusters = 3



# Dataset generation and preprocessing steps
X, y_true = datasets.make_blobs(
    n_samples=500,
    n_features=2,
    centers=3,
    cluster_std=2,
    center_box=(-10.0, 10.0),
    shuffle=True,
    random_state=random_state,
)
sc = StandardScaler()
X = sc.fit_transform(X)



# KMeans
model = KMeans(n_clusters=n_clusters, random_state=random_state)
model.fit(X)

y_pred = model.predict(X)

print(f"Silhouette Coefficient K-Means: {silhouette_score(X, y_pred)}")
plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=n_clusters)
plt.show()



#KMedoids
model = KMedoids(n_clusters=n_clusters, random_state=random_state)
model.fit(X)

y_pred = model.predict(X)

print(f"Silhouette Coefficient K-Medoids: {silhouette_score(X, y_pred)}")
plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=n_clusters)
plt.show()