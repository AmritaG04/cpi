
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

def kmeans_clustering(features, n_clusters=2, random_state=42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(features)
    return labels, kmeans

def evaluate_clustering(features, labels):
    cb_index = calinski_harabasz_score(features, labels)
    db_index = davies_bouldin_score(features, labels)
    silhouette = silhouette_score(features, labels)
    return cb_index, db_index, silhouette
