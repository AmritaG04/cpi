from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import numpy as np


def kmeans_clustering(X_train, X_test, feature_type=""):
    kmeans = KMeans(n_clusters=2, random_state=42)
    cluster_labels_train = kmeans.fit_predict(X_train)
    cluster_labels_test = kmeans.predict(X_test)

    silhouette_kmeans = silhouette_score(X_train, cluster_labels_train)
    print(f"Silhouette Score for K-Means on {feature_type} features: {silhouette_kmeans}")

    return cluster_labels_train, silhouette_kmeans


def svm_classification(X_train, X_test, feature_type=""):
    svm = SVC(kernel='linear')
    svm.fit(X_train, np.zeros(X_train.shape[0]))  # Dummy labels for unsupervised classification

    svm_labels_train = svm.predict(X_train)
    svm_labels_test = svm.predict(X_test)

    print(f"SVM Classification completed on {feature_type} features.")
    return svm_labels_train


def gaussian_mixture_model(X_train, X_test, feature_type=""):
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(X_train)

    gmm_labels_train = gmm.predict(X_train)
    gmm_labels_test = gmm.predict(X_test)

    silhouette_gmm = silhouette_score(X_train, gmm_labels_train)
    print(f"Silhouette Score for GMM on {feature_type} features: {silhouette_gmm}")

    return gmm_labels_train, silhouette_gmm
