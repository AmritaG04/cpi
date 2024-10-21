# classifiers.py
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt

def run_clustering(X_train_fft, X_train_qft, X_train, labels=None):
    if labels is None:
        labels = np.zeros(X_train.shape[0])  # Dummy labels for clustering

    clustering_methods = {
        'KNN': {
            'FFT': KNeighborsClassifier(n_neighbors=3).fit(X_train_fft, labels).predict(X_train_fft),
            'QFT': KNeighborsClassifier(n_neighbors=3).fit(X_train_qft, labels).predict(X_train_qft),
            'CNN': KNeighborsClassifier(n_neighbors=3).fit(X_train, labels).predict(X_train),
        },
        'One-Class SVM': {  
            'FFT': one_class_svm_clustering(X_train_fft),
            'QFT': one_class_svm_clustering(X_train_qft),
            'CNN': one_class_svm_clustering(X_train),
        },
        'Spectral Clustering': {  
            'FFT': spectral_clustering(X_train_fft),
            'QFT': spectral_clustering(X_train_qft),
            'CNN': spectral_clustering(X_train),
        }
    }
    return clustering_methods

# One-Class SVM for novelty detection
def one_class_svm_clustering(X_train):
    one_class_svm = OneClassSVM(kernel='linear', nu=0.1)
    one_class_svm.fit(X_train)
    return one_class_svm.predict(X_train)

# Spectral clustering function
def spectral_clustering(X_train, n_clusters=2):
    spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors')
    return spectral.fit_predict(X_train)

# Compute silhouette score
def compute_silhouette_score(X_fft, X_qft, X_cnn, kmeans_fft, kmeans_qft, kmeans_cnn):
    silhouette_fft = silhouette_score(X_fft, kmeans_fft.labels_)
    silhouette_qft = silhouette_score(X_qft, kmeans_qft.labels_)
    silhouette_cnn = silhouette_score(X_cnn, kmeans_cnn.labels_)
    
    print(f"Silhouette Score for FFT features: {silhouette_fft}")
    print(f"Silhouette Score for QFT features: {silhouette_qft}")
    print(f"Silhouette Score for CNN features: {silhouette_cnn}")
