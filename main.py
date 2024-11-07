
from data_loader import load_data, extract_vgg16_features
from feature_extraction import extract_features_with_qft, extract_features_with_fft
from classifiers import kmeans_clustering, evaluate_clustering
from visualize import visualize_features, visualize_pca_clusters
from sklearn.decomposition import PCA

data_path = "/content/gdrive/My Drive/DRIVE/test"

# Load data
generator = load_data(data_path)
features_df, image_filenames = extract_vgg16_features(generator)
X = features_df.values[:, -32:]

# QFT and FFT feature extraction
X_qft = extract_features_with_qft(X)
X_fft = extract_features_with_fft(X)

# Visualization
visualize_features(X_qft, X_fft)

# Clustering
pca = PCA(n_components=2)
fft_pca = pca.fit_transform(X_fft)
qft_pca = pca.fit_transform(X_qft)

labels_fft, _ = kmeans_clustering(X_fft)
labels_qft, _ = kmeans_clustering(X_qft)

visualize_pca_clusters(fft_pca, labels_fft, "K-Means Clustering on FFT Features")
visualize_pca_clusters(qft_pca, labels_qft, "K-Means Clustering on QFT Features")

# Evaluation
cb_fft, db_fft, silhouette_fft = evaluate_clustering(X_fft, labels_fft)
cb_qft, db_qft, silhouette_qft = evaluate_clustering(X_qft, labels_qft)

print(f"FFT - Calinski-Harabasz: {cb_fft}, Davies-Bouldin: {db_fft}, Silhouette: {silhouette_fft}")
print(f"QFT - Calinski-Harabasz: {cb_qft}, Davies-Bouldin: {db_qft}, Silhouette: {silhouette_qft}")
