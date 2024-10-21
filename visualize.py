# visualize.py
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Visualization
def visualize_features(qft_features, fft_features):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("QFT Output Probabilities")
    plt.imshow(qft_features.T, aspect='auto', cmap='viridis')
    plt.colorbar(label='Probability')
    plt.xlabel('Samples')
    plt.ylabel('QFT States')

    plt.subplot(1, 2, 2)
    plt.title("FFT Magnitude")
    plt.imshow(fft_features.T, aspect='auto', cmap='viridis')
    plt.colorbar(label='Magnitude')
    plt.xlabel("Frequency Index")
    plt.ylabel("Magnitude")
    plt.tight_layout()
    plt.show()

# Clustering and PCA visualization
def visualize_clustering_results(X_train_fft, X_train_qft, X_train, clustering_results):
    pca_fft = PCA(n_components=2).fit_transform(X_train_fft)
    pca_qft = PCA(n_components=2).fit_transform(X_train_qft)
    pca_cnn = PCA(n_components=2).fit_transform(X_train)

    plt.figure(figsize=(18, 6))
    
    for i, (method, results) in enumerate(clustering_results.items()):
        plt.subplot(1, 4, i+1)
        plt.title(method)
        plt.scatter(pca_fft[:, 0], pca_fft[:, 1], c=results['FFT'], cmap='viridis', label='FFT')
        plt.scatter(pca_qft[:, 0], pca_qft[:, 1], c=results['QFT'], cmap='plasma', label='QFT', alpha=0.7)
        plt.scatter(pca_cnn[:, 0], pca_cnn[:, 1], c=results['CNN'], cmap='coolwarm', label='CNN', alpha=0.5)
        plt.legend()
    
    plt.tight_layout()
    plt.show()
