import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def visualize_pca(X_train, cluster_labels, feature_type=""):
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(X_train)

    plt.scatter(pca_features[:, 0], pca_features[:, 1], c=cluster_labels, cmap='viridis')
    plt.title(f"PCA of {feature_type} Features")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()
