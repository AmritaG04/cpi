from data_loader import load_and_extract_cnn_features
from feature_extraction import split_and_scale_features, extract_fft_features, qft_feature_extraction
from classifiers import kmeans_clustering, svm_classification, gaussian_mixture_model
from utils import visualize_pca


def main():

    data_path = "/content/gdrive/My Drive/DRIVE/test"
    features = load_and_extract_cnn_features(data_path)


    X_train, X_test = split_and_scale_features(features)


    X_train_fft = extract_fft_features(X_train)
    X_train_qft = qft_feature_extraction(X_train)

    
    print("\n--- K-Means Clustering ---")
    kmeans_cnn, _ = kmeans_clustering(X_train, X_test, "CNN")
    kmeans_fft, _ = kmeans_clustering(X_train_fft, X_test, "FFT")
    kmeans_qft, _ = kmeans_clustering(X_train_qft, X_test, "QFT")

    print("\n--- SVM Classification ---")
    svm_cnn = svm_classification(X_train, X_test, "CNN")
    svm_fft = svm_classification(X_train_fft, X_test, "FFT")
    svm_qft = svm_classification(X_train_qft, X_test, "QFT")

    print("\n--- Gaussian Mixture Model ---")
    gmm_cnn, _ = gaussian_mixture_model(X_train, X_test, "CNN")
    gmm_fft, _ = gaussian_mixture_model(X_train_fft, X_test, "FFT")
    gmm_qft, _ = gaussian_mixture_model(X_train_qft, X_test, "QFT")

    visualize_pca(X_train, kmeans_cnn, "CNN (VGG16)")
    visualize_pca(X_train_fft, kmeans_fft, "FFT")
    visualize_pca(X_train_qft, kmeans_qft, "QFT")


if __name__ == "__main__":
    main()
