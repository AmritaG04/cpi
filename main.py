# main.py
from data_loader import load_dataset, extract_vgg16_features, preprocess_data
from feature_extraction import create_qft_circuit, extract_features_with_qft, extract_fft_features
from classifiers import run_clustering
from visualize import visualize_clustering_results

def main(data_path):
    # Load dataset
    generator = load_dataset(data_path)
    features = extract_vgg16_features(generator)

    # Preprocess data
    X_train, X_test = preprocess_data(features)
    
    # Extract QFT and FFT features
    qft_circuit = create_qft_circuit(num_qubits=2) 
    X_train_qft = extract_features_with_qft(X_train, qft_circuit, num_qubits=2)
    X_train_fft = extract_fft_features(X_train)

    clustering_results = run_clustering(X_train_fft, X_train_qft, X_train)
    visualize_clustering_results(X_train_fft, X_train_qft, X_train, clustering_results)

if __name__ == "__main__":
    data_path = "test"  
    main(data_path)
