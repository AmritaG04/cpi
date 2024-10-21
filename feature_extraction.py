# feature_extraction.py
import numpy as np
import pennylane as qml

# Quantum Fourier Transform (QFT) circuit
def create_qft_circuit(num_qubits):
    device = qml.device("default.qubit", wires=num_qubits)
    
    @qml.qnode(device)
    def qft_circuit(features):
        norm = np.linalg.norm(features)
        if norm == 0:
            features = np.ones_like(features) * 1e-10
        qml.AmplitudeEmbedding(features=features, wires=range(num_qubits), pad_with=0.0, normalize=True)
        qml.QFT(wires=range(num_qubits))
        return qml.probs(wires=range(num_qubits))
    
    return qft_circuit

# Feature extraction using QFT
def extract_features_with_qft(data, qft_circuit, num_qubits):
    transformed_features = []
    for sample in data:
        transformed_sample = qft_circuit(sample[:num_qubits])
        transformed_features.append(transformed_sample)
    return np.array(transformed_features)

# FFT transformation
def extract_fft_features(data):
    fft_transformed = np.fft.fft(data, axis=1)
    return np.abs(fft_transformed)
