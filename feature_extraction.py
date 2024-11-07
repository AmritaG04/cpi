
import numpy as np
import pennylane as qml
from sklearn.preprocessing import StandardScaler

num_qubits = 5
device = qml.device("default.qubit", wires=num_qubits)

@qml.qnode(device)
def qft_circuit(features):
    norm = np.linalg.norm(features)
    if norm == 0:
        features = np.ones_like(features) * 1e-10
    qml.AmplitudeEmbedding(features=features, wires=range(num_qubits), pad_with=0.0, normalize=True)
    qml.QFT(wires=range(num_qubits))
    return qml.probs(wires=range(num_qubits))

def extract_features_with_qft(data):
    transformed_features = [qft_circuit(sample[:num_qubits]) for sample in data]
    return np.array(transformed_features)

def extract_features_with_fft(data):
    fft_transformed = np.fft.fft(data, axis=1)
    return np.abs(fft_transformed)
