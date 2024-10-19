import numpy as np
import pennylane as qml

def split_and_scale_features(features, test_size=0.1, random_state=42):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    X = features[:, -32:]  
    X_train, X_test = train_test_split(X, test_size=test_size, random_state=random_state)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled


def extract_fft_features(X_train):
    fft_transformed = np.fft.fft(X_train, axis=1)
    X_train_fft = np.abs(fft_transformed)
    return X_train_fft


def qft_feature_extraction(X_train, num_qubits=5):
    device = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(device)
    def qft_circuit(features):
        qml.AmplitudeEmbedding(features=features, wires=range(num_qubits), pad_with=0.0, normalize=True)
        qml.QFT(wires=range(num_qubits))
        return qml.probs(wires=range(num_qubits))

    transformed_features = [qft_circuit(sample[:num_qubits]) for sample in X_train]
    return np.array(transformed_features)
