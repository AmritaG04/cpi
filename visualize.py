
import matplotlib.pyplot as plt
import numpy as np

def visualize_features(qft_features, fft_features):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 2)
    plt.title("QFT Output Probabilities")
    plt.imshow(qft_features.T, aspect='auto', cmap='viridis')
    plt.colorbar(label='Probability')
    plt.xlabel('Samples')
    plt.ylabel('QFT States')

    plt.subplot(1, 3, 3)
    plt.imshow(fft_features.T, aspect='auto', cmap='viridis')
    plt.colorbar(label='Magnitude')
    plt.title("FFT of Last 32 Features")
    plt.xlabel("Frequency Index")
    plt.ylabel("Magnitude")

    plt.tight_layout()
    plt.show()

def visualize_pca_clusters(data, labels, title):
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
    plt.title(title)
    plt.show()
