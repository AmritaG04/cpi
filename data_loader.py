# data_loader.py
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_dataset(data_path):
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    generator = datagen.flow_from_directory(
        data_path,
        target_size=(224, 224),
        batch_size=10,
        class_mode=None,  
        shuffle=False  
    )
    return generator

# Extract features using VGG16
def extract_vgg16_features(generator):
    model = VGG16(weights='imagenet', include_top=False)  
    features = []
    for _ in range(len(generator)):
        batch = next(generator)  
        batch_features = model.predict(batch)  
        features.append(batch_features)

    features = np.vstack(features) 
    features_flattened = features.reshape(features.shape[0], -1)
    return features_flattened

# Split and scale data
def preprocess_data(X):
    X_train, X_test = train_test_split(X, test_size=0.1, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test
