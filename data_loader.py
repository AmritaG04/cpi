
import pandas as pd
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input

def load_data(data_path):
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    generator = datagen.flow_from_directory(
        data_path,
        target_size=(224, 224),
        batch_size=10,
        class_mode=None,
        shuffle=False
    )
    return generator

def extract_vgg16_features(generator):
    model = VGG16(weights='imagenet', include_top=False)
    features = model.predict(generator)
    features_flattened = features.reshape(features.shape[0], -1)
    return pd.DataFrame(features_flattened), generator.filenames
