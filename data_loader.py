import numpy as np
from google.colab import drive
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input


def load_and_extract_cnn_features(data_path, batch_size=10, target_size=(224, 224)):
    drive.mount('/content/gdrive')

    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    generator = datagen.flow_from_directory(
        data_path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=None,
        shuffle=False
    )

    model = VGG16(weights='imagenet', include_top=False)
    features = model.predict(generator)
    features_flattened = features.reshape(features.shape[0], -1)

    image_filenames = generator.filenames
    generator.reset()

    for i in range(len(image_filenames)):
        next(generator)
        print(f"Processed image {i + 1}/{len(image_filenames)}")

    return features_flattened
