import numpy as np
from PIL import Image
import io
from tensorflow.keras.applications.resnet50 import preprocess_input

def extract_image_features_from_bytes(image_bytes, cnn_model):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))

    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    features = cnn_model.predict(img_array)
    return features.flatten()
