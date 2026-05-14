import os
import joblib
import numpy as np
import cv2
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

clinical_model = joblib.load(os.path.join(MODELS_DIR, "clinical_model.pkl"))
clinical_scaler = joblib.load(os.path.join(MODELS_DIR, "clinical_scaler.pkl"))
clinical_imputer = joblib.load(os.path.join(MODELS_DIR, "clinical_imputer.pkl"))

combined_model = joblib.load("models/tabular_combined_model.pkl")
combined_scaler = joblib.load("models/tabular_combined_scaler.pkl")
combined_imputer = joblib.load("models/tabular_combined_imputer.pkl")

image_model = joblib.load(os.path.join(MODELS_DIR, "image_xgb_model.pkl"))
image_scaler = joblib.load(os.path.join(MODELS_DIR, "image_xgb_scaler.pkl"))

resnet = ResNet50(weights="imagenet", include_top=False, pooling="avg")


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or unreadable")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


def extract_image_features(img):
    features = resnet.predict(img, verbose=0)
    return features


def predict_cardiotoxicity(clinical_input, image_path):
    clinical_input = np.array(clinical_input).reshape(1, -1)
    clinical_input = clinical_imputer.transform(clinical_input)
    clinical_input = clinical_scaler.transform(clinical_input)

    clinical_prob = clinical_model.predict_proba(clinical_input)[0][1]

    img = preprocess_image(image_path)
    img_features = extract_image_features(img)
    img_features = image_scaler.transform(img_features)

    image_prob = image_model.predict_proba(img_features)[0][1]

    final_prob = (clinical_prob + image_prob) / 2

    return {
        "clinical_risk": float(clinical_prob),
        "image_risk": float(image_prob),
        "final_risk": float(final_prob)
    }

def predict_clinical_functional(csv_path):
    df = pd.read_csv(csv_path, sep=";")

    for col in df.columns:
        df[col] = df[col].astype(str).str.replace(",", ".").astype(float)

    X = df.drop("CTRCD", axis=1)

    X = combined_imputer.transform(X)
    X = combined_scaler.transform(X)

    preds = combined_model.predict(X)
    probs = combined_model.predict_proba(X)[:, 1]

    return preds, probs

