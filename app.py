from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import base64

from tensorflow.keras.applications.resnet50 import ResNet50
from image_feature_extractor import extract_image_features_from_bytes

app = Flask(__name__)
CORS(app)

clinical_model = joblib.load("models/clinical_model.pkl")
clinical_scaler = joblib.load("models/clinical_scaler.pkl")
clinical_imputer = joblib.load("models/clinical_imputer.pkl")

image_model = joblib.load("models/image_xgb_model.pkl")
image_scaler = joblib.load("models/image_xgb_scaler.pkl")

cnn_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

all_features = [
    'heart_rate','age','weight','height','time','heart_rhythm','LVEF','PWT',
    'LAd','LVDd','LVSd','AC','antiHER2','HTA','DL','DM','smoker','exsmoker',
    'ACprev','antiHER2prev','RTprev','CIprev','ICMprev','ARRprev','VALVprev','cxvalv'
]

def preprocess_clinical(ui):
    row = {
        "heart_rate": ui.get("hr", 0),
        "age": ui.get("age", 0),
        "weight": ui.get("weight", 70),
        "height": ui.get("height", 165),
        "time": ui.get("time", 0),
        "heart_rhythm": 0,
        "LVEF": ui.get("lvef", 55),
        "PWT": 0,
        "LAd": 0,
        "LVDd": 0,
        "LVSd": 0,
        "AC": 0,
        "antiHER2": 1,
        "HTA": 0,
        "DL": 0,
        "DM": 0,
        "smoker": 0,
        "exsmoker": 0,
        "ACprev": 0,
        "antiHER2prev": 0,
        "RTprev": 0,
        "CIprev": 0,
        "ICMprev": 0,
        "ARRprev": 0,
        "VALVprev": 0,
        "cxvalv": 0
    }

    df = pd.DataFrame([row])[all_features]
    df = clinical_imputer.transform(df)
    df = clinical_scaler.transform(df)
    return df


def preprocess_image(image_bytes):
    feats = extract_image_features_from_bytes(image_bytes, cnn_model)
    feats = np.array(feats).reshape(1, -1)

    print("Image feature shape:", feats.shape)

    feats = image_scaler.transform(feats)
    return feats


def apply_clinical_rules(ui, score):
    age = ui.get("age", 0)
    hr = ui.get("hr", 0)
    lvef = ui.get("lvef", 55)
    time = ui.get("time", 0)

    adjustment = 0.0

    if lvef >= 70 and hr < 90 and age < 55:
        adjustment -= 0.05

    if hr >= 100:
        adjustment += 0.05

    if lvef < 50:
        adjustment += 0.08

    if time > 6:
        adjustment += 0.05

    adjusted_score = score + adjustment
    adjusted_score = max(0.0, min(1.0, adjusted_score))

    return adjusted_score


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        clinical_input = preprocess_clinical(data["clinical"])

        image_bytes = base64.b64decode(data["image_base64"])
        image_input = preprocess_image(image_bytes)

        clinical_score = clinical_model.predict_proba(clinical_input)[:, 1][0]
        image_score = image_model.predict_proba(image_input)[:, 1][0]

        raw_score = 0.6 * clinical_score + 0.4 * image_score
        final_score = apply_clinical_rules(data["clinical"], raw_score)


        if final_score < 0.25:
            risk = "Low"
        elif final_score < 0.45:
            risk = "Moderate"
        elif final_score < 0.65:
            risk = "High-Moderate"
        else:
            risk = "High"

        return jsonify({
            "cardiotoxicity_score": float(final_score),
            "risk": risk
        })

    except Exception as e:
        print("❌ ERROR:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000, debug=False)

