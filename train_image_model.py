import os
import cv2
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from image_feature_extractor import extract_image_features_from_bytes

X, y = [], []

for label in [0, 1]:
    folder = f"data/images/class{label}"
    for file in os.listdir(folder):
        with open(os.path.join(folder, file), "rb") as f:
            feats = extract_image_features_from_bytes(f.read())
            X.append(feats.flatten())
            y.append(label)

X = np.array(X)
y = np.array(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    eval_metric="logloss"
)

model.fit(X_scaled, y)

joblib.dump(model, "models/image_xgb_model.pkl")
joblib.dump(scaler, "models/image_xgb_scaler.pkl")

print(" Image model retrained successfully")
