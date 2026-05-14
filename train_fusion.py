import pandas as pd
import numpy as np
import joblib
import cv2
import os

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -------------------------------
# 1. Load CNN (Image Feature Extractor)
# -------------------------------
resnet = ResNet50(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

# -------------------------------
# 2. Load fusion dataset
# -------------------------------
clinical_df = pd.read_csv("data/fusion_dataset.csv", low_memory=False)

# Fix European decimal separator
for col in clinical_df.columns:
    if clinical_df[col].dtype == object:
        clinical_df[col] = clinical_df[col].str.replace(",", ".", regex=False)

# -------------------------------
# 3. Define target and drop non-numeric columns
# -------------------------------
TARGET_COL = "label"          # ✅ correct target
DROP_COLS = ["image_path", TARGET_COL]

y = clinical_df[TARGET_COL].values

feature_cols = [c for c in clinical_df.columns if c not in DROP_COLS]

# Convert only feature columns to numeric
clinical_df[feature_cols] = clinical_df[feature_cols].apply(pd.to_numeric)

X_tabular = clinical_df[feature_cols].values

# -------------------------------
# 4. Extract image features using ResNet50
# -------------------------------
image_features = []

for img_path in clinical_df["image_path"]:
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img.astype(np.float32))
    img = np.expand_dims(img, axis=0)

    feat = resnet.predict(img, verbose=0)
    image_features.append(feat.flatten())

X_image = np.array(image_features)

# -------------------------------
# 5. Concatenate tabular + image features
# -------------------------------
X = np.concatenate([X_tabular, X_image], axis=1)

# -------------------------------
# 6. Scale features
# -------------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

# -------------------------------
# 7. Train / Test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------
# 8. Train XGBoost model
# -------------------------------
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

# -------------------------------
# 10. Model Evaluation
# -------------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

print("\n📊 Model Evaluation Results")
print("Accuracy :", acc)
print("ROC-AUC  :", auc)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

import matplotlib.pyplot as plt

fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Multimodal Cardiotoxicity Model")
plt.legend()
plt.show()


# -------------------------------
# 9. Save model and scaler
# -------------------------------
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/fusion_model.pkl")
joblib.dump(scaler, "models/fusion_scaler.pkl")

print(" Clinical + Functional + Image Fusion Model Trained Successfully")