import os
import numpy as np
import pandas as pd
import joblib

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

IMAGE_DIR = "data/IMAGES"
LABELS_PATH = "data/labels.csv"
IMG_SIZE = (224, 224)

labels_df = pd.read_csv(LABELS_PATH)

image_to_label = dict(zip(labels_df["image_name"], labels_df["label"]))

print(f"Total labels: {len(image_to_label)}")

cnn = ResNet50(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(224, 224, 3)
)

X_features = []
y_labels = []
used_images = 0

for img_name, label in image_to_label.items():
    img_path = os.path.join(IMAGE_DIR, img_name)

    if not os.path.exists(img_path):
        continue

    try:
        img = load_img(img_path, target_size=IMG_SIZE)
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)

        features = cnn.predict(img, verbose=0)

        X_features.append(features.flatten())
        y_labels.append(int(label))
        used_images += 1

    except Exception as e:
        print(f"Skipped {img_name}: {e}")

X = np.array(X_features)
y = np.array(y_labels)

print("Images used:", used_images)
print("Feature matrix shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/image_xgb_model.pkl")
joblib.dump(scaler, "models/image_xgb_scaler.pkl")

print("Image XGBoost model trained and saved successfully")
