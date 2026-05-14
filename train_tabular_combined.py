import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

DATA_PATH = "data/BC_cardiotox_clinical_and_functional_variables.csv"
MODEL_DIR = "models"

df = pd.read_csv(DATA_PATH, sep=";")

for col in df.columns:
    df[col] = df[col].astype(str).str.replace(",", ".").astype(float)

X = df.drop(columns=["CTRCD"])
y = df["CTRCD"]

imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = XGBClassifier(
    n_estimators=500,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    scale_pos_weight=8.7,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

joblib.dump(model, f"{MODEL_DIR}/tabular_combined_model.pkl")
joblib.dump(imputer, f"{MODEL_DIR}/tabular_combined_imputer.pkl")
joblib.dump(scaler, f"{MODEL_DIR}/tabular_combined_scaler.pkl")

print("✅ Combined clinical + functional model trained successfully")
