import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import joblib

data = pd.read_csv(
    "data/BC_cardiotox_clinical_variables.csv",
    sep=";",
    engine="python",
    on_bad_lines="skip"
)

print(f"Loaded clinical data: {data.shape[0]} rows, {data.shape[1]} columns")
print("Columns:", list(data.columns))

target_column = "CTRCD"

X = data.drop(columns=[target_column])
y = data[target_column]

X = X.replace(',', '.', regex=True)

X = X.apply(pd.to_numeric, errors='coerce')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

joblib.dump(model, "clinical_model.pkl")
joblib.dump(imputer, "clinical_imputer.pkl")
joblib.dump(scaler, "clinical_scaler.pkl")

print("✅ Clinical model training complete")
