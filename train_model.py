import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Paths
BASE_DIR = os.path.dirname(__file__)
DATA_CSV = os.path.join(BASE_DIR, "..", "data", "isl_twohands_dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
MODEL_PATH = os.path.join(MODELS_DIR, "isl_twohands_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler_twohands.pkl")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder_twohands.pkl")

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = pd.read_csv(DATA_CSV)

    X = df.drop(columns=["label"]).values
    y = df["label"].values

    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    joblib.dump(clf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(label_encoder, ENCODER_PATH)

    print("Saved:")
    print(" Model:", MODEL_PATH)
    print(" Scaler:", SCALER_PATH)
    print(" Encoder:", ENCODER_PATH)

if __name__ == "__main__":
    main()