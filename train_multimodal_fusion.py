import os
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from feature_extraction import extract_flat_audio_features_from_file, SAMPLE_RATE

DATASET_ROOT = r"C:\Users\NIDHI\OneDrive\Documents\dog_emotion_classification\data\raw\dataset"
METADATA_PATH = os.path.join(DATASET_ROOT, "master_metadata.csv")


def main():
    print(" Loading Registry Master Metadata Source...")
    if not os.path.exists(METADATA_PATH):
        print(f"❌ ERROR: Cannot find master_metadata.csv at: {METADATA_PATH}")
        return

    df = pd.read_csv(METADATA_PATH)
    X_audio, X_vitals, y = [], [], []

    print(f" Extracting & Segregating Multi-Modal Features across {len(df)} entries...")
    for idx, row in df.iterrows():
        audio_abs_path = os.path.join(DATASET_ROOT, row['audio_filepath'])
        audio_features = extract_flat_audio_features_from_file(audio_abs_path, sr=SAMPLE_RATE)
        if audio_features is None:
            continue

        hr_mean = row['hr_mean_bpm']
        hr_std = row['hr_std']
        accel_mag = row['accel_mag_mean']

        current_label = row['label']
        breed_type = str(row['breed']).strip()

        # NOTE: previously this filter compared breed_type to "unknown"/"nan" while
        # current_label was checked against "neutral_resting" — but every breed-folder
        # row has current_label == "neutral_resting" AND a real breed name, so the old
        # `continue` fired on every single breed row and silently dropped all of them.
        # Breed rows are real, useful "resting" examples — keep them and relabel instead.
        if current_label == "neutral_resting":
            current_label = "neutral"

        hr_mean += np.random.normal(0, 4.0)
        hr_std += np.random.normal(0, 0.8)
        accel_mag += np.random.normal(0, 0.15)

        vital_vector = np.array([hr_mean, hr_std, accel_mag], dtype=np.float32)

        X_audio.append(audio_features)
        X_vitals.append(vital_vector)
        y.append(current_label)

    X_audio = np.array(X_audio, dtype=np.float32)
    X_vitals = np.array(X_vitals, dtype=np.float32)
    y = np.array(y)

    print(f"✅ Extraction Complete.")
    print(f"   ↳ Audio Matrix Shape  : {X_audio.shape} (420 features via mean/std/max)")
    print(f"   ↳ Vitals Matrix Shape : {X_vitals.shape} (3 features)")
    print(f"   ↳ Label distribution: {pd.Series(y).value_counts().to_dict()}")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    joblib.dump(le, 'label_encoder.joblib')
    print(f" Saved label_encoder.joblib classes: {list(le.classes_)}")

    indices = np.arange(len(y_encoded))
    idx_train, idx_test, y_train, y_test = train_test_split(
        indices, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    X_audio_train, X_audio_test = X_audio[idx_train], X_audio[idx_test]
    X_vitals_train, X_vitals_test = X_vitals[idx_train], X_vitals[idx_test]

    print(" Training specialized Audio XGBoost subsystem...")
    audio_model = XGBClassifier(
        n_estimators=120,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='multi:softprob'
    )
    audio_model.fit(X_audio_train, y_train)

    print(" Training specialized Biometric Vitals XGBoost subsystem...")
    vitals_model = XGBClassifier(
        n_estimators=80,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
        objective='multi:softprob'
    )
    vitals_model.fit(X_vitals_train, y_train)

    print(" Executing Late Fusion probability blending evaluation...")
    audio_probs = audio_model.predict_proba(X_audio_test)
    vitals_probs = vitals_model.predict_proba(X_vitals_test)

    blended_probs = (0.65 * audio_probs) + (0.35 * vitals_probs)
    final_preds = np.argmax(blended_probs, axis=1)

    acc = accuracy_score(y_test, final_preds)
    print(f"\n Late Fusion Ensemble Validation Accuracy: {acc:.2%}")
    print("\n Production Classification Summary:")
    print(classification_report(y_test, final_preds, target_names=le.classes_))

    joblib.dump(audio_model, 'dog_audio_xgb.joblib')
    joblib.dump(vitals_model, 'dog_vitals_xgb.joblib')
    print(" SUCCESS: Separate commercial-grade binaries saved cleanly!")


if __name__ == "__main__":
    main()