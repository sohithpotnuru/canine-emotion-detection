"""
ml/train_emotion_classifier.py — Train the multi-modal (audio + vitals)
emotion/health fusion classifier (section 4.4, Phase 3).

This one needs a trained dog_voiceprint encoder first (or bark detector
encoder — see note below) to turn audio into embeddings, PLUS synchronized
vitals readings for each labeled clip. This is the most data-intensive
piece of the whole system — plan for this to take the longest.

REQUIRED DATA LAYOUT:

    data/emotion/
        labels.csv   — columns: wav_path,heart_rate_mean,heart_rate_std,accel_mag,emotion
        clips/*.wav  — referenced by wav_path in labels.csv

`emotion` must be one of ml.models.EMOTION_CLASSES_V2:
    excited_playful, alert_guarding, lonely_anxious, communication,
    caution_intimidation, possible_pain, normal

Usage:
    python ml/train_emotion_classifier.py --data-dir data/emotion \\
        --encoder-path models_trained/dog_voiceprint.keras --epochs 40
    python ml/train_emotion_classifier.py --smoke-test
"""

import os
import csv
import argparse
import numpy as np
import tensorflow as tf
import soundfile as sf
import librosa

from models import build_emotion_classifier, EMOTION_CLASSES_V2
from convert_tflite import convert_to_tflite
from features_tf import dog_mel_spectrogram

TARGET_SR = 22050


def load_wav_as_mel(path: str) -> np.ndarray:
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
    return dog_mel_spectrogram(y, sample_rate=TARGET_SR)


def load_dataset(data_dir: str, encoder: tf.keras.Model):
    labels_path = os.path.join(data_dir, "labels.csv")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Expected {labels_path} — see the data layout at the top of this script.")

    audio_embeddings, vitals, emotion_idx = [], [], []
    label_to_idx = {name: i for i, name in enumerate(EMOTION_CLASSES_V2)}

    with open(labels_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["emotion"] not in label_to_idx:
                raise ValueError(
                    f"Unknown emotion label '{row['emotion']}' in labels.csv — "
                    f"must be one of {list(label_to_idx.keys())}"
                )
            wav_path = os.path.join(data_dir, "clips", row["wav_path"]) \
                if not os.path.isabs(row["wav_path"]) else row["wav_path"]
            mel = load_wav_as_mel(wav_path)
            emb = encoder(np.expand_dims(mel, 0), training=False).numpy()[0]

            audio_embeddings.append(emb)
            vitals.append([float(row["heart_rate_mean"]), float(row["heart_rate_std"]), float(row["accel_mag"])])
            emotion_idx.append(label_to_idx[row["emotion"]])

    return (np.array(audio_embeddings, dtype=np.float32),
            np.array(vitals, dtype=np.float32),
            np.array(emotion_idx, dtype=np.int32))


def make_synthetic_smoke_data(n=60, embedding_dim=128):
    rng = np.random.default_rng(0)
    audio = rng.normal(0, 1, (n, embedding_dim)).astype(np.float32)
    vitals = rng.normal(0, 1, (n, 3)).astype(np.float32)
    labels = rng.integers(0, len(EMOTION_CLASSES_V2), n).astype(np.int32)
    return audio, vitals, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/emotion")
    parser.add_argument("--encoder-path", default="models_trained/dog_voiceprint.keras",
                         help="Path to a trained dog_voiceprint encoder (produces audio embeddings)")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", default="models_trained/emotion_classifier.tflite")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        print("[SMOKE TEST] Using synthetic embeddings/vitals — proves the training loop runs,")
        print("[SMOKE TEST] does NOT produce a usable emotion classifier.\n")
        audio, vitals, labels = make_synthetic_smoke_data()
        epochs = 2
    else:
        if not os.path.exists(args.encoder_path):
            raise FileNotFoundError(
                f"No trained encoder at {args.encoder_path}. Train ml/train_dog_voiceprint.py "
                "first — this script needs it to turn raw audio into embeddings."
            )
        encoder = tf.keras.models.load_model(args.encoder_path, compile=False)
        print(f"Loading real dataset from {args.data_dir} ...")
        audio, vitals, labels = load_dataset(args.data_dir, encoder)
        epochs = args.epochs
        print(f"Loaded {len(audio)} labeled clips across {len(set(labels))} emotion classes")

    split = int(len(audio) * 0.85)
    idx = np.random.default_rng(42).permutation(len(audio))
    audio, vitals, labels = audio[idx], vitals[idx], labels[idx]

    model = build_emotion_classifier(num_emotion_classes=len(EMOTION_CLASSES_V2),
                                      embedding_dim=audio.shape[1])
    model.summary()

    callbacks = [tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True, monitor="val_loss")]

    model.fit(
        [audio[:split], vitals[:split]], labels[:split],
        validation_data=([audio[split:], vitals[split:]], labels[split:]) if split < len(audio) else None,
        epochs=epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    keras_path = args.output.replace(".tflite", ".keras")
    model.save(keras_path)
    print(f"[OK] Saved full Keras model to {keras_path}")

    convert_to_tflite(model, [audio[:split], vitals[:split]], args.output)


if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────────
# WHAT DATA YOU ACTUALLY NEED (section 4.1, Phase 3) — this is the hard
# part:
#   - Synchronized audio + vitals for each labeled clip. In practice:
#     record your dog during known emotional contexts (play sessions,
#     alone time, a vet visit, mealtime, doorbell/stranger, rest) WHILE
#     the collar is logging heart rate/temp/activity, then label each
#     clip afterward with what was actually happening.
#   - Target from the doc: accuracy > 75% on the 7-class task, validated
#     against your own (or a vet's) judgment of what the dog was
#     actually feeling — not just the model agreeing with itself.
# This is genuinely the multi-week-to-multi-month part of the whole
# project; there's no shortcut around collecting real labeled data here.
