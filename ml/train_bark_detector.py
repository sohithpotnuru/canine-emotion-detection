"""
ml/train_bark_detector.py — Train the Bark vs Non-Bark classifier (section 4.3, Phase 1).

REQUIRED DATA LAYOUT (you must collect this — see explanation at the
bottom of this file for exactly what/how much):

    data/bark_detector/
        bark/           *.wav  (clips of the target dog barking/whining/growling)
        non_bark/       *.wav  (silence, human speech, other noise, other animals,
                                 UrbanSound8K / FSD50K clips per section 4.1)

Each .wav should be mono, any sample rate (resampled to 22050Hz here),
roughly 1-3 seconds long.

Usage:
    python ml/train_bark_detector.py --data-dir data/bark_detector --epochs 30
    python ml/train_bark_detector.py --smoke-test   # verify the code runs, no real data needed
"""

import os
import glob
import argparse
import numpy as np
import tensorflow as tf
import soundfile as sf
import librosa

from models import build_bark_detector
from convert_tflite import convert_to_tflite
from features_tf import dog_mel_spectrogram

TARGET_SR = 22050


def load_wav_as_features(path: str) -> np.ndarray:
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)  # downmix to mono
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
    return dog_mel_spectrogram(y, sample_rate=TARGET_SR)


def load_dataset(data_dir: str):
    bark_files = glob.glob(os.path.join(data_dir, "bark", "*.wav"))
    non_bark_files = glob.glob(os.path.join(data_dir, "non_bark", "*.wav"))

    if not bark_files or not non_bark_files:
        raise FileNotFoundError(
            f"Expected .wav files under {data_dir}/bark/ and {data_dir}/non_bark/ "
            f"— found {len(bark_files)} bark and {len(non_bark_files)} non_bark files. "
            "See the data layout described at the top of this script."
        )

    X, y = [], []
    for f in bark_files:
        X.append(load_wav_as_features(f))
        y.append(1)
    for f in non_bark_files:
        X.append(load_wav_as_features(f))
        y.append(0)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def make_synthetic_smoke_data(n=40):
    """Random data with the RIGHT SHAPE, just to prove the training loop
    runs end-to-end without crashing. This does NOT produce a usable
    model — it's a code-correctness check only."""
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (n, 96, 80, 1)).astype(np.float32)
    y = rng.integers(0, 2, n).astype(np.float32)
    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/bark_detector")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", default="models_trained/bark_detector.tflite")
    parser.add_argument("--smoke-test", action="store_true",
                         help="Run with tiny synthetic data to verify the script works, no real dataset needed.")
    args = parser.parse_args()

    if args.smoke_test:
        print("[SMOKE TEST] Using synthetic data — this proves the training loop runs, ")
        print("[SMOKE TEST] it does NOT produce a usable bark detector.\n")
        X, y = make_synthetic_smoke_data()
        epochs = 2
    else:
        print(f"Loading real dataset from {args.data_dir} ...")
        X, y = load_dataset(args.data_dir)
        epochs = args.epochs
        print(f"Loaded {len(X)} clips ({int(y.sum())} bark, {int((1 - y).sum())} non-bark)")

    split = int(len(X) * 0.85)
    idx = np.random.default_rng(42).permutation(len(X))
    X, y = X[idx], y[idx]
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = build_bark_detector()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss"),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val) if len(X_val) > 0 else None,
        epochs=epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    keras_path = args.output.replace(".tflite", ".keras")
    model.save(keras_path)
    print(f"[OK] Saved full Keras model to {keras_path}")

    convert_to_tflite(model, X_train, args.output)


if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────────
# WHAT DATA YOU ACTUALLY NEED (section 4.1):
#   - bark/: as much real audio of YOUR dog as you can get — barking,
#     whining, growling, in different contexts (play, distress, alone,
#     guarding). More context diversity matters more than raw volume;
#     50-100 clips covering varied situations beats 500 clips of the
#     same context.
#   - non_bark/: human speech, silence, household noise (TV, fan,
#     appliances), and ideally a public dataset for volume — download
#     UrbanSound8K (https://urbansounddataset.weebly.com/urbansound8k.html)
#     or FSD50K (https://labs.freesound.org/datasets/) and drop a subset
#     of clips in here as negative examples.
# Target from the doc: recall > 90%, precision > 70% on a held-out set
# — meaning it's fine to occasionally misfire on non-bark sounds, but it
# should almost never MISS a real bark.
