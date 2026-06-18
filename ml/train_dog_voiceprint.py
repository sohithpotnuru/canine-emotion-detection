"""
ml/train_dog_voiceprint.py — Train the dog identity ("voiceprint") encoder
(section 4.3/4.4, Phase 2).

REQUIRED DATA LAYOUT (see explanation at the bottom — this needs OTHER
dogs' recordings too, not just yours):

    data/dog_voiceprint/
        target_dog/       *.wav   (your dog's barks — as many contexts as possible)
        other_dogs/
            dog_a/        *.wav   (a different dog, several clips)
            dog_b/        *.wav
            ...             (5-10 other dogs recommended, section 4.1)

Usage:
    python ml/train_dog_voiceprint.py --data-dir data/dog_voiceprint --epochs 30
    python ml/train_dog_voiceprint.py --smoke-test
"""

import os
import glob
import argparse
import numpy as np
import tensorflow as tf
import soundfile as sf
import librosa

from models import build_dog_encoder, triplet_loss
from convert_tflite import convert_to_tflite
from features_tf import dog_mel_spectrogram

TARGET_SR = 22050


def load_wav_as_features(path: str) -> np.ndarray:
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
    return dog_mel_spectrogram(y, sample_rate=TARGET_SR)


def load_dataset(data_dir: str):
    """Returns a dict: {dog_id: [feature_array, feature_array, ...]}"""
    dogs = {}
    target_files = glob.glob(os.path.join(data_dir, "target_dog", "*.wav"))
    if target_files:
        dogs["target_dog"] = [load_wav_as_features(f) for f in target_files]

    for dog_dir in glob.glob(os.path.join(data_dir, "other_dogs", "*")):
        if os.path.isdir(dog_dir):
            files = glob.glob(os.path.join(dog_dir, "*.wav"))
            if files:
                dogs[os.path.basename(dog_dir)] = [load_wav_as_features(f) for f in files]

    if "target_dog" not in dogs or len(dogs) < 2:
        raise FileNotFoundError(
            f"Need target_dog/ AND at least one other_dogs/<name>/ folder with .wav files under {data_dir}. "
            "See the data layout described at the top of this script."
        )
    return dogs


def make_triplet_batch(dogs: dict, batch_size: int, rng: np.random.Generator):
    """Randomly samples (anchor, positive, negative) triplets: anchor and
    positive are two different clips of the SAME dog, negative is a clip
    from a DIFFERENT dog."""
    dog_ids = list(dogs.keys())
    anchors, positives, negatives = [], [], []
    for _ in range(batch_size):
        pos_dog = rng.choice(dog_ids)
        # need at least 2 clips of the same dog for anchor+positive
        clips = dogs[pos_dog]
        if len(clips) < 2:
            continue
        a_idx, p_idx = rng.choice(len(clips), size=2, replace=False)

        neg_dog_candidates = [d for d in dog_ids if d != pos_dog]
        if not neg_dog_candidates:
            continue
        neg_dog = rng.choice(neg_dog_candidates)
        n_idx = rng.integers(0, len(dogs[neg_dog]))

        anchors.append(clips[a_idx])
        positives.append(clips[p_idx])
        negatives.append(dogs[neg_dog][n_idx])

    return (np.array(anchors, dtype=np.float32),
            np.array(positives, dtype=np.float32),
            np.array(negatives, dtype=np.float32))


def make_synthetic_smoke_data():
    """Two fake 'dogs' with random features — verifies the triplet
    sampling + training step runs, NOT that anything is learned."""
    rng = np.random.default_rng(0)
    return {
        "target_dog": [rng.normal(0, 1, (96, 80, 1)).astype(np.float32) for _ in range(6)],
        "other_dog_a": [rng.normal(0.5, 1, (96, 80, 1)).astype(np.float32) for _ in range(6)],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/dog_voiceprint")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps-per-epoch", type=int, default=50)
    parser.add_argument("--output", default="models_trained/dog_voiceprint.tflite")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    rng = np.random.default_rng(42)

    if args.smoke_test:
        print("[SMOKE TEST] Using synthetic dogs — proves the triplet training loop runs,")
        print("[SMOKE TEST] does NOT produce a usable voiceprint model.\n")
        dogs = make_synthetic_smoke_data()
        epochs, steps = 2, 5
    else:
        print(f"Loading real dataset from {args.data_dir} ...")
        dogs = load_dataset(args.data_dir)
        epochs, steps = args.epochs, args.steps_per_epoch
        for name, clips in dogs.items():
            print(f"  {name}: {len(clips)} clips")

    encoder = build_dog_encoder()
    optimizer = tf.keras.optimizers.Adam(1e-3)
    loss_fn = triplet_loss(margin=0.3)

    for epoch in range(epochs):
        epoch_losses = []
        for _ in range(steps):
            a, p, n = make_triplet_batch(dogs, args.batch_size, rng)
            if len(a) == 0:
                continue
            with tf.GradientTape() as tape:
                a_emb = encoder(a, training=True)
                p_emb = encoder(p, training=True)
                n_emb = encoder(n, training=True)
                loss = loss_fn(a_emb, p_emb, n_emb)
            grads = tape.gradient(loss, encoder.trainable_variables)
            optimizer.apply_gradients(zip(grads, encoder.trainable_variables))
            epoch_losses.append(float(loss))
        print(f"Epoch {epoch + 1}/{epochs} — triplet loss: {np.mean(epoch_losses):.4f}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    keras_path = args.output.replace(".tflite", ".keras")
    encoder.save(keras_path)
    print(f"[OK] Saved full Keras model to {keras_path}")

    # Representative data for quantization: real (or smoke) clips flattened
    all_clips = np.array([c for clips in dogs.values() for c in clips], dtype=np.float32)
    convert_to_tflite(encoder, all_clips, args.output)


if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────────
# WHAT DATA YOU ACTUALLY NEED (section 4.1, Phase 2):
#   - target_dog/: as many varied clips of YOUR dog as possible.
#   - other_dogs/<name>/: recordings of 5-10 OTHER dogs (ask neighbors,
#     a dog park, a friend's dog) — the model can't learn "this is NOT
#     my dog" without hearing other dogs at all.
# Target from the doc: False Acceptance Rate < 5%, False Rejection Rate
# < 10% in a real multi-dog test (e.g. at a dog park).
