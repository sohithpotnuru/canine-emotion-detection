"""
ml/features_tf.py — TensorFlow-native audio front-end.

Replaces librosa-based feature extraction with tf.signal ops, which is
what lets these features run inside a TFLite graph on a Raspberry Pi
instead of needing a full Python + librosa install on the edge device.

This does NOT replace feature_extraction.py (the 420-dim XGBoost feature
vector) — that stays as-is because the currently-deployed app.py/XGBoost
model was trained against it, and changing it would silently break the
model already in use (exactly the kind of train/inference mismatch bug
that was already found and fixed in bridge_server.py). This module is
for the NEW Keras/TFLite models described in the production design doc,
which are trained from scratch against mel-spectrograms, not the old
420-dim vector.

Two configurations, matching the design doc exactly:
  - DOG:   22,050 Hz, band-pass 200Hz-8kHz (canine vocalization range)
  - HUMAN: 16,000 Hz, full band (speech range)
"""

import numpy as np
import tensorflow as tf

# ── Shared spectrogram parameters (per section 4.2) ─────────────
FRAME_LENGTH = 2048   # samples
HOP_LENGTH = 512      # samples (75% overlap)
NUM_MEL_BINS = 80
TARGET_TIME_FRAMES = 96  # matches the model input shape (96, 80, 1) in section 4.3


def _mel_spectrogram(waveform: np.ndarray, sample_rate: int,
                      lower_hz: float, upper_hz: float) -> np.ndarray:
    """
    waveform: 1-D float32 array, already normalized to roughly [-1, 1].
    Returns a (TARGET_TIME_FRAMES, NUM_MEL_BINS) log-mel spectrogram,
    padded or center-cropped to a fixed time dimension so every output
    is the same shape regardless of input clip length (models need a
    fixed input shape).
    """
    waveform = tf.convert_to_tensor(waveform, dtype=tf.float32)

    stft = tf.signal.stft(
        waveform,
        frame_length=FRAME_LENGTH,
        frame_step=HOP_LENGTH,
        fft_length=FRAME_LENGTH,
    )
    spectrogram = tf.abs(stft)

    num_spectrogram_bins = spectrogram.shape[-1]
    mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=NUM_MEL_BINS,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=sample_rate,
        lower_edge_hertz=lower_hz,
        upper_edge_hertz=upper_hz,
    )
    mel_spectrogram = tf.tensordot(spectrogram, mel_weight_matrix, 1)
    mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(mel_weight_matrix.shape[-1:]))

    log_mel = tf.math.log(mel_spectrogram + 1e-6)

    # z-score normalize (per section 4.2: "Log-Mel + z-score")
    mean = tf.reduce_mean(log_mel)
    std = tf.math.reduce_std(log_mel) + 1e-6
    log_mel = (log_mel - mean) / std

    log_mel = log_mel.numpy()

    # Fix the time dimension to TARGET_TIME_FRAMES via pad/crop so every
    # output tensor is the same shape for the model's fixed input layer.
    t = log_mel.shape[0]
    if t < TARGET_TIME_FRAMES:
        pad = TARGET_TIME_FRAMES - t
        log_mel = np.pad(log_mel, ((0, pad), (0, 0)), mode="edge")
    elif t > TARGET_TIME_FRAMES:
        start = (t - TARGET_TIME_FRAMES) // 2
        log_mel = log_mel[start:start + TARGET_TIME_FRAMES]

    return log_mel.astype(np.float32)


def dog_mel_spectrogram(waveform: np.ndarray, sample_rate: int = 22050) -> np.ndarray:
    """
    Dog vocalization band-pass: 200 Hz - 8 kHz, at 22,050 Hz (section 3.2/4.2).
    Returns shape (96, 80, 1) ready to feed into build_bark_detector /
    build_dog_encoder.
    """
    if sample_rate != 22050:
        raise ValueError(
            f"Dog audio must be resampled to 22050 Hz before this call (got {sample_rate}). "
            "Resample on the collar/bridge side, not here — resampling inside the "
            "hot feature-extraction path adds latency you don't need on every frame."
        )
    mel = _mel_spectrogram(waveform, sample_rate, lower_hz=200.0, upper_hz=8000.0)
    return mel[..., np.newaxis]  # (96, 80, 1)


def human_mel_spectrogram(waveform: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """
    Human speech, full band, at 16,000 Hz (section 3.2/5.1).
    Returns shape (96, 80, 1).
    """
    if sample_rate != 16000:
        raise ValueError(f"Human audio must be resampled to 16000 Hz before this call (got {sample_rate}).")
    mel = _mel_spectrogram(waveform, sample_rate, lower_hz=0.0, upper_hz=8000.0)
    return mel[..., np.newaxis]  # (96, 80, 1)


if __name__ == "__main__":
    # Self-test: confirm both configs produce exactly the shape the
    # models in ml/models.py expect, before anything gets trained.
    rng = np.random.default_rng(0)

    dog_wave = rng.normal(0, 0.05, 22050).astype(np.float32)  # 1s @ 22.05kHz
    dog_feat = dog_mel_spectrogram(dog_wave)
    assert dog_feat.shape == (96, 80, 1), f"dog shape mismatch: {dog_feat.shape}"
    print(f"[OK] dog_mel_spectrogram shape: {dog_feat.shape}")

    human_wave = rng.normal(0, 0.05, 16000).astype(np.float32)  # 1s @ 16kHz
    human_feat = human_mel_spectrogram(human_wave)
    assert human_feat.shape == (96, 80, 1), f"human shape mismatch: {human_feat.shape}"
    print(f"[OK] human_mel_spectrogram shape: {human_feat.shape}")

    print("All feature extraction self-tests passed.")
