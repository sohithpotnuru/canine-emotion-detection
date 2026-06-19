"""
ml/inference.py — On-device inference using the trained TFLite models.

Used by bridge_server.py. If the trained model files don't exist yet
(you haven't run the ml/train_*.py scripts against real data), every
method here returns None so the CALLER can fall back to the existing,
already-fixed cloud XGBoost pipeline in app.py — nothing breaks while
these are still untrained. Once you've trained and dropped the .tflite
files in models_trained/, this starts actually being used automatically,
no other code changes needed.
"""

import os
import numpy as np
import tensorflow as tf

from features_tf import dog_mel_spectrogram
from models import EMOTION_CLASSES_V2

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models_trained")
BARK_DETECTOR_PATH = os.path.join(MODELS_DIR, "bark_detector.tflite")
DOG_VOICEPRINT_PATH = os.path.join(MODELS_DIR, "dog_voiceprint.tflite")
EMOTION_CLASSIFIER_PATH = os.path.join(MODELS_DIR, "emotion_classifier.tflite")
TARGET_DOG_PROFILE_PATH = os.path.join(MODELS_DIR, "target_dog_profile.npy")

BARK_THRESHOLD = 0.5
# Voiceprint similarity threshold — start here per the doc's FAR/FRR
# targets, then tune against your own real target-vs-other-dog data.
VOICEPRINT_SIMILARITY_THRESHOLD = 0.6


class _TFLiteModel:
    """
    Thin wrapper around tf.lite.Interpreter that handles INT8
    quantize/dequantize correctly — this project's convert_tflite.py
    always exports full-INT8 models (inference_input_type=int8,
    output_type=int8), so raw float arrays must be quantized using each
    tensor's own (scale, zero_point) before feeding the interpreter, and
    outputs must be dequantized back to float the same way. Skipping
    this step produces a model that "runs" without error but returns
    meaningless numbers — a silent-failure bug easy to miss.
    """
    def __init__(self, path: str):
        self.interpreter = tf.lite.Interpreter(model_path=path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, *inputs: np.ndarray):
        for detail, arr in zip(self.input_details, inputs):
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim == len(detail["shape"]) - 1:
                arr = np.expand_dims(arr, 0)  # add batch dim

            scale, zero_point = detail["quantization"]
            if scale and scale != 0:
                quantized = np.round(arr / scale + zero_point).astype(detail["dtype"])
            else:
                quantized = arr.astype(detail["dtype"])
            self.interpreter.set_tensor(detail["index"], quantized)

        self.interpreter.invoke()

        outputs = []
        for detail in self.output_details:
            raw = self.interpreter.get_tensor(detail["index"])
            scale, zero_point = detail["quantization"]
            if scale and scale != 0:
                raw = (raw.astype(np.float32) - zero_point) * scale
            outputs.append(raw[0])  # drop batch dim
        return outputs if len(outputs) > 1 else outputs[0]


class OnDeviceEmotionPipeline:
    """
    Lazily loads whichever trained models actually exist on disk.
    Each stage degrades gracefully: if a required model is missing,
    the corresponding method returns None rather than raising, so
    bridge_server.py can cleanly fall back to the cloud XGBoost path.
    """
    def __init__(self):
        self.bark_detector = self._try_load(BARK_DETECTOR_PATH)
        self.dog_encoder = self._try_load(DOG_VOICEPRINT_PATH)
        self.emotion_classifier = self._try_load(EMOTION_CLASSIFIER_PATH)
        self.target_profile = None
        if os.path.exists(TARGET_DOG_PROFILE_PATH):
            self.target_profile = np.load(TARGET_DOG_PROFILE_PATH)

    @staticmethod
    def _try_load(path: str):
        if os.path.exists(path):
            print(f"[ON-DEVICE] Loaded {os.path.basename(path)}")
            return _TFLiteModel(path)
        return None

    def is_fully_available(self) -> bool:
        return self.bark_detector is not None and self.dog_encoder is not None \
            and self.emotion_classifier is not None and self.target_profile is not None

    def is_bark(self, waveform: np.ndarray, sample_rate: int = 22050):
        if self.bark_detector is None:
            return None
        mel = dog_mel_spectrogram(waveform, sample_rate=sample_rate)
        prob = float(self.bark_detector.predict(mel)[0])
        return prob >= BARK_THRESHOLD

    def is_target_dog(self, waveform: np.ndarray, sample_rate: int = 22050):
        if self.dog_encoder is None or self.target_profile is None:
            return None
        embedding = self.embed(waveform, sample_rate)
        if embedding is None:
            return None
        similarity = float(np.dot(embedding, self.target_profile) /
                            (np.linalg.norm(embedding) * np.linalg.norm(self.target_profile) + 1e-8))
        return similarity >= VOICEPRINT_SIMILARITY_THRESHOLD

    def embed(self, waveform: np.ndarray, sample_rate: int = 22050):
        if self.dog_encoder is None:
            return None
        mel = dog_mel_spectrogram(waveform, sample_rate=sample_rate)
        return np.asarray(self.dog_encoder.predict(mel))

    def classify_emotion(self, waveform: np.ndarray, vital_features: list, sample_rate: int = 22050):
        """
        Returns (emotion_label, confidence) or (None, None) if the
        on-device pipeline isn't ready yet (missing any required model).
        """
        if self.emotion_classifier is None:
            return None, None
        embedding = self.embed(waveform, sample_rate)
        if embedding is None:
            return None, None
        vitals = np.asarray(vital_features, dtype=np.float32)
        probs = np.asarray(self.emotion_classifier.predict(embedding, vitals))
        idx = int(np.argmax(probs))
        return EMOTION_CLASSES_V2[idx], float(probs[idx])


if __name__ == "__main__":
    # Confirms the wrapper handles "nothing trained yet" cleanly —
    # this is the actual state right after a fresh checkout, before any
    # ml/train_*.py script has been run.
    pipeline = OnDeviceEmotionPipeline()
    print("Bark detector loaded:      ", pipeline.bark_detector is not None)
    print("Dog voiceprint loaded:     ", pipeline.dog_encoder is not None)
    print("Emotion classifier loaded: ", pipeline.emotion_classifier is not None)
    print("Fully available:          ", pipeline.is_fully_available())

    dummy_wave = np.random.randn(22050).astype(np.float32) * 0.05
    print("classify_emotion() on empty pipeline ->", pipeline.classify_emotion(dummy_wave, [80, 5, 1.0]))
