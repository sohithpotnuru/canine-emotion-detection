"""
ml/convert_tflite.py — Post-training INT8 quantization (section 6.3).

Usage:
    from ml.convert_tflite import convert_to_tflite
    convert_to_tflite(model, representative_data, "bark_detector.tflite")

`representative_data` should be a numpy array (or list of numpy arrays
for multi-input models) of a few hundred REAL examples from your
training set — not random noise. INT8 quantization calibrates its
scale/zero-point from this data, so if it doesn't look like your real
audio, the quantized model will be inaccurate even though conversion
succeeds without any error.
"""

import numpy as np
import tensorflow as tf


def convert_to_tflite(model: tf.keras.Model, representative_samples,
                       output_path: str, num_calibration_steps: int = 200):
    """
    representative_samples: either a single numpy array (single-input
    model) or a list of numpy arrays (multi-input model, one array per
    input, all same length along axis 0).
    """
    is_multi_input = isinstance(representative_samples, (list, tuple))
    n = len(representative_samples[0]) if is_multi_input else len(representative_samples)
    n_steps = min(num_calibration_steps, n)

    def representative_dataset():
        for i in range(n_steps):
            if is_multi_input:
                yield [np.expand_dims(arr[i], 0).astype(np.float32) for arr in representative_samples]
            else:
                yield [np.expand_dims(representative_samples[i], 0).astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    print(f"[OK] Wrote {output_path} ({len(tflite_model) / 1024:.1f} KB, INT8 quantized)")
    return output_path
