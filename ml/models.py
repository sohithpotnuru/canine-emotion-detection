"""
ml/models.py — Keras model architectures from the production design doc
(sections 4.3, 4.4). These are architecture DEFINITIONS ONLY — building
one of these gives you an untrained model with random weights. Nothing
here can classify anything correctly until you run the matching
ml/train_*.py script against real, labeled recordings.

One real bug fixed from the design doc's own pseudocode: its bark
detector used a hardcoded Reshape((-1, 64*64//4)) before the GRU layer.
For a (96, 80, 1) input after two (2,2) max-pools, the actual tensor is
(24, 20, 64) — 20 freq bins x 64 channels = 1280 features per timestep,
not the doc's hardcoded 1024. A hardcoded magic number here silently
breaks the moment the input shape changes, so this version computes
the reshape from the actual tensor shape instead.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


def build_bark_detector(input_shape=(96, 80, 1)):
    """Lightweight CRNN for bark vs non-bark classification (section 4.3)."""
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)

    # Reshape (time, freq, channels) -> (time, freq*channels) for the GRU.
    # Computed from the actual tensor shape (fixes the doc's hardcoded,
    # incorrect 64*64//4 magic number — see module docstring).
    _, t, f, c = x.shape
    x = layers.Reshape((t, f * c))(x)

    x = layers.GRU(64, return_sequences=False, unroll=True)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs, name="bark_detector")
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Recall()])
    return model


def build_dog_encoder(input_shape=(96, 80, 1), embedding_dim=128):
    """CNN encoder for dog bark embeddings (Siamese architecture, section 4.3)."""
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(embedding_dim, activation=None, name='embedding')(x)

    # L2-normalize embeddings for cosine similarity
    outputs = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=-1))(outputs)

    return Model(inputs, outputs, name="dog_voiceprint_encoder")


def triplet_loss(margin=0.3):
    def loss(anchor_emb, positive_emb, negative_emb):
        pos_dist = tf.reduce_sum(tf.square(anchor_emb - positive_emb), axis=-1)
        neg_dist = tf.reduce_sum(tf.square(anchor_emb - negative_emb), axis=-1)
        return tf.reduce_mean(tf.maximum(pos_dist - neg_dist + margin, 0.0))
    return loss


# 7-class schema from section 3.5 — an upgrade from the existing
# prototype's 5-class (aggressive, fearful, happy, neutral, pain).
EMOTION_CLASSES_V2 = [
    "excited_playful", "alert_guarding", "lonely_anxious",
    "communication", "caution_intimidation", "possible_pain", "normal",
]


def build_emotion_classifier(num_emotion_classes=7, embedding_dim=128):
    """Multi-modal fusion: audio embedding + vitals -> emotion state (section 4.4)."""
    audio_input = layers.Input(shape=(embedding_dim,), name='audio_embedding')
    audio_branch = layers.Dense(128, activation='relu')(audio_input)
    audio_branch = layers.BatchNormalization()(audio_branch)
    audio_branch = layers.Dropout(0.3)(audio_branch)

    # [hr_mean, hr_std, accel_mag] — matches bridge_server.py's
    # build_vital_features() exactly, already fixed to this contract.
    vitals_input = layers.Input(shape=(3,), name='vitals')
    vitals_branch = layers.Dense(32, activation='relu')(vitals_input)
    vitals_branch = layers.BatchNormalization()(vitals_branch)

    merged = layers.Concatenate()([audio_branch, vitals_branch])
    x = layers.Dense(128, activation='relu')(merged)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(num_emotion_classes, activation='softmax', name='emotion')(x)

    model = Model(inputs=[audio_input, vitals_input], outputs=outputs, name="emotion_fusion_classifier")
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


if __name__ == "__main__":
    import numpy as np

    print("Building and smoke-testing all architectures with dummy data...")
    print("(This proves the architectures are wired correctly. It does NOT")
    print(" mean anything is trained — see ml/train_*.py for that.)\n")

    bark_model = build_bark_detector()
    bark_out = bark_model(np.random.randn(2, 96, 80, 1).astype(np.float32))
    assert bark_out.shape == (2, 1)
    print(f"[OK] bark_detector: {bark_model.count_params():,} params, output {bark_out.shape}")

    enc = build_dog_encoder()
    emb = enc(np.random.randn(2, 96, 80, 1).astype(np.float32))
    assert emb.shape == (2, 128)
    norms = tf.reduce_sum(emb ** 2, axis=-1).numpy()
    assert np.allclose(norms, 1.0, atol=1e-4), "embeddings should be L2-normalized"
    print(f"[OK] dog_voiceprint_encoder: {enc.count_params():,} params, output {emb.shape}, L2-normalized")

    clf = build_emotion_classifier()
    probs = clf([np.random.randn(2, 128).astype(np.float32),
                 np.random.randn(2, 3).astype(np.float32)])
    assert probs.shape == (2, 7)
    assert np.allclose(np.sum(probs.numpy(), axis=-1), 1.0, atol=1e-4), "softmax should sum to 1"
    print(f"[OK] emotion_fusion_classifier: {clf.count_params():,} params, output {probs.shape}, valid softmax")

    print("\nAll architecture smoke-tests passed.")
