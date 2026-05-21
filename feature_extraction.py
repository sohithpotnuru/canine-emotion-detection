"""
train_multimodal_fusion.py, app.py, and streamlit_app.py must ALL
import extract_flat_audio_features (or the file-path wrapper) from this module.
Do NOT re-implement this logic anywhere else. If training and inference compute
features differently, the model receives inputs it was never trained on and
every prediction becomes meaningless, even though nothing crashes.

This is the exact aggregation used to train dog_audio_xgb.joblib:
for each of 6 frame-level matrices (mfcc, delta, delta2, chroma, contrast, rms),
take the mean, std, and max across time. That is (40+40+40+12+7+1) * 3 = 420.
"""

import numpy as np
import librosa

SAMPLE_RATE = 22050
N_MFCC = 40
EXPECTED_FEATURE_LEN = 420


def extract_flat_audio_features(y, sr=SAMPLE_RATE):
    """
    Extract the 420-length time-aggregated feature vector from a raw waveform.

    Parameters
    ----------
    y : np.ndarray, mono float waveform
    sr : int, sample rate of y

    Returns
    -------
    np.ndarray of shape (420,), dtype float32, or None if extraction failed.
    """
    try:
        if y is None or len(y) == 0:
            return None

        y = np.asarray(y, dtype=np.float32)

        if len(y) < 4096:
            y = np.pad(y, (0, 4096 - len(y)), mode='constant')

        mfccs_matrix = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        if mfccs_matrix.shape[1] < 9:
            mfccs_matrix = np.pad(
                mfccs_matrix, ((0, 0), (0, 9 - mfccs_matrix.shape[1])), mode='edge'
            )

        delta_matrix = librosa.feature.delta(mfccs_matrix)
        delta2_matrix = librosa.feature.delta(mfccs_matrix, order=2)
        chroma_matrix = librosa.feature.chroma_stft(y=y, sr=sr)
        contrast_matrix = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)
        rms_matrix = librosa.feature.rms(y=y)

        feature_blocks = []
        for matrix in [mfccs_matrix, delta_matrix, delta2_matrix,
                        chroma_matrix, contrast_matrix, rms_matrix]:
            feature_blocks.append(np.mean(matrix, axis=1))
            feature_blocks.append(np.std(matrix, axis=1))
            feature_blocks.append(np.max(matrix, axis=1))

        audio_vector = np.concatenate(feature_blocks).astype(np.float32)

        if audio_vector.shape[0] != EXPECTED_FEATURE_LEN:
            return None

        return audio_vector

    except Exception:
        return None


def extract_flat_audio_features_from_file(file_path, sr=SAMPLE_RATE):
    """
    Load a full audio file (no duration cap — training used full clips) and
    return its 420-length feature vector, or None on any failure.
    """
    try:
        y, loaded_sr = librosa.load(file_path, sr=sr)
    except Exception:
        return None
    return extract_flat_audio_features(y, sr=loaded_sr)