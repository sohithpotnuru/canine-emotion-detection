import streamlit as st
import numpy as np
import requests
import os

from feature_extraction import extract_flat_audio_features_from_file, SAMPLE_RATE

st.set_page_config(page_title="Dog Emotion Detection", page_icon="🐾", layout="centered")

st.title("🐾 Dog Emotion Detection")
st.write("Upload a dog audio clip and input biometrics to detect emotional states instantly.")

# --- SIDEBAR: Multi-Modal Vital Sign Inputs ---
st.sidebar.header("🐕 Live Biometric Sensors")
st.sidebar.write("Input the corresponding IoT sensor packets:")

hr_mean_bpm = st.sidebar.slider("Heart Rate Mean (BPM)", min_value=50.0, max_value=220.0, value=100.0, step=0.1)
hr_std = st.sidebar.slider("Heart Rate Volatility (STD)", min_value=0.0, max_value=30.0, value=5.2, step=0.1)
accel_mag_mean = st.sidebar.slider("Accelerometer Magnitude (Movement)", min_value=0.0, max_value=5.0, value=1.1, step=0.1)

# The live backend endpoint can be overridden via environment variables.
AI_BACKEND_URL = os.getenv("AI_BACKEND_URL", "http://127.0.0.1:5000/predict")


def get_audio_features(uploaded_file):
    """
    Saves the uploaded clip to a temp file and runs it through the EXACT
    same extraction used at training time (feature_extraction.py). This
    must never be reimplemented here — see feature_extraction.py docstring.
    """
    temp_filename = "temp_stream_audio.wav"
    try:
        with open(temp_filename, "wb") as f:
            f.write(uploaded_file.getbuffer())

        features = extract_flat_audio_features_from_file(temp_filename, sr=SAMPLE_RATE)
        return features.tolist() if features is not None else None
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


# --- UI Layout ---
uploaded = st.file_uploader("Choose an audio file...", type=["wav", "mp3"])

if uploaded:
    st.audio(uploaded)

    if st.button("Detect Emotion", type="primary"):
        with st.spinner("Analyzing multi-modal fusion matrices..."):

            audio_data = get_audio_features(uploaded)

            if audio_data is None:
                st.error("Could not read audio signatures. Try a longer or cleaner clip.")
                st.stop()

            # Pack the 3 vitals features matching shape (1, 3)
            vital_data = [
                float(hr_mean_bpm),
                float(hr_std),
                float(accel_mag_mean)
            ]

            payload = {
                "audio_features": audio_data,
                "vital_features": vital_data
            }

            try:
                response = requests.post(AI_BACKEND_URL, json=payload, timeout=15)

                if response.status_code == 200:
                    result = response.json()
                    emotion = result.get("emotion", "Unknown").upper()
                    confidence = result.get("confidence", 0.0)

                    st.success("Analysis Complete!")
                    st.metric(label="Detected Dog Emotion", value=emotion)
                    st.info(f"Prediction Confidence: {confidence:.2%}")
                else:
                    st.error(f"Server returned error code {response.status_code}: {response.text}")

            except requests.exceptions.Timeout:
                st.error("The server took too long to reply. Try clicking again!")
            except Exception as e:
                st.error(f"Network Connection Failed: {str(e)}")