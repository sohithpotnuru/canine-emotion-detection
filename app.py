import os
import threading
from pathlib import Path
import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent


def resolve_model_path(filename: str) -> Path:
    for candidate in (BASE_DIR / filename, BASE_DIR.parent / filename):
        if candidate.exists():
            return candidate
    return BASE_DIR / filename


# Target the production Late Fusion binary paths in the project root
AUDIO_MODEL_PATH = resolve_model_path('dog_audio_xgb.joblib')
VITALS_MODEL_PATH = resolve_model_path('dog_vitals_xgb.joblib')
ENCODER_PATH = resolve_model_path('label_encoder.joblib')

audio_model = None
vitals_model = None
label_encoder = None
model_load_error = None
model_lock = threading.Lock()

# ── EMOTION BACKUP MAPPING ────────────────────────────────────────
# Maps the numeric indices to their true respective classes as defined
# by the training script alphabetization constraints.
EMOTION_MAPPING = {
    0: "aggressive",
    1: "fearful",
    2: "happy",
    3: "neutral",
    4: "pain"
}
# ──────────────────────────────────────────────────────────────────

def load_models():
    global audio_model, vitals_model, label_encoder, model_load_error
    with model_lock:
        audio_model = None
        vitals_model = None
        label_encoder = None
        model_load_error = None

        print("🔄 Loading commercial Late Fusion dual-engine backend...")
        if AUDIO_MODEL_PATH.exists() and VITALS_MODEL_PATH.exists() and ENCODER_PATH.exists():
            try:
                audio_model = joblib.load(AUDIO_MODEL_PATH)
                vitals_model = joblib.load(VITALS_MODEL_PATH)
                label_encoder = joblib.load(ENCODER_PATH)
                print("✅ SUCCESS: Subsystem servers are live and completely stable!")
            except Exception as exc:
                model_load_error = str(exc)
                print(f"❌ CRITICAL INITIALIZATION ERROR: {exc}")
        else:
            missing = [str(path) for path in [AUDIO_MODEL_PATH, VITALS_MODEL_PATH, ENCODER_PATH] if not path.exists()]
            model_load_error = f"Missing model artifacts: {', '.join(missing)}"
            print(f"❌ ERROR: Production binaries missing. Looking for:\n - {AUDIO_MODEL_PATH}\n - {VITALS_MODEL_PATH}\n - {ENCODER_PATH}")


load_models()

@app.route("/", methods=["GET"])
def root():
    if audio_model is None or vitals_model is None or label_encoder is None:
        return jsonify({
            "status": "degraded",
            "detail": "Model artifacts are not available yet; prediction service is unavailable",
            "model_error": model_load_error,
        }), 200
    return jsonify({
        "status": "online",
        "detail": "Late Fusion dual-engine endpoint is active!",
        "model_status": "loaded",
    })


@app.route("/health", methods=["GET"])
def health():
    with model_lock:
        models_loaded = audio_model is not None and vitals_model is not None and label_encoder is not None
    return jsonify({
        "status": "ok",
        "service": "dog-emotion-api",
        "models_loaded": models_loaded,
    }), 200

@app.route("/predict", methods=["POST"])
def predict():
    if audio_model is None or vitals_model is None or label_encoder is None:
        return jsonify({"error": "Late Fusion Subsystems are offline"}), 503
        
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"error": "Empty or corrupted JSON payload"}), 400
            
        if "audio_features" not in data or "vital_features" not in data:
            return jsonify({"error": "Payload missing segregated 'audio_features' or 'vital_features' keys"}), 400
            
        # Parse inputs into correct shapes matching training shapes
        X_audio = np.array(data["audio_features"], dtype=np.float32).reshape(1, -1)   # Shape (1, 420)
        X_vitals = np.array(data["vital_features"], dtype=np.float32).reshape(1, -1)   # Shape (1, 3)
        
        # Defensive check against matrix corruption or transmission degradation
        if X_audio.shape[1] != 420 or X_vitals.shape[1] != 3:
            return jsonify({
                "error": f"Dimension structural error. Expected audio shape (1, 420) and vitals shape (1, 3). Got {X_audio.shape} and {X_vitals.shape}."
            }), 400

        # Extract independent multi-class probability vectors
        prob_audio = audio_model.predict_proba(X_audio)[0]
        prob_vitals = vitals_model.predict_proba(X_vitals)[0]
        
        # ── CORE FUSION LOGIC: WEIGHTED ENSEMBLE BLEND ──
        # 65% Audio Confidence + 35% Vital Biometrics
        blended_probabilities = (0.65 * prob_audio) + (0.35 * prob_vitals)
        pred_idx = int(np.argmax(blended_probabilities))
        confidence = float(blended_probabilities[pred_idx])
        
        try:
            detected_emotion = label_encoder.inverse_transform([pred_idx])[0]
        except Exception:
            detected_emotion = EMOTION_MAPPING.get(pred_idx, "unknown")
            
        # Intercept string definitions that are generic or invalid
        if str(detected_emotion).lower() == "dogs" or detected_emotion == "unknown":
            detected_emotion = EMOTION_MAPPING.get(pred_idx, "unknown")
        
        # ── VETERINARY SENSOR HEURISTICS VALIDATION GUARD ──
        # Pull live values from input variables to verify across clinical ranges
        hr_mean = X_vitals[0, 0]
        accel_mag = X_vitals[0, 2]
        
        if detected_emotion == "aggressive" and hr_mean < 115:
            # Physical aggression requires a sympathetic high spike; fallback to normal resting profile
            detected_emotion = "neutral"
            confidence = 0.50
            
        if detected_emotion == "pain" and accel_mag > 3.5:
            # Acute intense pain maps to cowering/guarding, not continuous high-impact sprints
            blended_probabilities[4] = 0.0  # Force clear pain weight
            pred_idx = int(np.argmax(blended_probabilities))
            try:
                detected_emotion = label_encoder.inverse_transform([pred_idx])[0]
            except Exception:
                detected_emotion = EMOTION_MAPPING.get(pred_idx, "unknown")
            confidence = float(blended_probabilities[pred_idx])

        return jsonify({
            "emotion": str(detected_emotion),
            "confidence": confidence
        }), 200
        
    except Exception as e:
        return jsonify({"error": f"Late Fusion pipeline evaluation failure: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    worker_count = int(os.environ.get("GUNICORN_WORKERS", "4"))
    print(f"Starting production-style server on port {port} with {worker_count} workers target")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
