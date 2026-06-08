import sys
import asyncio
import json
import threading
import io
import struct
from collections import deque
import requests
import numpy as np
import librosa
import paho.mqtt.client as mqtt
import websockets

from feature_extraction import extract_flat_audio_features

# On-device pipeline (ml/inference.py) — only becomes active once real
# trained .tflite files exist in models_trained/ (see ml/train_*.py).
# Until then, is_fully_available() is False and every call below falls
# straight through to the existing cloud XGBoost path, so nothing here
# changes current behavior until you've actually trained something.
try:
    import sys as _sys
    _sys.path.insert(0, "ml")
    from inference import OnDeviceEmotionPipeline
    _on_device = OnDeviceEmotionPipeline()
    if _on_device.is_fully_available():
        print("[ON-DEVICE] Trained models found — on-device inference is ACTIVE.")
    else:
        print("[ON-DEVICE] No trained models found yet — using existing cloud XGBoost pipeline "
              "(see ml/train_*.py to change this).")
except Exception as e:
    print(f"[ON-DEVICE] Not available ({e}) — using existing cloud XGBoost pipeline.")
    _on_device = None

# Rolling window of recent heart-rate readings. The trained model expects
# [hr_mean, hr_std, accel_mag] as its 3 vital features (see
# train_multimodal_fusion.py), which needs a short history, not just the
# single latest reading, to compute a meaningful std deviation.
hr_history = deque(maxlen=10)

# Force UTF-8 stdout/stderr so emoji in print() statements never crash the
# process when it's launched with output redirected to a file (this bit
# app.py on Windows — see app.err).
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# ── CONFIG ──────────────────────────────────────────────────────
HIVEMQ_HOST     = "broker.hivemq.com"
HIVEMQ_PORT     = 1883
WS_PORT         = 8765

# Connect with backend
AI_MODEL_URL    = "http://16.171.30.37:5000/predict"

# PC Mic sound detection settings (WebSocket path)
SILENCE_RMS     = 300
TRIGGER_RMS     = 400
SAMPLE_RATE     = 22050     # PC mic sample rate
SILENCE_HOLD_MS = 800

# Wokwi audio stream settings
WOKWI_SAMPLE_RATE = 22050   # must match sketch.ino SAMPLE_RATE

TOPICS = [
    "dogcollar/emotion",
    "dogcollar/location",
    "dogcollar/battery",
    "dogcollar/vitals",
    "dogcollar/temperature",
    "dogcollar/activity",
    "dogcollar/health",
    "dogcollar/audio/meta",     # Wokwi audio metadata
    "dogcollar/audio/stream",   # Wokwi raw PCM chunks
]
# ────────────────────────────────────────────────────────────────

connected_clients = set()
loop              = None
mqtt_client_ref   = None

# Shared dashboard state — all sensor values live here
state = {
    "emotion":    "Waiting...",
    "emotion_confidence": None,   # 0-1, from the real ML model; None when using the RMS fallback
    "audio_rms":  0,
    "session":    "",
    "mic_status": "idle",      # idle | listening | capturing | processing

    # Location
    "lat":        17.385044,
    "lng":        78.486671,

    # Battery
    "battery":    85.0,
    "voltage":    3.9,
    "charging":   False,
    "standby":    False,

    # Vitals
    "heart_rate": 0,
    "spo2":       0,
    "temperature": 0.0,

    # Activity
    "activity":   "Idle",
    "motion":     "Still",
    "steps":      0,

    # Health
    "health_score": 100,
}

# ── PC Mic capture buffer (WebSocket path) ───────────────────────
capture_buffer  = []
silence_counter = 0
triggered       = False

# ── Wokwi MQTT audio assembly buffer ────────────────────────────
wokwi_sessions  = {}
wokwi_lock      = threading.Lock()


# ── Helpers ─────────────────────────────────────────────────────
def compute_health_score() -> int:
    score = 100
    spo2  = state["spo2"]
    temp  = state["temperature"]
    hr    = state["heart_rate"]
    emo   = state["emotion"].lower()

    if   spo2 > 0 and spo2 < 90:  score -= 20
    elif spo2 > 0 and spo2 < 95:  score -= 10

    if   temp > 0 and temp > 40.0: score -= 20
    elif temp > 0 and temp > 39.0: score -= 10

    if   hr > 0 and hr > 180:      score -= 15
    elif hr > 0 and hr > 160:      score -= 8
    elif hr > 0 and hr < 50:       score -= 10

    if   "pain"       in emo:      score -= 20
    elif "aggressive" in emo:      score -= 15
    elif "scared"     in emo:      score -= 10
    elif "anxious"    in emo:      score -= 5

    return max(0, min(100, score))


# ── Vital feature vector for the fusion model ───────────────────
# Matches train_multimodal_fusion.py exactly: [hr_mean, hr_std, accel_mag].
# hr_mean/hr_std come from a real rolling window of recent readings.
# accel_mag has no true sensor equivalent yet in this telemetry pipeline —
# the collar firmware currently only reports a categorical activity/motion
# label, not a raw accelerometer magnitude. Until the IoT team adds that
# real value to the telemetry payload, this is a labeled best-effort proxy
# scaled to roughly the same range the model was trained on (per app.py's
# own pain-vs-sprint guard, which treats accel_mag > 3.5 as high-impact
# movement). Swap ACTIVITY_ACCEL_PROXY for a real sensor reading as soon
# as one is available — this is an approximation, not a measurement.
ACTIVITY_ACCEL_PROXY = {
    "idle": 0.9, "sleeping": 0.6, "resting": 1.0, "still": 1.0,
    "walking": 1.8, "running": 3.0, "pacing": 2.6,
}


def build_vital_features() -> list:
    if hr_history:
        hr_mean = float(np.mean(hr_history))
        hr_std = float(np.std(hr_history)) if len(hr_history) > 1 else 0.0
    else:
        hr_mean = float(state["heart_rate"]) if state["heart_rate"] else 70.0
        hr_std = 0.0

    activity_key = str(state.get("activity", "")).strip().lower()
    accel_mag = ACTIVITY_ACCEL_PROXY.get(activity_key, 1.0)

    return [hr_mean, hr_std, accel_mag]


# ── Multi-modal feature generator matching training exactly ────
def extract_and_call_api(samples: list, target_sr: int):
    """
    Builds the exact (audio_features[420], vital_features[3]) payload
    app.py's /predict expects, using the SAME extract_flat_audio_features
    function used at training time (per feature_extraction.py's explicit
    warning against reimplementing this logic separately). The previous
    version here computed a different, 140-length, mean-only feature set
    under a mismatched "features" key — every real call was rejected by
    app.py with a 400 error and silently fell back to the crude RMS
    heuristic below, meaning the trained model was never actually reached.

    Tries the on-device TFLite pipeline FIRST (only does anything once
    you've actually trained it — see ml/train_*.py), then falls back to
    this existing cloud XGBoost call, then to the RMS heuristic as a
    last resort. Each stage only activates if the one before it can't
    produce an answer, so nothing about current behavior changes until
    you've trained and dropped in real models_trained/*.tflite files.

    Returns (emotion, confidence) or (None, None) on any failure.
    """
    try:
        y = np.array(samples, dtype=np.float32) / 32768.0
        if len(y) == 0:
            return None, None

        vital_vector = build_vital_features()

        # ── Stage 1: on-device (only active once trained) ──────────
        if _on_device is not None and _on_device.is_fully_available():
            try:
                if _on_device.is_bark(y, sample_rate=target_sr) is False:
                    # Confidently not a bark at all — skip further stages
                    # entirely rather than forcing an emotion guess onto
                    # non-bark audio.
                    return None, None
                if _on_device.is_target_dog(y, sample_rate=target_sr) is False:
                    # A bark, but not from the enrolled dog — don't guess
                    # this dog's emotional state from someone else's bark.
                    return None, None
                emotion, confidence = _on_device.classify_emotion(y, vital_vector, sample_rate=target_sr)
                if emotion is not None:
                    return emotion, confidence
            except Exception as e:
                print(f"[ON-DEVICE] Inference error, falling back to cloud: {e}")

        # ── Stage 2: existing cloud XGBoost pipeline ────────────────
        audio_vector = extract_flat_audio_features(y, sr=target_sr)
        if audio_vector is None:
            print("[Feature Processing Error] -> extract_flat_audio_features returned None")
            return None, None

        payload = {
            "audio_features": audio_vector.tolist(),
            "vital_features": vital_vector,
        }
        headers = {"Content-Type": "application/json"}

        resp = requests.post(AI_MODEL_URL, json=payload, headers=headers, timeout=15)

        if resp.status_code == 200:
            data = resp.json()
            print(f"[AI Server Response] -> {data}")
            return data.get("emotion", None), data.get("confidence", None)
        else:
            print(f"[AI Error Code {resp.status_code}] -> {resp.text}")
            return None, None

    except Exception as e:
        print(f"[Feature Processing Error] -> {e}")
        return None, None


def rms_to_emotion(rms: int) -> str:
    if   rms > 8000: return "aggressive"
    elif rms > 4000: return "pain"
    elif rms > 1500: return "fearful"
    elif rms > 400:  return "happy"
    else:            return "neutral"


def publish_emotion_result(session_id: str, emotion: str, rms: int):
    global mqtt_client_ref
    if mqtt_client_ref is None:
        return

    result_payload = json.dumps({
        "session":    session_id,
        "emotion":    emotion,
        "confidence": 0.95,
    })
    mqtt_client_ref.publish("dogcollar/emotion/result", result_payload, retain=False)
    print(f"[MQTT] → dogcollar/emotion/result: {result_payload}")

    dash_payload = json.dumps({"emotion": emotion, "audio_rms": rms})
    mqtt_client_ref.publish("dogcollar/emotion", dash_payload, retain=True)
    print(f"[MQTT] → dogcollar/emotion: {dash_payload}")


# ── Broadcast to WebSocket dashboard ────────────────────────────
async def broadcast_all():
    if not connected_clients:
        return
    msg  = json.dumps(state)
    dead = set()
    for client in connected_clients:
        try:
            await client.send(msg)
        except Exception:
            dead.add(client)
    connected_clients.difference_update(dead)


# ── Process PC mic audio (WebSocket path) ────────────────────────
def process_audio_thread(samples: list, session_id: str = "pc_mic"):
    global state
    print(f"[PC-MIC] Extracting parameters from {len(samples)} samples...")
    state["mic_status"] = "processing"

    # Route through local feature compiler instead of wav file generation
    emotion, confidence = extract_and_call_api(samples, SAMPLE_RATE)

    if emotion is None:
        rms     = int((sum(s * s for s in samples) / len(samples)) ** 0.5)
        emotion = rms_to_emotion(rms)
        confidence = None
        print(f"[PC-MIC] Fallback RMS={rms} → {emotion}")
    else:
        rms = state["audio_rms"]

    state["emotion"]            = emotion
    state["emotion_confidence"] = confidence
    state["mic_status"]         = "listening"
    state["health_score"]       = compute_health_score()
    print(f"[PC-MIC] Emotion: {emotion}  Health: {state['health_score']}")

    publish_emotion_result(session_id, emotion, rms)

    if loop and loop.is_running():
        asyncio.run_coroutine_threadsafe(broadcast_all(), loop)


# ── Process Wokwi MQTT audio ─────────────────────────────────────
def process_wokwi_audio_thread(session_id: str, raw_bytes: bytes, sample_rate: int):
    global state
    print(f"[WOKWI] Processing session {session_id} — {len(raw_bytes)} bytes @ {sample_rate}Hz")
    state["mic_status"] = "processing"
    state["session"]    = session_id

    # Safely convert incoming bytes back into short integer arrays
    num_samples = len(raw_bytes) // 2
    if num_samples > 0:
        samples = list(struct.unpack(f"<{num_samples}h", raw_bytes))
        rms     = int((sum(s * s for s in samples) / num_samples) ** 0.5)
    else:
        samples = []
        rms     = 0

    # Route Wokwi samples right into the feature engine
    emotion, confidence = extract_and_call_api(samples, sample_rate)

    if emotion is None:
        emotion = rms_to_emotion(rms)
        confidence = None
        print(f"[WOKWI] Fallback RMS={rms} → {emotion}")

    state["emotion"]            = emotion
    state["emotion_confidence"] = confidence
    state["audio_rms"]          = rms
    state["mic_status"]         = "listening"
    state["health_score"]       = compute_health_score()
    print(f"[WOKWI] ✅ Session {session_id} → Emotion: {emotion}  RMS: {rms}  Health: {state['health_score']}")

    publish_emotion_result(session_id, emotion, rms)

    if loop and loop.is_running():
        asyncio.run_coroutine_threadsafe(broadcast_all(), loop)


# ── WebSocket handler (PC mic path) ──────────────────────────────
async def ws_handler(websocket):
    global capture_buffer, triggered, silence_counter

    connected_clients.add(websocket)
    print(f"[WS] Client connected — total: {len(connected_clients)}")
    await websocket.send(json.dumps(state))

    try:
        async for raw in websocket:
            try:
                data = json.loads(raw)
                if "pcm_samples" in data and "rms" in data:
                    rms     = int(data["rms"])
                    samples = data["pcm_samples"]
                    state["audio_rms"] = rms

                    if not triggered:
                        if rms >= TRIGGER_RMS:
                            print(f"[PC-MIC] Sound detected RMS={rms} — capturing...")
                            triggered        = True
                            silence_counter  = 0
                            capture_buffer   = list(samples)
                            state["mic_status"] = "capturing"
                            await broadcast_all()
                        else:
                            state["mic_status"] = "listening"
                    else:
                        if rms >= SILENCE_RMS:
                            capture_buffer.extend(samples)
                            silence_counter = 0
                        else:
                            capture_buffer.extend(samples)
                            silence_counter += 1
                            if silence_counter >= int(SILENCE_HOLD_MS / 100):
                                print(f"[PC-MIC] Bark ended — {len(capture_buffer)} samples")
                                triggered       = False
                                silence_counter = 0
                                buf_copy        = list(capture_buffer)
                                capture_buffer  = []
                                threading.Thread(
                                    target=process_audio_thread,
                                    args=(buf_copy,),
                                    daemon=True
                                ).start()
                    await broadcast_all()
            except Exception as e:
                print(f"[WS] Error: {e}")

    except websockets.exceptions.ConnectionClosedError:
        pass
    finally:
        connected_clients.discard(websocket)
        print(f"[WS] Client disconnected — total: {len(connected_clients)}")


# ── MQTT callbacks ────────────────────────────────────────────────
def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print("[MQTT] Connected to HiveMQ Broker!")
        for t in TOPICS:
            client.subscribe(t)
            print(f"[MQTT] Subscribed → {t}")


def on_message(client, userdata, msg):
    topic   = msg.topic
    payload = msg.payload

    if topic == "dogcollar/audio/meta":
        try:
            data         = json.loads(payload.decode(errors="ignore"))
            session_id   = data.get("session", "unknown")
            total_chunks = int(data.get("total_chunks", 0))
            sample_rate  = int(data.get("sample_rate", WOKWI_SAMPLE_RATE))
            print(f"[WOKWI] New session: {session_id}  chunks={total_chunks}  sr={sample_rate}")
            with wokwi_lock:
                wokwi_sessions[session_id] = {
                    "total_chunks": total_chunks,
                    "sample_rate":  sample_rate,
                    "chunks":       [],
                    "raw_bytes":    b"",
                }
        except Exception as e:
            print(f"[WOKWI-META] Error: {e}")
        return

    if topic == "dogcollar/audio/stream":
        if len(payload) < 17:
            return
        session_id = payload[:16].rstrip(b'\x00').decode(errors="ignore")
        audio_data = payload[16:]

        with wokwi_lock:
            if session_id not in wokwi_sessions:
                wokwi_sessions[session_id] = {
                    "total_chunks": 999,
                    "sample_rate":  WOKWI_SAMPLE_RATE,
                    "chunks":       [],
                    "raw_bytes":    b"",
                }
            sess = wokwi_sessions[session_id]
            sess["raw_bytes"] += audio_data
            sess["chunks"].append(len(sess["chunks"]) + 1)
            received = len(sess["chunks"])
            expected = sess["total_chunks"]
            print(f"[WOKWI] {session_id} chunk {received}/{expected}  ({len(audio_data)} bytes)")

            if received >= expected and expected < 999:
                raw_copy = sess["raw_bytes"]
                sr_copy  = sess["sample_rate"]
                del wokwi_sessions[session_id]
                print(f"[WOKWI] ✅ All {received} chunks received — compiling features...")
                threading.Thread(
                    target=process_wokwi_audio_thread,
                    args=(session_id, raw_copy, sr_copy),
                    daemon=True
                ).start()
        return

    raw = payload.decode(errors="ignore")
    try:
        data = json.loads(raw)

        if topic == "dogcollar/emotion":
            if state["mic_status"] == "idle":
                state["emotion"]   = data.get("emotion",   state["emotion"])
            state["audio_rms"] = data.get("audio_rms", state["audio_rms"])
            state["session"]   = data.get("session",   state["session"])

        elif topic == "dogcollar/location":
            state["lat"] = float(data.get("lat", state["lat"]))
            state["lng"] = float(data.get("lng", state["lng"]))

        elif topic == "dogcollar/battery":
            state["battery"]  = float(data.get("percent",  state["battery"]))
            state["voltage"]  = float(data.get("voltage",  state["voltage"]))
            state["charging"] = bool(data.get("charging",  state["charging"]))
            state["standby"]  = bool(data.get("standby",   state["standby"]))

        elif topic == "dogcollar/vitals":
            state["heart_rate"] = int(data.get("heart_rate", state["heart_rate"]))
            state["spo2"]       = int(data.get("spo2",       state["spo2"]))
            if state["heart_rate"]:
                hr_history.append(state["heart_rate"])

        elif topic == "dogcollar/temperature":
            state["temperature"] = float(data.get("temperature", state["temperature"]))

        elif topic == "dogcollar/activity":
            state["activity"] = data.get("activity", state["activity"])
            state["motion"]   = data.get("motion",   state["motion"])
            if "steps" in data:
                state["steps"] = int(data["steps"])

        elif topic == "dogcollar/health":
            if "heart_rate"   in data: state["heart_rate"]  = int(data["heart_rate"])
            if "spo2"         in data: state["spo2"]         = int(data["spo2"])
            if "temperature"  in data: state["temperature"]  = float(data["temperature"])
            if "steps"        in data: state["steps"]        = int(data["steps"])
            if   data.get("running"):  state["activity"] = "Running"
            elif data.get("sleeping"): state["activity"] = "Sleeping"
            elif data.get("resting"):  state["activity"] = "Resting"
            if "status" in data:       state["motion"]   = data["status"]
            if state["heart_rate"]:
                hr_history.append(state["heart_rate"])

        state["health_score"] = compute_health_score()

        if loop and loop.is_running():
            asyncio.run_coroutine_threadsafe(broadcast_all(), loop)

    except Exception as e:
        print(f"[ERROR] {e}")


def on_disconnect(client, userdata, disconnect_flags, reason_code, properties):
    print(f"[MQTT] Disconnected rc={reason_code}")


def mqtt_thread():
    global mqtt_client_ref
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect    = on_connect
    client.on_message    = on_message
    client.on_disconnect = on_disconnect
    client.max_inflight_messages_set(100)
    mqtt_client_ref = client
    while True:
        try:
            client.connect(HIVEMQ_HOST, HIVEMQ_PORT, keepalive=60)
            client.loop_forever()
        except Exception as e:
            print(f"[MQTT] Error: {e} — retry in 5s")
            import time; time.sleep(5)


async def heartbeat():
    while True:
        await asyncio.sleep(2)
        if connected_clients:
            await broadcast_all()


async def main():
    global loop
    loop = asyncio.get_running_loop()

    threading.Thread(target=mqtt_thread, daemon=True).start()

    print("=" * 55)
    print(" 🐾 Dog Collar Bridge (Wokwi + PC-Mic Live Tabular)")
    print(f"  WebSocket  : ws://localhost:{WS_PORT}")
    print(f"  AI Model   : {AI_MODEL_URL}")
    print("=" * 55)

    async with websockets.serve(ws_handler, "localhost", WS_PORT):
        await asyncio.gather(heartbeat(), asyncio.Future())


if __name__ == "__main__":
    asyncio.run(main())