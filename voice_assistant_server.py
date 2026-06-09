import os
import sys
import json
import time
import base64
import asyncio
import threading
import tempfile

# Force UTF-8 stdout/stderr so emoji in print() statements never crash the
# process when it's launched with output redirected to a file (this bit
# app.py on Windows — see app.err — so we guard against it here too).
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

try:
    from dotenv import load_dotenv
    load_dotenv()  # loads GROQ_API_KEY (and anything else) from a .env file if present
except ImportError:  # pragma: no cover - optional dependency
    pass

try:
    import websockets
except ImportError:  # pragma: no cover - optional dependency
    websockets = None

try:
    import edge_tts
except ImportError:  # pragma: no cover - optional dependency
    edge_tts = None

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None

from flask import Flask, request, jsonify

# ── CONFIG ──────────────────────────────────────────────────────
# This is bridge_server.py's existing dashboard websocket — the voice
# assistant piggybacks on it to read live dog state (emotion, vitals,
# activity) instead of re-running detection for every question.
BRIDGE_WS_URL = "ws://localhost:8765"

# Groq: the same `openai` Python package, just pointed at Groq's URL.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

STT_MODEL = "whisper-large-v3-turbo"   # fast, good fit for a live assistant
LLM_MODEL = "llama-3.3-70b-versatile"

# edge-tts: free, no API key needed at all, runs against Microsoft Edge's
# voice engine. "en-IN-NeerjaNeural" is a natural Indian-English voice —
# swap for any voice edge-tts supports.
TTS_VOICE = "en-IN-NeerjaNeural"

# How long a "continue chat" session stays alive without a new "Hi Buddy"
SESSION_TIMEOUT_SECONDS = 30
# ────────────────────────────────────────────────────────────────

client = None
if GROQ_API_KEY and OpenAI is not None:
    # timeout=15s + max_retries=1 keeps a flaky network from hanging a
    # request indefinitely — a slow/stuck call was one of the "why is
    # this so slow" causes, since the old client had no timeout at all.
    client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL, timeout=15.0, max_retries=1)

app = Flask(__name__)

# ── Live dog state, kept in sync with bridge_server.py's broadcasts ──
latest_dog_state = {}
state_lock = threading.Lock()

# ── Short-term conversation memory, keyed by session id ──────────
conversations = {}
conv_lock = threading.Lock()

# ── Keyword-based intent routing ─────────────────────────────────
# Fast and free — no LLM call needed just to figure out what kind of
# question this is. Falls through to "general" if nothing matches.
DOG_STATUS_KEYWORDS = {
    # Checked first: explicit requests for the "translation" — the core
    # product feature. If the user asks this directly, always give the
    # first-person translated read, not a dry stats readout.
    "translate":   ["what is he saying", "what's he saying", "translate",
                     "what is he trying to say", "what does he want",
                     "what is buddy saying", "what's buddy trying to say"],
    # Natural catch-all phrasing like "how's Buddy doing?" also gets the
    # translated read — that's the more useful, more sellable answer than
    # a dry number for a vague "how is he" question.
    "overall":     ["how's he doing", "how is he doing", "how's buddy doing",
                     "is he doing", "is he ok", "is he okay", "is he alright",
                     "how's he", "how is he"],
    "temperature": ["temperature", "temp", "fever", "hot"],
    "heart_rate":  ["heart rate", "heartbeat", "pulse", "bpm"],
    "emotion":     ["mood", "emotion", "feeling", "feel", "stressed", "happy",
                     "scared", "aggressive", "pain", "upset"],
    "activity":    ["activity", "moving", "walking", "running", "resting", "sleeping"],
    "battery":     ["battery", "charge", "charging"],
    "health":      ["health", "wellbeing", "well-being"],
}


def classify_intent(text: str):
    """Very lightweight keyword router. Returns (intent, subtopic)."""
    lowered = text.lower()
    for subtopic, keywords in DOG_STATUS_KEYWORDS.items():
        if any(kw in lowered for kw in keywords):
            return "dog_status", subtopic
    if any(w in lowered for w in ["remind", "reminder", "alarm", "timer"]):
        return "reminder", None
    return "general", None


# Thresholds mirror bridge_server.py's compute_health_score() so the
# assistant's "you should check with a vet" guardrail fires on the same
# real conditions the health score itself is built from — never just a
# vibe from the LLM. This check is deterministic and always runs; it does
# not depend on (and cannot be skipped by) anything the LLM generates.
def is_health_concerning(s: dict):
    spo2 = s.get("spo2", 0) or 0
    temp = s.get("temperature", 0) or 0
    hr = s.get("heart_rate", 0) or 0
    emo = str(s.get("emotion", "")).lower()
    score = s.get("health_score", 100)

    if spo2 and spo2 < 90:
        return "his oxygen reading is quite low"
    if temp and temp > 40.0:
        return "his temperature is quite high"
    if hr and (hr > 180 or hr < 50):
        return "his heart rate looks unusual"
    if "pain" in emo:
        return "he may be in pain"
    if score < 60:
        return "a few of his readings look off together"
    return None


EMOTION_PHRASING = {
    # Old 5-class labels (existing XGBoost cloud model)
    "pain": "seems to be in pain",
    "aggressive": "seems agitated",
    "fearful": "seems scared",
    "happy": "seems happy",
    "neutral": "seems calm",
    # New 7-class labels (on-device pipeline, once trained — see ml/models.py EMOTION_CLASSES_V2)
    "excited_playful": "seems excited and playful",
    "alert_guarding": "is alert, like he's noticed something",
    "lonely_anxious": "seems lonely or anxious",
    "communication": "is trying to tell you something",
    "caution_intimidation": "seems uneasy or on edge",
    "possible_pain": "might be in pain",
    "normal": "seems calm",
}


def phrase_emotion(emo: str) -> str:
    return EMOTION_PHRASING.get(str(emo).lower(), f"seems {emo}")


def answer_dog_status(subtopic: str) -> str:
    """
    Answers using the LIVE state from bridge_server.py, not a fresh
    model call for factual readings — those need to be numbers you can
    trust, not something an LLM could subtly reword or misstate.

    "overall", "emotion", and "translate" are different: that's the core
    "what is my dog actually feeling/saying" feature, and a flat stats
    readout ("he seems happy, health score 90") is not what makes this
    product worth buying — a natural first-person translation is. Those
    three route to translate_dog_voice() below.
    """
    with state_lock:
        s = dict(latest_dog_state)

    if not s:
        return "I don't have a current reading from the collar yet."

    if subtopic == "temperature":
        temp = s.get("temperature", 0)
        if not temp:
            return "I don't have a reliable temperature reading right now."
        note = " That's higher than normal for a dog — worth keeping an eye on." if temp > 39.5 else ""
        return f"Buddy's temperature is {temp:.1f} degrees Celsius.{note}"

    if subtopic == "heart_rate":
        hr = s.get("heart_rate", 0)
        if not hr:
            return "I don't have a reliable heart rate reading right now."
        return f"Buddy's heart rate is {hr} beats per minute."

    if subtopic in ("emotion", "overall", "translate"):
        return translate_dog_voice(s)

    if subtopic == "activity":
        return f"Buddy is currently {str(s.get('activity', 'unknown')).lower()}."

    if subtopic == "battery":
        return f"The collar battery is at {s.get('battery', 0):.0f} percent."

    if subtopic == "health":
        score = s.get("health_score", 100)
        concern = is_health_concerning(s)
        if concern:
            return f"Buddy's health score is {score} out of 100 — {concern}, so it might be worth a vet check."
        return f"Buddy's overall health score is {score} out of 100."

    return "I'm not sure how to answer that about Buddy yet."


def translate_dog_voice(s: dict) -> str:
    """
    The actual "translation" feature: turns the collar's real emotion +
    vitals reading into a short first-person line as if the dog were
    speaking, grounded ONLY in the facts we actually have (never invents
    symptoms or context we weren't given). Falls back to a plain,
    template-based statement if the LLM is unavailable, so the feature
    degrades gracefully instead of going silent.
    """
    emo = s.get("emotion", "unknown")
    if not emo or str(emo).lower() in ("unknown", "waiting..."):
        return "I haven't picked up a clear reading on his mood yet."

    activity = str(s.get("activity", "unknown")).lower()
    score = s.get("health_score", 100)
    confidence = s.get("emotion_confidence")
    concern = is_health_concerning(s)

    # Hedge language based on real model confidence, not a guess — a
    # translation the assistant is unsure about should sound unsure.
    if confidence is not None and confidence < 0.55:
        certainty_note = "I'm not fully certain, but it seems like "
    elif confidence is None:
        certainty_note = "Best guess based on how he sounds: "
    else:
        certainty_note = ""

    vet_note = f" You might want to get him checked — {concern}." if concern else ""

    if client is None:
        # Deterministic fallback if the LLM is unavailable
        return f"{certainty_note}Buddy {phrase_emotion(emo)} right now, and he's currently {activity}.{vet_note}".strip()

    facts = (
        f"emotion={emo}, activity={activity}, health_score={score}/100, "
        f"temperature={s.get('temperature')}, heart_rate={s.get('heart_rate')}, "
        f"spo2={s.get('spo2')}, confidence={confidence}"
    )
    system_prompt = (
        "You are translating a dog's real collar sensor reading into one short, "
        "warm, first-person sentence, as if giving voice to what the dog might be "
        "feeling or trying to say. Base it ONLY on these exact facts, never invent "
        "symptoms, causes, or details not given: " + facts + ". "
        "Under 20 words. Speak as the dog in first person ('I feel...', 'I'm..."
        "'). If a plausible everyday reason fits the given facts (e.g. running -> "
        "tired/excited, pacing -> restless), you may mention it briefly, but do not "
        "invent anything not implied by the facts above."
    )
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": "Translate his current state."}],
            max_tokens=60,
            temperature=0.7,
        )
        translated = resp.choices[0].message.content.strip().strip('"')
        return f"{certainty_note}{translated}{vet_note}".strip()
    except Exception as e:
        print(f"[TRANSLATE-ERROR] LLM translation failed, using fallback: {e}")
        return f"{certainty_note}Buddy {phrase_emotion(emo)} right now, and he's currently {activity}.{vet_note}".strip()


def answer_general(text: str, history: list) -> str:
    if client is None:
        return "The assistant isn't configured with an API key yet."

    with state_lock:
        s = dict(latest_dog_state)

    if s:
        context_note = (
            f" Current collar reading you can use if relevant: emotion={s.get('emotion')}, "
            f"activity={s.get('activity')}, health_score={s.get('health_score')}/100, "
            f"temperature={s.get('temperature')}, heart_rate={s.get('heart_rate')}. "
            "Only mention these if the question is actually about the dog or "
            "something that depends on his current state (e.g. 'should I walk him now')."
        )
    else:
        context_note = " No live collar reading is available right now."

    messages = [{
        "role": "system",
        "content": (
            "You are Buddy, a friendly voice assistant paired with a smart dog "
            "collar that tracks vitals and mood. Keep your response strictly "
            "under 15 words. Be extremely concise, conversational, and direct. "
            "Never give a definitive medical diagnosis — for anything that sounds "
            "like a real health concern, suggest checking with a vet instead of "
            "guessing what's wrong." + context_note
        )
    }]
    messages.extend(history)
    messages.append({"role": "user", "content": text})
    resp = client.chat.completions.create(model=LLM_MODEL, messages=messages, max_tokens=60)
    return resp.choices[0].message.content.strip()


def transcribe_audio(audio_path: str) -> str:
    if client is None:
        raise RuntimeError("Speech services are not configured; set GROQ_API_KEY and install the openai package")
    with open(audio_path, "rb") as f:
        result = client.audio.transcriptions.create(model=STT_MODEL, file=f)
    return result.text


def synthesize_speech(text: str) -> bytes:
    """
    edge-tts is async-only, so we run its event loop here and return
    plain bytes — same contract as before, callers don't need to change.
    """
    if edge_tts is None:
        raise RuntimeError("edge-tts is not installed")

    async def _generate():
        communicate = edge_tts.Communicate(text, voice=TTS_VOICE)
        chunks = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                chunks.extend(chunk["data"])
        return bytes(chunks)

    return asyncio.run(_generate())


def get_session(session_id: str):
    with conv_lock:
        now = time.time()
        sess = conversations.get(session_id)
        if sess is None or (now - sess["last_active"]) > SESSION_TIMEOUT_SECONDS:
            sess = {"history": [], "last_active": now}
            conversations[session_id] = sess
        return sess


def update_session(session_id: str, user_text: str, reply_text: str):
    with conv_lock:
        sess = conversations.get(session_id, {"history": [], "last_active": time.time()})
        sess["history"].append({"role": "user", "content": user_text})
        sess["history"].append({"role": "assistant", "content": reply_text})
        sess["history"] = sess["history"][-6:]  # keep last 3 exchanges
        sess["last_active"] = time.time()
        conversations[session_id] = sess


def _session_cleanup_loop():
    """Runs forever in the background, purging sessions that have been
    idle well past their timeout so a long-running server doesn't slowly
    accumulate stale entries in `conversations` in memory."""
    while True:
        time.sleep(300)  # every 5 minutes
        cutoff = time.time() - (SESSION_TIMEOUT_SECONDS * 20)
        with conv_lock:
            stale = [sid for sid, sess in conversations.items() if sess["last_active"] < cutoff]
            for sid in stale:
                del conversations[sid]
        if stale:
            print(f"[CLEANUP] Purged {len(stale)} stale session(s)")


@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "status": "online",
        "detail": "Hi Buddy voice assistant is active",
        "bridge_connected": bool(latest_dog_state),
    })


@app.route("/assistant/transcribe", methods=["POST"])
def assistant_transcribe():
    """
    Accepts an uploaded audio file (multipart/form-data, field 'audio')
    and returns its transcription using the fast Whisper STT model.
    """
    if "audio" not in request.files:
        return jsonify({"error": "Missing 'audio' file in form-data"}), 400

    audio_file = request.files["audio"]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        transcript = transcribe_audio(tmp_path)
    except Exception as e:
        print(f"[STT-ERROR] /assistant/transcribe failed: {e}")
        return jsonify({"error": f"Speech-to-text failed: {e}"}), 502
    finally:
        os.remove(tmp_path)

    if not transcript or not transcript.strip():
        return jsonify({"error": "Could not understand audio"}), 422

    return jsonify({"transcript": transcript}), 200


def handle_transcript(transcript: str, session_id: str):
    """
    Shared pipeline: transcript -> intent -> answer -> TTS.
    Used by both /assistant/query (audio in) and /assistant/query_text
    (already-known text in, e.g. a one-shot "Hi Buddy, how is he doing?").
    Returns a (response_dict, http_status) tuple.
    """
    intent, subtopic = classify_intent(transcript)
    sess = get_session(session_id)

    if intent == "dog_status":
        reply_text = answer_dog_status(subtopic)
    elif intent == "reminder":
        reply_text = "Reminders aren't hooked up yet, but I heard you loud and clear."
    else:
        try:
            reply_text = answer_general(transcript, sess["history"])
        except Exception as e:
            print(f"[LLM-ERROR] answer_general failed: {e}")
            return {"error": f"Assistant reply failed: {e}"}, 502

    update_session(session_id, transcript, reply_text)

    audio_b64 = None
    try:
        audio_bytes = synthesize_speech(reply_text)
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    except Exception as e:
        print(f"[TTS] failed: {e}")

    return {
        "transcript": transcript,
        "intent": intent,
        "reply_text": reply_text,
        "reply_audio_base64": audio_b64,
    }, 200


@app.route("/assistant/query", methods=["POST"])
def assistant_query():
    """
    Accepts an uploaded audio file (multipart/form-data, field 'audio')
    plus an optional 'session_id' field. Runs STT -> intent -> answer -> TTS.
    """
    if "audio" not in request.files:
        return jsonify({"error": "Missing 'audio' file in form-data"}), 400

    session_id = request.form.get("session_id", "default")
    audio_file = request.files["audio"]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        transcript = transcribe_audio(tmp_path)
    except Exception as e:
        print(f"[STT-ERROR] /assistant/query failed: {e}")
        return jsonify({"error": f"Speech-to-text failed: {e}"}), 502
    finally:
        os.remove(tmp_path)

    if not transcript or not transcript.strip():
        return jsonify({"error": "Could not understand audio"}), 422

    body, status = handle_transcript(transcript, session_id)
    return jsonify(body), status


@app.route("/assistant/query_text", methods=["POST"])
def assistant_query_text():
    """
    Accepts JSON {"text": "...", "session_id": "..."} and runs the same
    intent -> answer -> TTS pipeline as /assistant/query, skipping STT.
    This is what powers one-shot commands like "Hi Buddy, how is he doing?"
    where the wake word and the command arrive in the same breath, so the
    text is already known from the wake-word transcription and doesn't
    need to be re-recorded and re-transcribed.
    """
    data = request.get_json(force=True, silent=True) or {}
    text = (data.get("text") or "").strip()
    session_id = data.get("session_id", "default")

    if not text:
        return jsonify({"error": "Missing 'text' in JSON body"}), 400

    body, status = handle_transcript(text, session_id)
    return jsonify(body), status


# ── Background: stay connected to bridge_server.py's dashboard socket ──
async def _bridge_listener():
    global latest_dog_state
    if websockets is None:
        print("[BRIDGE-LINK] websockets package missing; bridge listener disabled")
        return

    fail_count = 0
    while True:
        try:
            async with websockets.connect(BRIDGE_WS_URL) as ws:
                print(f"[BRIDGE-LINK] Connected to {BRIDGE_WS_URL}")
                fail_count = 0
                async for raw in ws:
                    try:
                        data = json.loads(raw)
                        with state_lock:
                            latest_dog_state = data
                    except Exception:
                        pass
        except Exception as e:
            fail_count += 1
            # Bridge_server.py is optional — the assistant works fine without
            # it, so don't spam the console with a disconnect line every 3s
            # forever. Print once, then only every ~10th attempt (~30s).
            if fail_count == 1 or fail_count % 10 == 0:
                print(f"[BRIDGE-LINK] Not connected ({e}) — retrying quietly every 3s "
                      f"(this is optional; the assistant works without it)")
            await asyncio.sleep(3)


def start_bridge_listener_thread():
    threading.Thread(target=lambda: asyncio.run(_bridge_listener()), daemon=True).start()


if __name__ == "__main__":
    start_bridge_listener_thread()
    threading.Thread(target=_session_cleanup_loop, daemon=True).start()
    port = int(os.environ.get("PORT", 5001))
    print("=" * 55)
    print(" 🐾 Hi Buddy Voice Assistant Server")
    print(f"  HTTP       : http://0.0.0.0:{port}")
    print(f"  Bridge link: {BRIDGE_WS_URL}")
    print(f"  Groq key   : {'set' if GROQ_API_KEY else 'MISSING — set GROQ_API_KEY env var'}")
    print("=" * 55)

    import sys
    if "--dev" in sys.argv:
        # Flask's built-in dev server: single-threaded, only useful for
        # debugging one request at a time with the reloader.
        print("[DEV] Starting single-threaded Flask dev server (use for debugging only)...")
        app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
    else:
        try:
            from waitress import serve
            # Multi-threaded by default now — the old code only used Waitress
            # behind an explicit --prod flag, so every local run was actually
            # hitting Flask's single-threaded dev server, which serializes
            # requests (mic query + bridge state + text console all queued
            # up one at a time) and was a real source of the "slow" feeling.
            print("[SERVER] Starting Waitress (multi-threaded, 16 threads)...")
            serve(app, host="0.0.0.0", port=port, threads=16)
        except ImportError:
            print("[WARNING] Waitress not installed — falling back to Flask dev server (single-threaded).")
            app.run(host="0.0.0.0", port=port, debug=False, threaded=True)