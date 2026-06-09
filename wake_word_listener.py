import os
import sys
import time
import json
import wave
import argparse
import tempfile
import threading
import subprocess
from queue import Queue
from collections import deque
import numpy as np
import requests
import sounddevice as sd

# Force standard output to UTF-8 to prevent Windows terminal encoding crashes when printing emojis
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# ── CONFIG / GLOBALS ─────────────────────────────────────────────
DEFAULT_SERVER_URL = "http://localhost:5001"
SAMPLE_RATE = 16000  # 16kHz is standard for STT & audio classification
BLOCK_SIZE = 480     # 30ms frames at 16kHz
CHANNELS = 1

# States
STATE_SLEEPING = "SLEEPING"
STATE_LISTENING = "LISTENING"
STATE_PROCESSING = "PROCESSING"
STATE_SPEAKING = "SPEAKING"

state = STATE_SLEEPING
session_id = "buddy_session_" + str(int(time.time()))
last_interaction_time = 0
session_timeout = 30.0  # 30s session continuity

# ── SYNTHESIZE CHIMES (NO EXTERNAL FILES NEEDED) ────────────────
def play_tone(frequencies, durations, volume=0.3):
    """Synthesizes and plays a clean tone sequence without popping."""
    try:
        sample_rate = 22050
        audio_data = np.array([], dtype=np.float32)
        
        for freq, duration in zip(frequencies, durations):
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            tone = np.sin(2 * np.pi * freq * t)
            
            # Apply fade in/out envelope to prevent clicking
            envelope = np.ones_like(t)
            fade_samples = int(sample_rate * 0.015)
            if len(envelope) > fade_samples * 2:
                envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
                envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
            
            audio_data = np.concatenate([audio_data, tone * envelope])
            
        audio_data = audio_data * volume
        sd.play(audio_data, samplerate=sample_rate)
        sd.wait()
    except Exception as e:
        # Silently fail if sounddevice speaker is busy
        pass

def play_chime(chime_type="success"):
    if chime_type == "success":
        # Quick rising chime (wake word detected)
        play_tone([440, 880], [0.08, 0.12])
    elif chime_type == "processing":
        # Quick falling tones (speech finished)
        play_tone([660, 523], [0.07, 0.09])
    elif chime_type == "error":
        # Low flat warning buzzer
        play_tone([180], [0.35], volume=0.4)

# ── WINDOWS POWERSHELL HEADLESS MP3 PLAYER ──────────────────────
def play_audio_powershell(file_path):
    """Plays an MP3 file headlessly on Windows using System.Media.MediaPlayer."""
    global state
    state = STATE_SPEAKING
    
    abs_path = os.path.abspath(file_path)
    escaped_path = abs_path.replace("'", "''")
    
    ps_cmd = (
        f"$p = New-Object System.Media.MediaPlayer; "
        f"$p.Open('{escaped_path}'); "
        f"$p.Play(); "
        f"Start-Sleep -Milliseconds 300; "
        f"$timeout = 0; "
        f"while (($p.Position -lt $p.NaturalDuration -or $p.NaturalDuration.TotalMilliseconds -eq 0) -and $timeout -lt 200) {{ "
        f"  Start-Sleep -Milliseconds 100; "
        f"  $timeout++; "
        f"  if ($p.NaturalDuration.TotalMilliseconds -gt 0 -and $p.Position -ge $p.NaturalDuration) {{ break }} "
        f"}}"
    )
    
    try:
        result = subprocess.run(
            ["powershell", "-ExecutionPolicy", "Bypass", "-Command", ps_cmd],
            capture_output=True,
            timeout=25,
            text=True
        )
        if result.returncode != 0:
            print(f"⚠️  Audio playback failed (exit {result.returncode}): {result.stderr.strip()}")
    except subprocess.TimeoutExpired:
        print("⚠️  Audio playback timed out after 25s")
    except Exception as e:
        print(f"⚠️  Audio playback error: {e}")
    finally:
        # Safe cleanup of temporary file
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass

# ── ADAPTIVE VAD (VOICE ACTIVITY DETECTION) ─────────────────────
class AdaptiveVAD:
    """
    Production-grade adaptive energy-based voice activity detector,
    now gated by Silero VAD (ml/vad_silero.py) for the actual "is this
    speech" decision — RMS energy alone can't tell a loud fan/AC hum
    from a quiet voice, since both are just "loud" or "quiet" numbers.
    Silero classifies by the spectral shape of speech, so a steady
    non-speech noise gets rejected regardless of how loud it is.

    RMS is still used for speed/timing: it decides WHEN to check with
    Silero at all (cheap check, avoids running the model on obvious
    silence) and drives the silence-countdown for end-of-speech timing.
    The actual gate — "is this frame really speech" — is Silero's,
    not RMS's.
    """
    def __init__(self, threshold_mult=3.0, silence_seconds=1.2, sample_rate=16000,
                 use_silero=True, silero_threshold=0.5):
        self.threshold_mult = threshold_mult
        self.silence_seconds = silence_seconds
        self.sample_rate = sample_rate

        # Tracking variables
        self.noise_baseline = 50.0  # Initial placeholder RMS
        self.is_speech_active = False
        self.silence_frames_limit = int((silence_seconds * sample_rate) / BLOCK_SIZE)
        self.silence_counter = 0

        # Adaptation constants
        self.alpha = 0.05  # Slow drift to adapt to ambient noise

        self.silero = None
        if use_silero:
            try:
                from ml.vad_silero import SileroVADGate
                self.silero = SileroVADGate(sample_rate=sample_rate, speech_threshold=silero_threshold)
                print("[VAD] Silero speech classifier active — noise-robust wake detection enabled.")
            except Exception as e:
                print(f"[VAD] Silero unavailable ({e}) — falling back to RMS-only detection. "
                      f"Run: pip install -r requirements-voice-assistant.txt to enable it.")

    def process_frame(self, frame_samples):
        # Convert to float array & compute root mean square (RMS) energy
        samples = np.frombuffer(frame_samples, dtype=np.float32)
        rms = float(np.sqrt(np.mean(samples ** 2)) * 10000.0) # Scaled for visibility

        # Ambient noise calibration during silence
        if not self.is_speech_active:
            if rms < self.noise_baseline * 1.5:
                # Slowly adapt baseline to quiet room levels
                self.noise_baseline = (1 - self.alpha) * self.noise_baseline + self.alpha * rms

        # Guard minimum baseline level
        self.noise_baseline = max(10.0, self.noise_baseline)
        speech_threshold = self.noise_baseline * self.threshold_mult

        # State machine
        if not self.is_speech_active:
            if rms > speech_threshold:
                # RMS says "maybe" — confirm with Silero before committing
                # to a full recording. This is the actual fix for loud
                # non-speech noise (fans, AC, appliances) triggering
                # false wake attempts.
                if self.silero is not None and not self.silero.is_speech(samples):
                    return False, rms, speech_threshold
                self.is_speech_active = True
                self.silence_counter = 0
                return True, rms, speech_threshold
            return False, rms, speech_threshold
        else:
            if rms < speech_threshold * 0.8:  # Add a little hysteresis
                self.silence_counter += 1
                if self.silence_counter >= self.silence_frames_limit:
                    self.is_speech_active = False
                    self.silence_counter = 0
                    return False, rms, speech_threshold
            else:
                self.silence_counter = 0
            return True, rms, speech_threshold

# ── DRAW PREMIUM CONSOLE INTERFACE ─────────────────────────────
def print_ui(rms, threshold, is_speaking=False):
    """Prints a real-time VU meter and active status cleanly to the terminal."""
    global state
    
    # Visual status configurations
    status_colors = {
        STATE_SLEEPING: "\033[94m[💤 SLEEPING]\033[0m",      # Blue
        STATE_LISTENING: "\033[92m[🎙️  LISTENING]\033[0m",    # Green
        STATE_PROCESSING: "\033[93m[⏳ PROCESSING]\033[0m",   # Yellow
        STATE_SPEAKING: "\033[95m[🔊 SPEAKING]\033[0m",      # Purple
    }
    
    status_indicator = status_colors.get(state, state)
    
    # Calculate VU meter scale (0 to 30 characters)
    max_vu_chars = 25
    rms_ratio = min(1.0, rms / max(300.0, threshold * 2))
    vu_chars = int(rms_ratio * max_vu_chars)
    vu_meter = "█" * vu_chars + "░" * (max_vu_chars - vu_chars)
    
    # Print status line (carriage return keeps it on one line)
    sys.stdout.write(
        f"\r{status_indicator} VU: [{vu_meter}] RMS: {int(rms):<4} (Threshold: {int(threshold):<4})   "
    )
    sys.stdout.flush()

# ── WAV BUFFER WRITING ──────────────────────────────────────────
def save_audio_buffer(audio_chunks, output_path):
    """Saves accumulated float32 chunks into a standard 16kHz mono WAV file."""
    with wave.open(output_path, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(SAMPLE_RATE)
        
        # Concatenate and convert float32 to int16
        full_audio = np.concatenate(audio_chunks)
        int_audio = (full_audio * 32767.0).clip(-32768, 32767).astype(np.int16)
        wf.writeframes(int_audio.tobytes())

# ── WORKER THREAD FOR PROCESSING VOICE PIPELINE ────────────────
def process_conversation_thread(audio_chunks, server_url):
    """Sends recorded audio to the voice server and triggers response playback."""
    global state, last_interaction_time
    state = STATE_PROCESSING
    print_ui(0, 100)  # Refresh UI state
    
    # Save audio temporarily
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        temp_wav_path = tmp.name
        
    try:
        save_audio_buffer(audio_chunks, temp_wav_path)
        
        # Check if we are waiting for a wake word, or already in an active session
        is_session_active = (time.time() - last_interaction_time) < session_timeout
        
        if not is_session_active:
            # ──── STATE 1: WAKE WORD CHECK ────
            print(f"\n[SYSTEM] Checking wake word...")
            
            with open(temp_wav_path, "rb") as f:
                resp = requests.post(
                    f"{server_url}/assistant/transcribe",
                    files={"audio": f},
                    timeout=15
                )
                
            if resp.status_code == 200:
                transcript = resp.json().get("transcript", "").strip()
                print(f"[WAKE-CHECK] Transcribed: \"{transcript}\"")
                
                # Check for "buddy" in transcription
                cleaned_transcript = transcript.lower().replace(",", "").replace(".", "").replace("!", "")
                words = cleaned_transcript.split()
                
                if "buddy" in words or "body" in words:  # "body" added for acoustic overlap
                    print(f"🎉 Wake word detected!")
                    play_chime("success")
                    
                    # Update session
                    last_interaction_time = time.time()
                    state = STATE_LISTENING
                    
                    # Extract target command if user said it all in one breath
                    # e.g., "Hi Buddy, how is he doing?"
                    target_query = ""
                    buddy_index = -1
                    if "buddy" in words:
                        buddy_index = words.index("buddy")
                    elif "body" in words:
                        buddy_index = words.index("body")
                        
                    if buddy_index != -1 and buddy_index < len(words) - 1:
                        target_query = " ".join(words[buddy_index + 1:])
                        
                    if target_query:
                        print(f"[ONE-SHOT] Extracted command: \"{target_query}\"")
                        # Recurse process directly as an active conversation query
                        send_query_request(target_query, server_url)
                    else:
                        print("[SESSION] Listening for command...")
                else:
                    print("[SYSTEM] No wake word matched. Going back to sleep.")
                    state = STATE_SLEEPING
            else:
                state = STATE_SLEEPING
        else:
            # ──── STATE 2: ACTIVE COMMAND PROCESSING ────
            print(f"\n[SYSTEM] Sending query...")
            with open(temp_wav_path, "rb") as f:
                resp = requests.post(
                    f"{server_url}/assistant/query",
                    files={"audio": f},
                    data={"session_id": session_id},
                    timeout=20
                )
                
            if resp.status_code == 200:
                result = resp.json()
                print(f"🗣️  You said: {result.get('transcript')}")
                print(f"🤖 Buddy: {result.get('reply_text')}")
                
                # Play response audio if present
                audio_b64 = result.get("reply_audio_base64")
                if audio_b64:
                    import base64
                    audio_bytes = base64.b64decode(audio_b64)
                    
                    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as reply_tmp:
                        reply_path = reply_tmp.name
                        reply_tmp.write(audio_bytes)
                        
                    # Play headlessly in background and block thread until finished
                    play_audio_powershell(reply_path)
                else:
                    print("⚠️  No voice reply received (text-to-speech failed server-side — check the server's terminal for a [TTS] error)")
                    
                last_interaction_time = time.time()  # Keep session rolling
                state = STATE_LISTENING
            else:
                print(f"\n❌ Error from server: {resp.status_code} - {resp.text}")
                play_chime("error")
                state = STATE_LISTENING
                
    except Exception as e:
        print(f"\n❌ Pipeline thread error: {e}")
        play_chime("error")
        state = STATE_SLEEPING
    finally:
        try:
            if os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)
        except Exception:
            pass

def send_query_request(text_query, server_url):
    """
    Handles a one-shot command that arrived in the same breath as the wake
    word (e.g. "Hi Buddy, how is he doing?"). We already have the text from
    the wake-word transcription, so we hit /assistant/query_text directly
    instead of re-recording and re-transcribing audio.
    """
    global state, last_interaction_time
    state = STATE_PROCESSING
    try:
        resp = requests.post(
            f"{server_url}/assistant/query_text",
            json={"text": text_query, "session_id": session_id},
            timeout=20
        )

        if resp.status_code == 200:
            result = resp.json()
            print(f"🗣️  You said: {result.get('transcript')}")
            print(f"🤖 Buddy: {result.get('reply_text')}")

            audio_b64 = result.get("reply_audio_base64")
            if audio_b64:
                import base64
                audio_bytes = base64.b64decode(audio_b64)

                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as reply_tmp:
                    reply_path = reply_tmp.name
                    reply_tmp.write(audio_bytes)

                play_audio_powershell(reply_path)
            else:
                print("⚠️  No voice reply received (text-to-speech failed server-side — check the server's terminal for a [TTS] error)")

            last_interaction_time = time.time()
        else:
            print(f"\n❌ Error from server: {resp.status_code} - {resp.text}")
            play_chime("error")
    except Exception as e:
        print(f"\n❌ One-shot query error: {e}")
        play_chime("error")
    finally:
        state = STATE_LISTENING

# ── MAIN PIPELINE STREAM LOOP ────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="🐾 Hi Buddy Wake Word & Voice Pipeline Client")
    parser.add_argument("--server", default=DEFAULT_SERVER_URL, help="URL of voice assistant server")
    parser.add_argument("--threshold", type=float, default=3.0, help="VAD energy threshold multiplier")
    parser.add_argument("--silence", type=float, default=1.2, help="Seconds of silence to trigger end-of-speech")
    parser.add_argument("--device", type=int, default=None, help="Input audio device index")
    parser.add_argument("--list-devices", action="store_true", help="List available audio input devices and exit")
    parser.add_argument("--no-silero", action="store_true",
                         help="Disable Silero VAD speech classification, use RMS energy only (not recommended in noisy rooms)")
    parser.add_argument("--silero-threshold", type=float, default=0.5,
                         help="Silero speech-probability threshold (0-1). Raise if still getting false triggers, lower if missing soft speech.")
    args = parser.parse_args()
    
    if args.list_devices:
        print("🎙️ Available Audio Input Devices:")
        print(sd.query_devices())
        sys.exit(0)
        
    print("=" * 60)
    print(" 🐾 Hi Buddy: Production Voice Assistant Client")
    print(f"  Target Server : {args.server}")
    print(f"  VAD Threshold : {args.threshold}x baseline")
    print(f"  Silence window: {args.silence} seconds")
    print("=" * 60)
    
    # Test connection to assistant server
    try:
        resp = requests.get(args.server, timeout=3)
        if resp.status_code == 200:
            print("🟢 Connected to voice assistant server successfully!")
        else:
            print(f"🟡 Server online but returned code: {resp.status_code}")
    except Exception as e:
        print(f"🔴 WARNING: Cannot connect to voice assistant server ({e}).")
        print("   Make sure voice_assistant_server.py is running!")
        print("=" * 60)
        
    # Setup VAD
    vad = AdaptiveVAD(threshold_mult=args.threshold, silence_seconds=args.silence, sample_rate=SAMPLE_RATE,
                       use_silero=not args.no_silero, silero_threshold=args.silero_threshold)
    audio_queue = Queue()
    
    # sounddevice input stream callback
    def audio_callback(indata, frames, time_info, status):
        if status:
            pass
        audio_queue.put(indata.copy())
        
    # Start audio input stream
    try:
        stream = sd.InputStream(
            device=args.device,
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            dtype='float32',
            callback=audio_callback
        )
    except Exception as e:
        print(f"❌ Failed to open audio input stream: {e}")
        print("Please check your default microphone settings or list devices using --list-devices.")
        sys.exit(1)
        
    # Start the stream
    with stream:
        print("\n🟢 Monitoring microphone... Say 'Hi Buddy' or 'Buddy' to wake me up!")
        print_ui(0, 100)
        
        recorded_chunks = []
        # Rolling buffer of recent audio kept even while NOT speaking, so that
        # when speech is finally detected we can prepend the moment right
        # before the trigger — this is what was getting lost for slow/soft
        # speakers and made background noise (fan hum, etc.) worse, since a
        # higher noise threshold clips even more of the quiet word onset.
        PRE_ROLL_FRAMES = 10  # ~300ms at 30ms/frame
        pre_roll_buffer = deque(maxlen=PRE_ROLL_FRAMES)
        was_speaking = False
        
        while True:
            # Maintain session continuity timers
            global state, last_interaction_time
            is_session_active = (time.time() - last_interaction_time) < session_timeout
            
            if is_session_active and state == STATE_SLEEPING:
                state = STATE_LISTENING
                print("\n[SESSION] Session active! Listening without wake word...")
            elif not is_session_active and state == STATE_LISTENING:
                state = STATE_SLEEPING
                print("\n[SESSION] Session timed out. Going back to sleep...")
                
            # Process frames from queue
            if not audio_queue.empty():
                chunk = audio_queue.get()
                
                # We skip processing voice if the assistant is currently speaking or processing
                if state in (STATE_PROCESSING, STATE_SPEAKING):
                    # Keep VAD state reset
                    vad.is_speech_active = False
                    vad.silence_counter = 0
                    recorded_chunks = []
                    pre_roll_buffer.clear()
                    was_speaking = False
                    continue
                    
                # Run VAD on current frame
                is_speaking, rms, threshold = vad.process_frame(chunk.tobytes())
                print_ui(rms, threshold)
                
                if is_speaking:
                    if not was_speaking:
                        # Speech just started: splice in the pre-roll so we don't
                        # lose the soft/slow onset of the word.
                        recorded_chunks.extend(pre_roll_buffer)
                    recorded_chunks.append(chunk)
                else:
                    # Keep the pre-roll buffer topped up while quiet
                    pre_roll_buffer.append(chunk)
                    # Speech ended
                    if len(recorded_chunks) > 0:
                        # Ensure we recorded at least 0.5s of audio to prevent false triggers
                        total_samples = len(recorded_chunks) * BLOCK_SIZE
                        if total_samples >= SAMPLE_RATE * 0.5:
                            print(f"\n[VAD] Speech ended ({len(recorded_chunks)} frames). Processing...")
                            play_chime("processing")
                            
                            # Start processing thread to avoid blocking audio input stream
                            chunks_to_process = list(recorded_chunks)
                            threading.Thread(
                                target=process_conversation_thread,
                                args=(chunks_to_process, args.server),
                                daemon=True
                            ).start()
                            
                        recorded_chunks = []
                        pre_roll_buffer.clear()
                
                was_speaking = is_speaking
            else:
                # Idle delay
                time.sleep(0.01)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Wake word listener stopped. Goodbye!")
        sys.exit(0)
