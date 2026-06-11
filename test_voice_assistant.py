"""
Quick manual test for voice_assistant_server.py.

Records a few seconds from your laptop mic, sends it to the running
voice assistant server, prints the transcript + reply, and plays the
spoken reply back through your speakers.

Install what this needs first:
    pip install sounddevice soundfile requests

Run it while voice_assistant_server.py is already running.
"""

import requests
import sounddevice as sd
import soundfile as sf

SERVER_URL = "http://localhost:5001/assistant/query"
RECORD_SECONDS = 4
SAMPLE_RATE = 16000

print(f"🎙️  Recording for {RECORD_SECONDS} seconds... ask your question now!")
audio = sd.rec(int(RECORD_SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
sd.wait()
sf.write("test_question.wav", audio, SAMPLE_RATE)
print("✅ Recorded. Sending to assistant...")

with open("test_question.wav", "rb") as f:
    response = requests.post(
        SERVER_URL,
        files={"audio": f},
        data={"session_id": "manual_test"},
        timeout=30,
    )

if response.status_code != 200:
    print(f"❌ Server error {response.status_code}: {response.text}")
else:
    result = response.json()
    print(f"\n🗣️  You said: {result['transcript']}")
    print(f"🎯 Detected intent: {result['intent']}")
    print(f"🤖 Reply: {result['reply_text']}")

    if result.get("reply_audio_base64"):
        import base64
        audio_bytes = base64.b64decode(result["reply_audio_base64"])
        with open("reply.mp3", "wb") as out:
            out.write(audio_bytes)
        print("🔊 Playing reply...")
        import os
        os.startfile("reply.mp3")  # Windows-only, opens in default player
    else:
        print("⚠️ No audio came back — check the server's terminal for a TTS error.")