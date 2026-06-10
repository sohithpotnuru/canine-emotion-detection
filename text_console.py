"""
Hi Buddy — Text Command Console
================================
A second, completely independent way to talk to the assistant: type a
question, get a text + spoken reply back — no microphone, no wake word,
no VAD tuning. Useful for typing when speaking isn't convenient, for
quick testing, or as a fallback if your mic/Bluetooth setup is being
finicky.

This talks to the SAME running voice_assistant_server.py over HTTP, so
run that first. It does not touch wake_word_listener.py or its state at
all — the two can run side by side safely, sharing the same server.

Usage:
    py text_console.py
    py text_console.py --server http://localhost:5001
    py text_console.py --no-voice     # text replies only, skip TTS playback
    py text_console.py --session-id work-desk   # separate conversation memory
"""

import sys
import time
import base64
import argparse
import tempfile
import subprocess

import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def play_audio_powershell(file_path: str, timeout_s: int = 25):
    """Self-contained headless MP3 playback. Independent of
    wake_word_listener.py's state machine on purpose — this script has
    no wake/listen/speak states of its own to manage."""
    import os
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
            capture_output=True, timeout=timeout_s, text=True
        )
        if result.returncode != 0:
            print(f"⚠️  Audio playback failed: {result.stderr.strip()}")
    except subprocess.TimeoutExpired:
        print("⚠️  Audio playback timed out")
    except Exception as e:
        print(f"⚠️  Audio playback error: {e}")
    finally:
        try:
            os.remove(abs_path)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Hi Buddy — type commands, no mic needed")
    parser.add_argument("--server", default="http://localhost:5001", help="Voice assistant server URL")
    parser.add_argument("--session-id", default="text-console", help="Conversation session id (keeps its own memory)")
    parser.add_argument("--no-voice", action="store_true", help="Skip TTS playback, text replies only (fastest)")
    args = parser.parse_args()

    print("=" * 55)
    print(" ⌨️  Hi Buddy: Text Command Console")
    print(f"  Server     : {args.server}")
    print(f"  Session id : {args.session_id}")
    print(f"  Voice reply: {'off' if args.no_voice else 'on'}")
    print("=" * 55)

    # Fail fast with a clear message if the server isn't up, rather than
    # a confusing timeout on the first real question.
    try:
        r = requests.get(args.server + "/", timeout=5)
        r.raise_for_status()
        info = r.json()
        print(f"🟢 Connected. Bridge (dog collar) connected: {info.get('bridge_connected')}")
    except Exception as e:
        print(f"❌ Could not reach the server at {args.server}: {e}")
        print("   Make sure voice_assistant_server.py is running first.")
        sys.exit(1)

    print("\nType a message and press Enter. Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not text:
            continue
        if text.lower() in ("exit", "quit"):
            print("Bye!")
            break

        start = time.time()
        try:
            resp = requests.post(
                f"{args.server}/assistant/query_text",
                json={"text": text, "session_id": args.session_id},
                timeout=30
            )
        except Exception as e:
            print(f"❌ Request failed: {e}\n")
            continue

        elapsed = time.time() - start

        if resp.status_code != 200:
            print(f"❌ Server error ({resp.status_code}): {resp.text}\n")
            continue

        result = resp.json()
        print(f"🤖 Buddy ({elapsed:.1f}s): {result.get('reply_text')}\n")

        if not args.no_voice:
            audio_b64 = result.get("reply_audio_base64")
            if audio_b64:
                audio_bytes = base64.b64decode(audio_b64)
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = tmp.name
                play_audio_powershell(tmp_path)
            else:
                print("⚠️  No voice reply received (check server terminal for a [TTS] error)\n")


if __name__ == "__main__":
    main()
