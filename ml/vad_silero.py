"""
ml/vad_silero.py — Real speech detection using Silero VAD (section 3.3/5.4).

This is the actual fix for "background noise (fan, AC, TV) disturbs
commands" — not another threshold to tune. wake_word_listener.py's old
AdaptiveVAD only measures LOUDNESS (RMS energy). A steady fan hum can be
loud enough to look like speech, while your voice from across the room
can be quiet enough to look like noise — loudness alone can't tell them
apart. Silero VAD classifies audio by whether it has the spectral shape
of human speech, so a steady non-speech hum gets correctly ignored
regardless of how loud it is.

No training needed — this is a small, pretrained, MIT-licensed public
model (~2.3MB), not something built from scratch. It runs via
onnxruntime, not torch, so it stays lightweight enough for a Raspberry
Pi or even a phone.

Usage (see wake_word_listener.py for the full integration):
    vad = SileroVADGate(sample_rate=16000)
    is_speech = vad.is_speech(audio_chunk_int16_bytes_or_array)
"""

import numpy as np
import torch
from silero_vad import load_silero_vad


class SileroVADGate:
    def __init__(self, sample_rate: int = 16000, speech_threshold: float = 0.5):
        """
        speech_threshold: Silero's own probability output (0-1) that a
        frame is speech. 0.5 is Silero's own recommended default; raise
        it (e.g. 0.6-0.7) if you still get occasional false triggers in
        an especially noisy room, lower it (e.g. 0.35-0.4) if it's
        missing soft/distant speech.
        """
        if sample_rate not in (8000, 16000):
            raise ValueError("Silero VAD only supports 8000 or 16000 Hz — resample before calling this.")
        self.sample_rate = sample_rate
        self.threshold = speech_threshold
        self.model = load_silero_vad(onnx=True)
        self._reset_state()

    def _reset_state(self):
        # Silero VAD is stateful across consecutive frames from the same
        # stream (it looks at short-term context). Reset when starting a
        # new, unrelated audio stream (e.g. after a period of not
        # calling this at all).
        self.model.reset_states()

    def speech_probability(self, chunk) -> float:
        """
        chunk: 1-D array-like of audio samples, either float32 in
        [-1, 1] or int16 PCM (auto-detected and normalized).
        Returns Silero's raw speech probability (0.0-1.0) for this chunk.

        Silero expects frames of exactly 512 samples at 16kHz (or 256 at
        8kHz) — pad/truncate short chunks (e.g. this project's 480-sample
        30ms frames) to match, which is a normal, documented way to use it.
        """
        arr = np.asarray(chunk)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32) / 32768.0  # int16 -> float32 [-1, 1]

        required = 512 if self.sample_rate == 16000 else 256
        if len(arr) < required:
            arr = np.pad(arr, (0, required - len(arr)))
        elif len(arr) > required:
            arr = arr[:required]

        prob = self.model(torch.from_numpy(arr), self.sample_rate).item()
        return float(prob)

    def is_speech(self, chunk) -> bool:
        return self.speech_probability(chunk) >= self.threshold


if __name__ == "__main__":
    # Self-test: confirm real behavior on synthetic silence/noise before
    # wiring this into the live mic pipeline.
    gate = SileroVADGate(sample_rate=16000)

    silence = np.zeros(480, dtype=np.float32)
    noise = (np.random.randn(480) * 0.05).astype(np.float32)   # steady random "hum"-like noise
    loud_noise = (np.random.randn(480) * 0.5).astype(np.float32)  # LOUD noise — the case RMS-only VAD gets wrong

    print(f"Silence           -> speech prob: {gate.speech_probability(silence):.3f}")
    print(f"Quiet random noise -> speech prob: {gate.speech_probability(noise):.3f}")
    print(f"LOUD random noise  -> speech prob: {gate.speech_probability(loud_noise):.3f}  "
          f"(this is the case pure loudness-based VAD gets wrong — should still be low here)")
