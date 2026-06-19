# Implementation Guide — Production ML Pipeline Upgrade

This covers everything added in this session: a new `ml/` package implementing
the production design doc against your existing, already-fixed
`canine_emotion_detection-main` codebase. Nothing here breaks what already
works — every new piece degrades gracefully to the current behavior until
you've actually trained it.

## 1. What's real right now vs. what needs your data

| Component | Status | Needs training data? |
|---|---|---|
| Silero VAD (noise-robust wake detection) | **Working now** | No — pretrained |
| Speaker verification (Resemblyzer) | **Working now** | No — pretrained, just needs *your* enrollment |
| TF feature extraction (`features_tf.py`) | **Working now** | N/A — it's a fixed pipeline, not a model |
| TFLite INT8 converter | **Working now** | N/A |
| Bark detector | Architecture ready, untrained | **Yes** — your dog + non-bark sounds |
| Dog voiceprint (identity) | Architecture ready, untrained | **Yes** — your dog + 5-10 other dogs |
| Multi-modal emotion classifier | Architecture ready, untrained | **Yes** — synchronized audio+vitals, labeled |
| On-device inference wiring | **Working now**, cleanly does nothing until trained | N/A |

"Working now" means I actually ran it and verified real, correct output — not
just wrote the code. The three untrained models will run without crashing but
their predictions are meaningless (random weights) until you train them —
there's no way around collecting real data for those three specifically.

## 2. Install on your machine

```powershell
cd canine_emotion_detection-main
py -m pip install -r requirements-voice-assistant.txt
py -m pip install tensorflow-cpu onnxruntime silero-vad resemblyzer setuptools soundfile librosa
```

If you hit `ModuleNotFoundError: No module named 'pkg_resources'` — this is a
known conflict between very new `setuptools` versions and older packages
(`webrtcvad`, pulled in by resemblyzer) that still expect it. Fix:
```powershell
py -m pip install "setuptools<81"
```

## 3. Enable noise-robust wake detection (works immediately)

Nothing to train — just run as usual:
```powershell
py wake_word_listener.py --threshold 5 --silence 1.5
```
You should see `[VAD] Silero speech classifier active — noise-robust wake
detection enabled.` at startup. If you instead see `[VAD] Silero unavailable
(...) — falling back to RMS-only detection`, the install step above didn't
fully complete — check the error message it prints for which package is
missing.

Tuning:
- Still getting false triggers in a loud room? Raise `--silero-threshold` (try `0.65`).
- Missing soft/distant speech? Lower it (try `0.4`).

## 4. Enable speaker verification (works immediately, once you enroll)

Record 15-30 seconds of yourself talking naturally (counting, a sentence,
free speech) as a `.wav` file, then:
```powershell
py ml\speaker_id.py enroll --name santhosh --wav my_voice.wav
```
Test it:
```powershell
py ml\speaker_id.py verify --name santhosh --wav another_clip.wav
```
This isn't wired into `wake_word_listener.py`'s live loop yet — it's a
working, tested module ready to be called as an extra gate after wake-word
detection (e.g. "only proceed if `verify_from_array()` returns True"). I left
this as a deliberate integration point rather than forcing it live, since
you'll want to decide the UX (e.g. should an unrecognized voice be silently
ignored, or get a "sorry, I don't recognize you" reply?).

## 5. Training the three models — in this order, not in parallel

They depend on each other, so train in this sequence:

**Step 1 — Bark detector** (Phase 1, needs the least data):
```powershell
mkdir data\bark_detector\bark
mkdir data\bark_detector\non_bark
# drop .wav clips into each folder — see the comment block at the
# bottom of ml/train_bark_detector.py for exactly what to collect
py ml\train_bark_detector.py --data-dir data\bark_detector --epochs 30
```

**Step 2 — Dog voiceprint** (Phase 2, needs other dogs too):
```powershell
mkdir data\dog_voiceprint\target_dog
mkdir data\dog_voiceprint\other_dogs\neighbor_dog_1
# ... at least one other_dogs/<name>/ folder, ideally 5-10
py ml\train_dog_voiceprint.py --data-dir data\dog_voiceprint --epochs 30
```
After training, enroll your dog's profile for the identity check:
```powershell
py -c "
import numpy as np
from ml.inference import OnDeviceEmotionPipeline
p = OnDeviceEmotionPipeline()
embeddings = [p.embed(load_your_wav_here) for load_your_wav_here in [...]]  # your dog's clips
profile = np.mean(embeddings, axis=0)
np.save('models_trained/target_dog_profile.npy', profile)
"
```
(This enrollment script is intentionally left for you to run once you have
real clips loaded — say the word if you want me to turn it into a proper
CLI tool like `speaker_id.py`'s.)

**Step 3 — Emotion classifier** (Phase 3, the hard one — see section 6):
```powershell
py ml\train_emotion_classifier.py --data-dir data\emotion --encoder-path models_trained\dog_voiceprint.keras --epochs 40
```

After each step, models land in `models_trained/*.tflite`. `bridge_server.py`
automatically checks for these on startup — you'll see:
```
[ON-DEVICE] Trained models found — on-device inference is ACTIVE.
```
instead of the current:
```
[ON-DEVICE] No trained models found yet — using existing cloud XGBoost pipeline.
```
No code changes needed to switch over — just train and restart.

## 6. The hard part: collecting emotion-labeled data

This is genuinely the multi-week-to-multi-month piece, and no shortcut exists.
Practically:
- Record your dog during clearly-known contexts: play/fetch (excited_playful),
  home alone (lonely_anxious), doorbell/stranger (alert_guarding), a vet visit
  or known discomfort (possible_pain), resting (normal).
- Have the collar logging vitals *at the same time* as each recording.
- Fill in `data/emotion/labels.csv` per the format documented at the top of
  `ml/train_emotion_classifier.py`.
- Start small — even 15-20 clips per class is enough to get a first working
  model to iterate from; it won't be accurate yet, but the pipeline (data →
  train → tflite → bridge_server picks it up) will be proven end-to-end, and
  you improve accuracy by adding more data over time.

## 7. Common bugs and how I already fixed them (so you recognize them if they resurface)

- **GRU layers silently failing INT8 TFLite conversion.** Fixed by adding
  `unroll=True` to the bark detector's GRU layer — a known TFLite limitation
  with dynamic recurrent ops. If you ever add another recurrent layer
  elsewhere, remember this.
- **Wrong `Reshape` dimensions before a GRU/Dense layer.** The design doc's
  own example code had a hardcoded, incorrect reshape number. Fixed by
  computing it from the actual tensor shape (`x.shape`) instead of a magic
  number — always prefer this pattern over hardcoding when you modify these
  architectures later.
- **INT8 TFLite outputs being garbage-looking numbers.** Quantized models
  need inputs quantized and outputs dequantized using each tensor's own
  `(scale, zero_point)` — `ml/inference.py`'s `_TFLiteModel` wrapper handles
  this. If you ever call a `.tflite` model directly without this wrapper,
  you'll get technically-valid-looking but meaningless numbers, not a crash —
  the dangerous kind of bug.
- **`float(array)` crashing on a length-1 array.** Newer NumPy no longer
  auto-converts single-element arrays to scalars — use `array[0]` or
  `.item()` explicitly. Hit this exact bug in `is_bark()` and fixed it.
- **`ModuleNotFoundError: pkg_resources`** — see section 2 above.
- **Disk space errors installing `torch`/`tensorflow`** — if this happens on
  your machine too, uninstall the CUDA-specific `nvidia-*` packages that get
  pulled in unnecessarily for CPU-only use:
  `py -m pip uninstall nvidia-cublas-cu12 nvidia-cudnn-cu12 ... triton`
  (only relevant if you're not using a GPU).

## 8. What I'd do next, in priority order

1. Wire `ml/speaker_id.py` into `wake_word_listener.py`'s live loop (I left it
   as a ready-to-call module rather than forcing the UX decision).
2. Collect bark detector data first — smallest lift, proves the whole
   pipeline end-to-end before investing in the harder identity/emotion data.
3. Turn the manual voiceprint-enrollment snippet in section 5 into a real
   CLI script once Step 2 is trained.
