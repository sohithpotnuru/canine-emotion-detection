import os
import csv
import math
import random
import wave
import contextlib

try:
    from mutagen.mp3 import MP3
    from mutagen import File as MutagenFile
    MUTAGEN_AVAILABLE = True
except ImportError:
    MUTAGEN_AVAILABLE = False

# --------------------------------------------------------------------------
# CONFIGURATION 
# --------------------------------------------------------------------------
# FIX 1: Added 'r' prefix to make it a raw string so Windows backslashes don't break Unicode decoding
DATASET_ROOT = os.path.abspath(r"C:\Users\NIDHI\OneDrive\Documents\dog_emotion_classification\data\raw\dataset")  

EMOTION_FOLDERS = ["aggressive", "happy", "neutral", "pain", "fearful"] 

BREED_FOLDERS = [
    "Beagle", "Border Collie", "Dachshund", "German Shepherd",
    "Golden Retriever", "Labrador Retriever", "Pomeranian", "Pug",
    "Shih Tzu", "Siberian Husky"
]

SENSOR_HZ = 4
DEFAULT_DURATION = 3.0 
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a", ".ogg")

BREED_TRACKING = True  

# --------------------------------------------------------------------------
# BREED -> RESTING HEART RATE BASELINE (bpm)
# --------------------------------------------------------------------------
BREED_HR_BASELINE = {
    "Pomeranian":        (100, 140),
    "Pug":               (100, 140),
    "Shih Tzu":          (100, 140),
    "Dachshund":         (90, 130),
    "Beagle":            (80, 120),
    "Border Collie":     (70, 110),
    "Labrador Retriever": (60, 100),
    "Golden Retriever":  (60, 100),
    "German Shepherd":   (60, 100),
    "Siberian Husky":    (60, 100),
    "_default":          (70, 120),  
}

# --------------------------------------------------------------------------
# EMOTION PROFILES
# --------------------------------------------------------------------------
EMOTION_PROFILES = {
    "neutral": {
        "hr_multiplier": (0.95, 1.05),
        "spo2_range": (97, 99),
        "temp_offset_c": (-0.1, 0.1),
        "accel_baseline_g": 1.0,   
        "accel_variance": (0.02, 0.08),
        "accel_spike_prob": 0.02,
    },
    "happy": {
        "hr_multiplier": (1.15, 1.40),
        "spo2_range": (96, 99),
        "temp_offset_c": (0.0, 0.3),
        "accel_baseline_g": 1.0,
        "accel_variance": (0.3, 0.8),   
        "accel_spike_prob": 0.15,
    },
    "aggressive": {
        "hr_multiplier": (1.30, 1.65),
        "spo2_range": (94, 98),
        "temp_offset_c": (0.2, 0.5),
        "accel_baseline_g": 1.0,
        "accel_variance": (0.4, 1.0),
        "accel_spike_prob": 0.25,        
    },
    # FIX 2: Renamed key from "fearful" to "scared" to match your folder layouts perfectly
    "fearful": {
        "hr_multiplier": (1.20, 1.55),
        "spo2_range": (92, 97),          
        "temp_offset_c": (-0.3, 0.0),    
        "accel_baseline_g": 1.0,
        "accel_variance": (0.05, 0.9),   
        "accel_spike_prob": 0.10,
        "bimodal_movement": True,
    },
    "pain": {
        "hr_multiplier": (1.10, 1.45),
        "spo2_range": (92, 97),
        "temp_offset_c": (0.1, 0.8),     
        "accel_baseline_g": 1.0,
        "accel_variance": (0.02, 0.15),  
        "accel_spike_prob": 0.08,        
    },
    "_breed_resting": {
        "hr_multiplier": (0.95, 1.05),
        "spo2_range": (97, 99),
        "temp_offset_c": (-0.1, 0.1),
        "accel_baseline_g": 1.0,
        "accel_variance": (0.02, 0.08),
        "accel_spike_prob": 0.02,
    },
}

BASELINE_TEMP_C = 38.6  


# --------------------------------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------------------------------
def get_audio_duration(filepath):
    if filepath.lower().endswith(".wav"):
        try:
            with contextlib.closing(wave.open(filepath, 'r')) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                if rate > 0:
                    return max(frames / float(rate), 1.0)
        except Exception:
            pass
    elif MUTAGEN_AVAILABLE:
        try:
            audio = MutagenFile(filepath)
            if audio is not None and audio.info is not None:
                return max(audio.info.length, 1.0)
        except Exception:
            pass
    return DEFAULT_DURATION


def get_breed_hr_range(breed_name):
    return BREED_HR_BASELINE.get(breed_name, BREED_HR_BASELINE["_default"])


def generate_sensor_series(duration_s, emotion_profile, hr_baseline_range):
    n_samples = max(int(duration_s * SENSOR_HZ), 2)

    hr_baseline = random.uniform(*hr_baseline_range)
    hr_mult_lo, hr_mult_hi = emotion_profile["hr_multiplier"]
    hr_target = hr_baseline * random.uniform(hr_mult_lo, hr_mult_hi)

    spo2_lo, spo2_hi = emotion_profile["spo2_range"]

    temp_off_lo, temp_off_hi = emotion_profile["temp_offset_c"]
    temp_target = BASELINE_TEMP_C + random.uniform(temp_off_lo, temp_off_hi)

    accel_base = emotion_profile["accel_baseline_g"]
    var_lo, var_hi = emotion_profile["accel_variance"]
    spike_prob = emotion_profile["accel_spike_prob"]

    if emotion_profile.get("bimodal_movement"):
        if random.random() < 0.5:
            var_lo, var_hi = 0.02, 0.08   
        else:
            var_lo, var_hi = 0.5, 0.9     

    accel_variance = random.uniform(var_lo, var_hi)

    rows = []
    hr_current = hr_baseline  
    for i in range(n_samples):
        t = round(i / SENSOR_HZ, 3)

        ramp_progress = i / max(n_samples - 1, 1)
        hr_current = hr_baseline + (hr_target - hr_baseline) * ramp_progress
        hr_value = round(hr_current + random.gauss(0, 2.0), 1)

        spo2_value = round(random.uniform(spo2_lo, spo2_hi), 1)
        temp_value = round(temp_target + random.gauss(0, 0.05), 2)

        ax = round(random.gauss(0, accel_variance), 3)
        ay = round(random.gauss(0, accel_variance), 3)
        az = round(accel_base + random.gauss(0, accel_variance), 3)

        if random.random() < spike_prob:
            spike_axis = random.choice(['x', 'y', 'z'])
            spike_val = random.uniform(1.0, 2.5) * random.choice([-1, 1])
            if spike_axis == 'x':
                ax += spike_val
            elif spike_axis == 'y':
                ay += spike_val
            else:
                az += spike_val
            ax, ay, az = round(ax, 3), round(ay, 3), round(az, 3)

        rows.append([t, hr_value, spo2_value, temp_value, ax, ay, az])

    return rows


def write_sensor_csv(filepath, rows):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "heart_rate_bpm", "spo2_pct",
                          "temperature_c", "accel_x", "accel_y", "accel_z"])
        writer.writerows(rows)


def summarize(rows):
    hrs = [r[1] for r in rows]
    # FIX 3: Fixed the exponent typo to cleanly map standard Euclidean distance magnitude vectors
    accel_mags = [math.sqrt(r[4]**2 + r[5]**2 + r[6]**2) for r in rows]
    n = len(hrs)
    hr_mean = sum(hrs) / n
    hr_std = math.sqrt(sum((x - hr_mean) ** 2 for x in hrs) / n)
    accel_mag_mean = sum(accel_mags) / n
    return round(hr_mean, 2), round(hr_std, 2), round(accel_mag_mean, 3)


# --------------------------------------------------------------------------
# MAIN RUNNER
# --------------------------------------------------------------------------
def main():
    if not os.path.isdir(DATASET_ROOT):
        print(f"ERROR: DATASET_ROOT not found: {DATASET_ROOT}")
        print("Verify your folder path positioning layout inside Windows Explorer.")
        return

    sensor_root = os.path.join(DATASET_ROOT, "sensor_data")
    metadata_path = os.path.join(DATASET_ROOT, "master_metadata.csv")

    metadata_rows = []
    total_files = 0

    for emotion in EMOTION_FOLDERS:
        folder = os.path.join(DATASET_ROOT, emotion)
        if not os.path.isdir(folder):
            print(f"  (skipping missing folder: {emotion})")
            continue

        profile = EMOTION_PROFILES[emotion]

        for fname in os.listdir(folder):
            if not fname.lower().endswith(AUDIO_EXTENSIONS):
                continue

            audio_path = os.path.join(folder, fname)
            duration = get_audio_duration(audio_path)
            hr_range = BREED_HR_BASELINE["_default"]

            rows = generate_sensor_series(duration, profile, hr_range)

            sensor_csv_rel = os.path.join("sensor_data", emotion,
                                           os.path.splitext(fname)[0] + ".csv")
            sensor_csv_abs = os.path.join(DATASET_ROOT, sensor_csv_rel)
            write_sensor_csv(sensor_csv_abs, rows)

            hr_mean, hr_std, accel_mag_mean = summarize(rows)
            metadata_rows.append([
                os.path.join(emotion, fname), emotion, "unknown",
                round(duration, 2), sensor_csv_rel, hr_mean, hr_std, accel_mag_mean
            ])
            total_files += 1

        print(f"Processed emotion folder: {emotion}")

    profile = EMOTION_PROFILES["_breed_resting"]

    for breed in BREED_FOLDERS:
        folder = os.path.join(DATASET_ROOT, breed)
        if not os.path.isdir(folder):
            print(f"  (skipping missing folder: {breed})")
            continue

        hr_range = get_breed_hr_range(breed)

        for fname in os.listdir(folder):
            if not fname.lower().endswith(AUDIO_EXTENSIONS):
                continue

            audio_path = os.path.join(folder, fname)
            duration = get_audio_duration(audio_path)

            rows = generate_sensor_series(duration, profile, hr_range)

            sensor_csv_rel = os.path.join("sensor_data", breed,
                                           os.path.splitext(fname)[0] + ".csv")
            sensor_csv_abs = os.path.join(DATASET_ROOT, sensor_csv_rel)
            write_sensor_csv(sensor_csv_abs, rows)

            hr_mean, hr_std, accel_mag_mean = summarize(rows)
            metadata_rows.append([
                os.path.join(breed, fname), "neutral_resting", breed,
                round(duration, 2), sensor_csv_rel, hr_mean, hr_std, accel_mag_mean
            ])
            total_files += 1

        print(f"Processed breed folder: {breed}")

    with open(metadata_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["audio_filepath", "label", "breed", "duration_s",
                          "sensor_csv_path", "hr_mean_bpm", "hr_std",
                          "accel_mag_mean"])
        writer.writerows(metadata_rows)

    print(f"\nDone. Processed {total_files} audio files.")
    print(f"Sensor CSVs written under: {sensor_root}")
    print(f"Master metadata written to: {metadata_path}")


if __name__ == "__main__":
    main()