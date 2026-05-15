import sys
import os
import logging
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import config as cfg

# Set up clean logging structure
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

def load_audio(path, sr=22050): 
    return librosa.load(path, sr=sr)

def apply_noise_gate(y, top_db=30):
    trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    return trimmed

def spectral_subtraction(y, sr):
    stft = librosa.stft(y)
    mag = np.abs(stft)
    phase = np.angle(stft)
    n_noise = max(10, mag.shape[1] // 10)
    noise_est = np.mean(mag[:, :n_noise], axis=1, keepdims=True)
    mag_clean = np.maximum(mag - 1.5 * noise_est, 0.0)
    return librosa.istft(mag_clean * np.exp(1j * phase))

# Data Augmentation definitions
def aug_time_stretch(y, sr): 
    return librosa.effects.time_stretch(y, rate=np.random.uniform(0.85, 1.15))

def aug_pitch_shift(y, sr):  
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=np.random.uniform(-2, 2))

def aug_add_noise(y, sr):    
    return y + 0.005 * np.random.randn(len(y))

def aug_time_shift(y, sr):   
    shift = int(np.random.uniform(-0.2, 0.2) * len(y))
    return np.roll(y, shift)

AUGMENTATIONS = {
    'timestretch': aug_time_stretch, 
    'pitchshift': aug_pitch_shift,
    'noise': aug_add_noise, 
    'timeshift': aug_time_shift
}

def process_one_file(f_path_str, dest_folder_str, sample_rate, top_db):
    f_path = Path(f_path_str)
    dest_folder = Path(dest_folder_str)
    saved = []
    try:
        y, sr = load_audio(f_path_str, sr=sample_rate)
        y = apply_noise_gate(y, top_db=top_db)
        
        if len(y) < sample_rate * 0.5: 
            return f'SKIP (too short):{f_path.name}', 0
            
        y = spectral_subtraction(y, sr)
        y = librosa.util.normalize(y)
        
        # Save original cleaned file
        sf.write(dest_folder / f_path.name, y, sr)
        saved.append(f_path.name)
        
        # Loop through configurations and generate augmentations
        for aug_name, aug_fn in AUGMENTATIONS.items():
            try:
                y_aug = aug_fn(y, sr)
                y_aug = librosa.util.normalize(y_aug)
                aug_path = dest_folder / f'{f_path.stem}_{aug_name}.wav'
                sf.write(aug_path, y_aug, sr)
                saved.append(aug_path.name)
            except Exception as e: 
                logger.warning(f'Aug {aug_name} failed on {f_path.name}: {e}')
    except Exception as e: 
        return f'ERROR {f_path.name}:{e}', 0
        
    return f'OK:{f_path.name}->{len(saved)} files', len(saved)

def run_preprocessing():
    src_root = Path(cfg.DATA_RAW_DIR)
    dst_root = Path(cfg.DATA_PROC_DIR)
    total = 0
    
    if not src_root.exists(): 
        print(f'Error: {src_root} not found. Ensure your data/raw/dataset folder is configured.')
        return
        
    for label in cfg.EMOTION_CLASSES:
        ep = src_root / label
        if not ep.is_dir(): 
            logger.warning(f'Skipping {label}')
            continue
            
        df = dst_root / label
        df.mkdir(parents=True, exist_ok=True)
        files = list(ep.glob('*.wav')) + list(ep.glob('*.mp3'))
        
        logger.info(f'[{label}] Processing {len(files)} raw files -> up to {len(files) * 5} augmented files')
        for f in files:
            msg, n = process_one_file(str(f), str(df), cfg.SAMPLE_RATE, cfg.TOP_DB)
            total += n
            if not msg.startswith('OK'): 
                print(msg)
                
    print(f'\nPipeline 1 complete. Total augmented files written locally: {total}')
    for label in cfg.EMOTION_CLASSES:
        folder = dst_root / label
        count = len(list(folder.glob('*.wav'))) if folder.exists() else 0
        print(f'  {label:<12}:{count} files')

# ── LOCAL RESUME GUARD ──
if __name__ == "__main__":
    proc_root = Path(cfg.DATA_PROC_DIR)
    already_done = proc_root.exists() and any(proc_root.rglob('*.wav'))
    
    if already_done:
        print('Pipeline 1 already complete — skipping processing.')
        print('To force a full rerun, delete your local folder: "data/processed/"')
        for cls in cfg.EMOTION_CLASSES:
            count = len(list((proc_root / cls).glob('*.wav'))) if (proc_root / cls).exists() else 0
            print(f'  {cls:<14}: {count} files')
    else:
        run_preprocessing()