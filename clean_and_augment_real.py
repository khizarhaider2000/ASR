
"""
Clean and augment real audio recordings with enhanced speech-focused processing.

Pipeline:
1. Load each WAV file from `real_data/`.
2. Remove leading/trailing silence aggressively.
3. Apply high-pass (80 Hz) and low-pass (8 kHz) filters to remove rumble and hiss.
4. Perform spectral noise reduction with an expanded noise profile.
5. Boost intelligibility in the 300 Hz – 3 kHz band.
6. Compress dynamics and normalize to -20 LUFS for consistent loudness.
7. Save enhanced clips to `cleaned_real_data_v2/`.
8. Apply the original aggressive augmentation strategy and write results to
   `augmented_real_data_v2/`.
"""
import os
from pathlib import Path
from typing import Iterable, Tuple

import pyloudnorm as pyln
import librosa
import noisereduce as nr
import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt

# Directories
RAW_DIR = Path("real_data")
CLEAN_DIR = Path("cleaned_real_data_v2")
AUG_DIR = Path("augmented_real_data_v2")

# Audio processing constants
SAMPLE_RATE = 16_000
TARGET_RMS_DB = -20.0
BANDPASS_RANGE = (80.0, 8_000.0)  # Hz
BANDPASS_ORDER = 4
SPEECH_BAND = (300.0, 3_000.0)
TRIM_TOP_DB = 35
COMP_THRESHOLD_DB = -18.0
COMP_RATIO = 4.0

# Augmentation settings
AUGMENT_TIMES = 5
PITCH_SHIFT_RANGE = (-2.0, 2.0)  # semitones
TIME_STRETCH_RANGE = (0.9, 1.1)
NOISE_STD = 0.003


def ensure_directories() -> None:
    """Create output directories when they do not exist."""
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    AUG_DIR.mkdir(parents=True, exist_ok=True)


def list_audio_files(directory: Path) -> Iterable[Path]:
    """Yield all `.wav` files from the given directory sorted by name."""
    return sorted(f for f in directory.glob("*.wav") if f.is_file())


def butter_filter(audio: np.ndarray, sr: int, cutoff: float, btype: str, order: int = 4) -> np.ndarray:
    """Apply a Butterworth high/low-pass filter."""
    nyquist = sr / 2.0
    normalized = cutoff / nyquist
    if btype == "low":
        normalized = min(normalized, 0.9999)
    else:
        normalized = max(normalized, 1e-4)
    b, a = butter(order, normalized, btype=btype)
    return filtfilt(b, a, audio).astype(np.float32)


def bandpass_filter(audio: np.ndarray, sr: int, freq_range: Tuple[float, float], order: int = 4) -> np.ndarray:
    """Apply a Butterworth bandpass filter."""
    nyquist = sr / 2.0
    low = max(freq_range[0] / nyquist, 1e-4)
    high = min(freq_range[1] / nyquist, 0.9999)
    if not 0 < low < high < 1:
        raise ValueError(f"Invalid bandpass range: {freq_range}")

    b, a = butter(order, [low, high], btype="band")
    filtered = filtfilt(b, a, audio)
    return filtered.astype(np.float32)


def trim_silence(audio: np.ndarray) -> np.ndarray:
    """Remove leading and trailing silence."""
    trimmed, _ = librosa.effects.trim(audio, top_db=TRIM_TOP_DB)
    if trimmed.size == 0:
        return audio
    return trimmed


def estimate_noise_profile(audio: np.ndarray, sr: int) -> np.ndarray:
    """Extract a noise profile from low-energy segments."""
    # Use the first half-second plus quiet frames detected via RMS.
    noise_head = audio[: int(0.5 * sr)]
    frame_length = int(0.05 * sr)
    hop_length = max(frame_length // 2, 1)
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length).flatten()
    if rms.size == 0:
        return noise_head if noise_head.size else audio[: sr]

    threshold = np.percentile(rms, 25)
    quiet_segments = []
    for idx, value in enumerate(rms):
        if value <= threshold:
            start = idx * hop_length
            end = min(start + frame_length, audio.size)
            quiet_segments.append(audio[start:end])

    if quiet_segments:
        noise_profile = np.concatenate([noise_head, *quiet_segments])
    else:
        noise_profile = noise_head

    if noise_profile.size == 0:
        noise_profile = audio[: sr]
    return noise_profile


def reduce_noise(audio: np.ndarray, sr: int) -> np.ndarray:
    """Apply aggressive spectral noise reduction using an expanded noise profile."""
    noise_profile = estimate_noise_profile(audio, sr)
    cleaned = nr.reduce_noise(
        y=audio,
        y_noise=noise_profile,
        sr=sr,
        stationary=False,
        prop_decrease=1.0,
        time_mask_smooth_ms=128,
        freq_mask_smooth_hz=200,
        n_std_thresh_stationary=1.5,
    )
    return cleaned.astype(np.float32)


def apply_compression(audio: np.ndarray, threshold_db: float, ratio: float) -> np.ndarray:
    """Apply a simple soft-knee compression above the threshold."""
    threshold_linear = 10 ** (threshold_db / 20.0)
    compressed = audio.copy()
    mask = np.abs(compressed) > threshold_linear
    compressed[mask] = np.sign(compressed[mask]) * (
        threshold_linear + (np.abs(compressed[mask]) - threshold_linear) / ratio
    )
    return compressed.astype(np.float32)


def normalize_loudness(audio: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
    """Normalize audio to the target LUFS level."""
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio)
    normalized = pyln.normalize.loudness(audio, loudness, target_lufs)
    peak = np.max(np.abs(normalized))
    if peak > 1.0:
        normalized = normalized / peak
    return normalized.astype(np.float32)


def enhance_speech(audio: np.ndarray, sr: int) -> np.ndarray:
    """Boost energy in the speech band to aid intelligibility."""
    speech_band = bandpass_filter(audio, sr, SPEECH_BAND, order=2)
    enhanced = audio + 0.3 * speech_band
    peak = np.max(np.abs(enhanced))
    if peak > 1.0:
        enhanced = enhanced / peak
    return enhanced.astype(np.float32)


def augment_audio(audio: np.ndarray, sr: int) -> np.ndarray:
    """Apply random pitch, speed, and noise augmentations to an audio clip."""
    # Pitch shift
    n_steps = np.random.uniform(*PITCH_SHIFT_RANGE)
    augmented = librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=n_steps)

    # Time stretch
    rate = np.random.uniform(*TIME_STRETCH_RANGE)
    augmented = librosa.effects.time_stretch(augmented, rate=rate)

    # Additive Gaussian noise
    noise = np.random.normal(scale=NOISE_STD, size=len(augmented))
    augmented = augmented + noise

    return normalize_loudness(augmented, sr, TARGET_RMS_DB)


def process_file(path: Path) -> Path:
    """Clean a single audio file and save it to the cleaned directory."""
    audio, sr = librosa.load(path, sr=SAMPLE_RATE)

    audio = trim_silence(audio)
    if audio.size == 0:
        audio = np.zeros(1, dtype=np.float32)

    audio = butter_filter(audio, sr, cutoff=BANDPASS_RANGE[0], btype="high", order=4)
    audio = butter_filter(audio, sr, cutoff=BANDPASS_RANGE[1], btype="low", order=4)
    audio = reduce_noise(audio, sr)
    audio = enhance_speech(audio, sr)
    audio = apply_compression(audio, COMP_THRESHOLD_DB, COMP_RATIO)
    cleaned = normalize_loudness(audio, sr, TARGET_RMS_DB)

    cleaned_name = f"cleaned_{path.name}"
    cleaned_path = CLEAN_DIR / cleaned_name
    sf.write(cleaned_path, cleaned, sr)

    return cleaned_path


def augment_cleaned_file(path: Path) -> None:
    """Generate augmented versions from a cleaned audio file."""
    audio, sr = librosa.load(path, sr=SAMPLE_RATE)
    stem = path.stem  # e.g., cleaned_real_1a

    for idx in range(AUGMENT_TIMES):
        augmented = augment_audio(audio, sr)
        out_name = f"{stem}_aug{idx}.wav"
        out_path = AUG_DIR / out_name
        sf.write(out_path, augmented, sr)


def main() -> None:
    ensure_directories()

    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Input directory '{RAW_DIR}' does not exist.")

    raw_files = list(list_audio_files(RAW_DIR))
    if not raw_files:
        raise RuntimeError(f"No WAV files found in '{RAW_DIR}'.")

    cleaned_paths = []
    for wav_file in raw_files:
        print(f"Cleaning {wav_file.name}...")
        cleaned_path = process_file(wav_file)
        cleaned_paths.append(cleaned_path)

    for cleaned_file in cleaned_paths:
        print(f"Augmenting {cleaned_file.name}...")
        augment_cleaned_file(cleaned_file)

    print(f"Finished cleaning {len(cleaned_paths)} files.")
    print(f"Generated {len(cleaned_paths) * AUGMENT_TIMES} augmented clips in '{AUG_DIR}'.")


if __name__ == "__main__":
    main()
