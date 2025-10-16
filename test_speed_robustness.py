"""
Test robustness of the MFCC-based model to tempo changes.

Loads the trained model artifacts produced by `train_mfcc_model.py`, then
evaluates the classifier on time-stretched versions of `cleaned_real_3c.wav`
to confirm that predictions remain stable when speaking speed varies.
"""
from __future__ import annotations

import joblib
import librosa
import numpy as np
import torch
from pathlib import Path

from real_phrases import phrases

SAMPLE_RATE = 16_000
TARGET_FILE = Path("test_data/cleaned_real_3c.wav")
MODEL_PATH = Path("output/mfcc_content_rf.joblib")
SCALER_PATH = Path("output/mfcc_scaler.joblib")
SPEED_FACTORS = {
    "1.0x": 1.0,
    "0.9x": 0.9,
    "1.1x": 1.1,
}


def load_artifacts():
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Expected trained model artifacts in 'output/': "
            f"{MODEL_PATH.name} and {SCALER_PATH.name}. "
            "Run train_mfcc_model.py first."
        )
    clf = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return clf, scaler


def extract_features(audio: np.ndarray, sr: int) -> np.ndarray:
    """Replicate feature extraction from train_mfcc_model.py."""
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)

    delta = librosa.feature.delta(mfcc)
    delta_mean = delta.mean(axis=1)
    delta_std = delta.std(axis=1)

    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    chroma_mean = chroma.mean(axis=1)

    features = np.concatenate(
        [mfcc_mean, mfcc_std, delta_mean, delta_std, chroma_mean],
        axis=0,
    )
    return features.astype(np.float32)


def main() -> None:
    if not TARGET_FILE.exists():
        raise FileNotFoundError(f"Target file '{TARGET_FILE}' not found.")

    clf, scaler = load_artifacts()

    base_audio, _ = librosa.load(TARGET_FILE, sr=SAMPLE_RATE)

    print("Speed robustness test for cleaned_real_3c.wav")
    true_label = 2  # cleaned_real_3c -> class 2
    print(f"True phrase: {phrases[true_label]}\n")

    for label, factor in SPEED_FACTORS.items():
        if factor == 1.0:
            audio = base_audio
        else:
            audio = librosa.effects.time_stretch(base_audio, rate=factor)

        duration = len(audio) / SAMPLE_RATE
        features = extract_features(audio, SAMPLE_RATE)
        features_scaled = scaler.transform(features.reshape(1, -1))
        pred = clf.predict(features_scaled)[0]

        verdict = "✅" if pred == true_label else "❌"
        print(f"{verdict} Speed {label}: predicted '{phrases[pred]}' (duration {duration:.2f}s)")


if __name__ == "__main__":
    main()
