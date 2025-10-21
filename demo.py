#!/usr/bin/env python3
"""
Demonstration script showing the end-to-end ASR pipeline:
original audio → feature extraction → model prediction → spoken output.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Tuple

import joblib
import librosa
import numpy as np

from real_phrases import phrases
from train_combined_model import SAMPLE_RATE, compute_content_features

MODEL_PATH = Path("output/combined_rf.joblib")
SCALER_PATH = Path("output/combined_scaler.joblib")
DEFAULT_AUDIO = Path("test_data/cleaned_real_3c.wav")
SEPARATOR = "=" * 60


def require_module(name: str) -> None:
    """Raise a helpful error if an optional dependency is missing."""
    try:
        __import__(name)
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise SystemExit(
            f"Missing optional dependency '{name}'. "
            f"Install it with 'pip install {name}'."
        ) from exc


def load_artifacts() -> Tuple[object, object]:
    """Load the trained random forest model and feature scaler."""
    if not MODEL_PATH.is_file():
        raise SystemExit(
            f"Model artifact not found at '{MODEL_PATH}'. "
            "Train the model first using train_combined_model.py."
        )
    if not SCALER_PATH.is_file():
        raise SystemExit(
            f"Scaler artifact not found at '{SCALER_PATH}'. "
            "Train the model first using train_combined_model.py."
        )
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def prepare_features(audio_path: Path, scaler) -> Tuple[np.ndarray, float, np.ndarray]:
    """Extract and scale features for the provided audio file."""
    if not audio_path.is_file():
        raise SystemExit(f"Audio file '{audio_path}' not found.")

    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
    content = compute_content_features(audio, SAMPLE_RATE)
    duration = librosa.get_duration(y=audio, sr=SAMPLE_RATE)
    feature_vector = np.concatenate([content, [duration]], axis=0)
    scaled = scaler.transform([feature_vector])[0]
    return feature_vector, duration, scaled


def play_audio(audio_path: Path) -> None:
    """Play an audio clip using pygame."""
    require_module("pygame")
    import pygame

    pygame.mixer.init()
    try:
        sound = pygame.mixer.Sound(str(audio_path))
        channel = sound.play()
        while channel.get_busy():
            time.sleep(0.1)
    finally:
        pygame.mixer.quit()


def speak_text(text: str, rate: int = 175, volume: float = 1.0) -> None:
    """Speak the predicted phrase aloud using gTTS and pygame playback."""
    require_module("gtts")
    from gtts import gTTS

    temp_path = Path("temp_tts.mp3")
    try:
        print(f"Speaking: {text}")
        tts = gTTS(text=text, lang="en")
        tts.save(temp_path)
        play_audio(temp_path)
    except Exception as exc:
        print(f"TTS Error: {exc}")
        print("Text-to-speech failed, but prediction was successful.")
    finally:
        if temp_path.exists():
            temp_path.unlink()


def predict_phrase(model, scaled_features: np.ndarray) -> Tuple[int, float]:
    """Return the predicted label and confidence percentage."""
    probabilities = model.predict_proba([scaled_features])[0]
    label = int(np.argmax(probabilities))
    confidence = float(probabilities[label]) * 100.0
    return label, confidence


def display_banner(audio_path: Path) -> None:
    """Print a formatted banner for clarity."""
    print(SEPARATOR)
    print("Combined Model Demo")
    print(SEPARATOR)
    print(f"Audio file: {audio_path}")
    print(SEPARATOR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a live ASR demo.")
    parser.add_argument(
        "audio_path",
        nargs="?",
        default=str(DEFAULT_AUDIO),
        help="Path to a WAV file to evaluate (default: %(default)s)",
    )
    args = parser.parse_args()
    audio_path = Path(args.audio_path)

    display_banner(audio_path)

    model, scaler = load_artifacts()

    print("Playing original audio...")
    time.sleep(0.5)
    play_audio(audio_path)

    print("\nProcessing through model...")
    feature_vector, duration, scaled = prepare_features(audio_path, scaler)
    label, confidence = predict_phrase(model, scaled)
    phrase = phrases.get(label, f"[Unknown label {label}]")

    print(f"Duration: {duration:.2f}s")
    print(f"Feature vector length: {len(feature_vector)} (64 content + duration)")
    print(f"Model prediction: {phrase}")
    print(f"Confidence: {confidence:.2f}%")

    print("\nSpeaking prediction...")
    time.sleep(0.5)
    classification = phrase
    print(f"About to speak: '{classification}'")
    speak_text(classification, rate=175, volume=1.0)

    print("\nDemo complete!")
    print(SEPARATOR)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
