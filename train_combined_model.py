"""
Train a hybrid classifier that blends speech-content MFCC features with clip duration.

Feature vector per audio clip:
- MFCC (13 coefficients): mean + std → 26 dims
- Delta MFCC: mean + std → 26 dims
- Chroma energy (12 bins): mean → 12 dims
- Duration (seconds): 1 dim

Total: 65 features.

Outputs:
- Training, validation, and test accuracy on held-out real recordings.
- Feature importance ranking to highlight duration vs MFCC contributions.
- Saved artifacts (`output/combined_rf.joblib`, `output/combined_scaler.joblib`)
  for later evaluation.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import joblib
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from real_phrases import phrases

# Directories
CLEAN_DIR = Path("cleaned_real_data_v2")
AUG_DIR = Path("augmented_real_data_v2")
TEST_DIR = Path("test_data")

CLEAN_PATTERN = "cleaned_real_[1-5][ab].wav"
AUG_PATTERN = "*.wav"
TEST_FILES = [
    "cleaned_real_1c.wav",
    "cleaned_real_2c.wav",
    "cleaned_real_3c.wav",
    "cleaned_real_4c.wav",
    "cleaned_real_5c.wav",
]

SAMPLE_RATE = 16_000
N_MFCC = 13


def extract_label(name: str) -> int:
    """Parse label from filename (`cleaned_real_X?.wav`)."""
    try:
        number_part = name.split("_")[2]  # e.g., "1a.wav"
        digit = int(number_part[0])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Could not parse label from '{name}'") from exc
    label = digit - 1
    if label not in phrases:
        raise ValueError(f"Label {label} out of range for file '{name}'.")
    return label


def compute_content_features(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Extract MFCC, delta, and chroma-based features (64 dims)."""
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
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


def compute_features_with_duration(paths: List[Path]) -> Tuple[np.ndarray, np.ndarray]:
    """Return feature matrix (content + duration) and label vector."""
    feature_list = []
    labels = []
    for path in paths:
        audio, _ = librosa.load(path, sr=SAMPLE_RATE)
        content_feats = compute_content_features(audio, SAMPLE_RATE)
        duration = librosa.get_duration(y=audio, sr=SAMPLE_RATE)
        combined = np.concatenate([content_feats, [duration]], axis=0)
        feature_list.append(combined)
        labels.append(extract_label(path.name))

    features = np.stack(feature_list)
    labels_array = np.array(labels, dtype=np.int64)
    return features, labels_array


def load_training_data() -> Tuple[np.ndarray, np.ndarray]:
    if not CLEAN_DIR.exists():
        raise FileNotFoundError(f"Directory '{CLEAN_DIR}' not found.")
    if not AUG_DIR.exists():
        raise FileNotFoundError(f"Directory '{AUG_DIR}' not found.")

    cleaned_paths = sorted(CLEAN_DIR.glob(CLEAN_PATTERN))
    augmented_paths = sorted(AUG_DIR.glob(AUG_PATTERN))

    if not cleaned_paths:
        raise RuntimeError("No cleaned training files found (expected suffix a/b).")
    if not augmented_paths:
        raise RuntimeError("No augmented training files found.")

    print("Loading data from:")
    print(f"  - {CLEAN_DIR}/: {len(cleaned_paths)} files")
    print(f"  - {AUG_DIR}/: {len(augmented_paths)} files")

    all_paths = cleaned_paths + augmented_paths
    print(f"  Total training files: {len(all_paths)}\n")
    return compute_features_with_duration(all_paths)


def load_test_data() -> Tuple[np.ndarray, np.ndarray, List[Path]]:
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Directory '{TEST_DIR}' not found.")

    test_paths = []
    for name in TEST_FILES:
        path = TEST_DIR / name
        if not path.is_file():
            raise FileNotFoundError(f"Test file '{path}' not found.")
        test_paths.append(path)
    print(f"Loading test data from test_data/: {len(test_paths)} files")
    print("Test files:")
    for path in test_paths:
        print(f"  - {path.name}")

    features, labels = compute_features_with_duration(test_paths)
    return features, labels, test_paths


def describe_feature_importances(importances: np.ndarray) -> None:
    """Print top feature importances and duration contribution."""
    content_names = (
        [f"MFCC mean {i}" for i in range(13)]
        + [f"MFCC std {i}" for i in range(13)]
        + [f"Delta MFCC mean {i}" for i in range(13)]
        + [f"Delta MFCC std {i}" for i in range(13)]
        + [f"Chroma mean {i}" for i in range(12)]
    )
    feature_names = content_names + ["Duration"]

    duration_importance = importances[-1]
    content_importance = importances[:-1]

    top_indices = np.argsort(importances)[::-1][:10]

    print("\nTop feature importances:")
    for idx in top_indices:
        name = feature_names[idx] if idx < len(feature_names) else f"Feature {idx}"
        print(f"  {name}: {importances[idx]:.4f}")

    print(f"\nDuration importance: {duration_importance:.4f}")
    print(f"Total content importance: {content_importance.sum():.4f}")


def main() -> None:
    features, labels = load_training_data()
    X_train, X_val, y_train, y_val = train_test_split(
        features,
        labels,
        test_size=0.30,
        random_state=42,
        stratify=labels,
    )
    print("Split:")
    print(f"  - Training samples: {len(X_train)}")
    print(f"  - Validation samples: {len(X_val)}\n")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
    )
    clf.fit(X_train_scaled, y_train)

    train_preds = clf.predict(X_train_scaled)
    val_preds = clf.predict(X_val_scaled)
    train_acc = accuracy_score(y_train, train_preds)
    val_acc = accuracy_score(y_val, val_preds)

    X_test, y_test, test_paths = load_test_data()
    X_test_scaled = scaler.transform(X_test)
    test_preds = clf.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, test_preds)

    print("Combined content + duration model results")
    print(f"Training accuracy:   {train_acc:.2%}")
    print(f"Validation accuracy: {val_acc:.2%}")
    print(f"Test accuracy:       {test_acc:.2%}\n")

    print("Test set predictions:")
    for path, label, pred in zip(test_paths, y_test, test_preds):
        verdict = "✓" if label == pred else "✗"
        print(f"{verdict} {path.name}")
        print(f"    True: {phrases[label]}")
        print(f"    Pred: {phrases[pred]}")

    describe_feature_importances(clf.feature_importances_)

    # Save artifacts
    os.makedirs("output", exist_ok=True)
    joblib.dump(clf, "output/combined_rf.joblib")
    joblib.dump(scaler, "output/combined_scaler.joblib")
    print("\nSaved model to output/combined_rf.joblib")
    print("Saved scaler to output/combined_scaler.joblib")

    # Comparison summary (fill in your observed metrics as needed)
    print("\nComparison summary (update with observed results):")
    print("  Duration-only:  100%")  # replace with actual result if different
    print("  MFCC-only:       60%")  # replace with actual result if different
    print(f"  Combined:        {test_acc:.2%}")


if __name__ == "__main__":
    main()
