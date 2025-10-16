"""
Baseline classifier using only audio durations to distinguish Alexa phrases.

Steps:
1. Collect durations (in seconds) for every training clip from
   `cleaned_real_data/` (suffixes a/b) and `augmented_real_data/`.
2. Extract labels (0-4) from filenames.
3. Train/test split (70/30 stratified) to form training and validation sets.
4. Fit an sklearn classifier (logistic regression) using the single duration feature.
5. Evaluate on validation and the held-out real test clips in `test_data/`.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from real_phrases import phrases

# Directories
CLEAN_DIR = Path("cleaned_real_data_v2")
AUG_DIR = Path("augmented_real_data_v2")
TEST_DIR = Path("test_data")

# File patterns
CLEAN_PATTERN = "cleaned_real_[1-5][ab].wav"
AUG_PATTERN = "*.wav"
TEST_FILES = [
    "cleaned_real_1c.wav",
    "cleaned_real_2c.wav",
    "cleaned_real_3c.wav",
    "cleaned_real_4c.wav",
    "cleaned_real_5c.wav",
]


def extract_label(name: str) -> int:
    """Parse label from filename `cleaned_real_X?.wav`."""
    try:
        number_part = name.split("_")[2]  # e.g., "1a.wav"
        digit = int(number_part[0])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Could not parse label from '{name}'") from exc
    label = digit - 1
    if label not in phrases:
        raise ValueError(f"Label {label} out of range for file '{name}'.")
    return label


def collect_durations(paths: List[Path]) -> Tuple[np.ndarray, np.ndarray]:
    """Return durations (seconds) and labels for provided paths."""
    features = []
    labels = []
    for path in paths:
        duration = librosa.get_duration(path=path, sr=None)
        label = extract_label(path.name)
        features.append([duration])
        labels.append(label)
    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int64)


def load_training_data() -> Tuple[np.ndarray, np.ndarray]:
    """Load durations/labels for cleaned (a/b) + augmented data."""
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

    all_paths = cleaned_paths + augmented_paths
    features, labels = collect_durations(all_paths)
    return features, labels


def load_test_data() -> Tuple[np.ndarray, np.ndarray, List[Path]]:
    """Load durations/labels for held-out test files."""
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Directory '{TEST_DIR}' not found.")

    test_paths = []
    for name in TEST_FILES:
        path = TEST_DIR / name
        if not path.is_file():
            raise FileNotFoundError(f"Test file '{path}' not found.")
        test_paths.append(path)

    features, labels = collect_durations(test_paths)
    return features, labels, test_paths


def main() -> None:
    features, labels = load_training_data()
    X_train, X_val, y_train, y_val = train_test_split(
        features,
        labels,
        test_size=0.30,
        random_state=42,
        stratify=labels,
    )

    clf = LogisticRegression(max_iter=1000, multi_class="multinomial")
    clf.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, clf.predict(X_train))
    val_acc = accuracy_score(y_val, clf.predict(X_val))

    X_test, y_test, test_paths = load_test_data()
    test_preds = clf.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)

    print("Duration-only baseline results")
    print(f"Training accuracy:   {train_acc:.2%}")
    print(f"Validation accuracy: {val_acc:.2%}")
    print(f"Test accuracy:       {test_acc:.2%}\n")

    print("Test set predictions:")
    for path, label, pred in zip(test_paths, y_test, test_preds):
        verdict = "✓" if label == pred else "✗"
        print(f"{verdict} {path.name}")
        print(f"    True: {phrases[label]}")
        print(f"    Pred: {phrases[pred]}")


if __name__ == "__main__":
    main()
