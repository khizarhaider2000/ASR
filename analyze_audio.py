"""
Analyze real Alexa recordings to understand data separability.

Metrics per cleaned clip (`cleaned_real_*.wav`):
- Duration
- Mean absolute amplitude (volume proxy)
- Signal energy (mean squared amplitude)
- Wav2Vec2 embedding

Aggregated statistics per phrase plus cosine similarity comparisons allow us to
check whether recordings for the same phrase cluster together and differ across
phrases.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import Wav2Vec2Model, Wav2Vec2Processor

from real_phrases import phrases

CLEAN_DIR = Path("cleaned_real_data")
SAMPLE_RATE = 16_000


def extract_label(path: Path) -> int:
    """Parse label from `cleaned_real_X?.wav` filenames."""
    name = path.name
    try:
        number_part = name.split("_")[2]  # e.g., "1a.wav"
        digit = int(number_part[0])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Could not parse label from '{name}'") from exc
    return digit - 1


def compute_embedding(
    audio: np.ndarray,
    processor: Wav2Vec2Processor,
    wav2vec2: Wav2Vec2Model,
) -> np.ndarray:
    """Return mean-pooled Wav2Vec2 embedding for an audio array."""
    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    with torch.no_grad():
        hidden = wav2vec2(inputs.input_values).last_hidden_state
        embedding = hidden.mean(dim=1).squeeze(0).cpu().numpy()
    return embedding


def analyze() -> None:
    if not CLEAN_DIR.exists():
        raise FileNotFoundError(f"Directory '{CLEAN_DIR}' not found.")

    files = sorted(CLEAN_DIR.glob("cleaned_real_[1-5][abc].wav"))
    if not files:
        raise RuntimeError("No cleaned real audio files found.")

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    wav2vec2.eval()

    import torch

    stats = []
    embeddings = []

    for path in files:
        audio, sr = librosa.load(path, sr=SAMPLE_RATE)
        duration = len(audio) / SAMPLE_RATE
        mean_abs = float(np.mean(np.abs(audio)))
        energy = float(np.mean(np.square(audio)))
        label = extract_label(path)

        emb = compute_embedding(audio, processor, wav2vec2)

        stats.append(
            {
                "path": path,
                "label": label,
                "duration": duration,
                "mean_abs": mean_abs,
                "energy": energy,
            }
        )
        embeddings.append(emb)

    print("Per-file metrics:")
    for item, emb in zip(stats, embeddings):
        print(
            f"- {item['path'].name}: label {item['label']} "
            f"({phrases[item['label']]}) | duration {item['duration']:.2f}s | "
            f"mean|amp| {item['mean_abs']:.4f} | energy {item['energy']:.6f}"
        )
    print()

    # Grouped statistics
    grouped: Dict[int, List[int]] = defaultdict(list)
    for idx, item in enumerate(stats):
        grouped[item["label"]].append(idx)

    print("Average metrics per phrase:")
    for label in sorted(grouped):
        indices = grouped[label]
        durations = np.mean([stats[i]["duration"] for i in indices])
        volumes = np.mean([stats[i]["mean_abs"] for i in indices])
        energies = np.mean([stats[i]["energy"] for i in indices])
        print(
            f"- Label {label} ({phrases[label]}): "
            f"avg duration {durations:.2f}s | avg volume {volumes:.4f} | "
            f"avg energy {energies:.6f}"
        )
    print()

    # Cosine similarities
    embeddings_array = np.stack(embeddings)
    cosine_matrix = cosine_similarity(embeddings_array)

    same_phrase_sims = []
    diff_phrase_sims = []
    for i in range(len(stats)):
        for j in range(i + 1, len(stats)):
            sim = cosine_matrix[i, j]
            if stats[i]["label"] == stats[j]["label"]:
                same_phrase_sims.append(sim)
            else:
                diff_phrase_sims.append(sim)

    if same_phrase_sims and diff_phrase_sims:
        print(
            f"Cosine similarity (same phrase): mean {np.mean(same_phrase_sims):.4f} "
            f"| std {np.std(same_phrase_sims):.4f}"
        )
        print(
            f"Cosine similarity (different phrases): mean {np.mean(diff_phrase_sims):.4f} "
            f"| std {np.std(diff_phrase_sims):.4f}"
        )
    else:
        print("Insufficient data to compute cosine similarity statistics.")
    print()

    # Phrase-level similarity analysis
    print("Average cross-phrase cosine similarities:")
    for label_a in sorted(grouped):
        sims = []
        for label_b in sorted(grouped):
            if label_a == label_b:
                continue
            indices_a = grouped[label_a]
            indices_b = grouped[label_b]
            for i in indices_a:
                for j in indices_b:
                    sims.append(cosine_matrix[i, j])
        if sims:
            print(
                f"- Label {label_a} ({phrases[label_a]}) vs others: "
                f"mean {np.mean(sims):.4f} | min {np.min(sims):.4f} | max {np.max(sims):.4f}"
            )


if __name__ == "__main__":
    analyze()
