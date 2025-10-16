## Project Overview

This repository contains a lightweight speech-recognition prototype tailored for assistive use-cases, specifically five high-priority Alexa-style commands that support people with cerebral palsy. The system combines speech-content features (MFCCs and chroma) with timing signals (clip duration) to achieve up to 80 % accuracy on a small, curated dataset. The current production model (Random Forest) balances content awareness with timing cues while remaining simple enough to retrain quickly on new data.

**Supported commands:**

1. “Alexa, give me the Ottawa Senators score from last night.”
2. “Alexa please play Elvis Presley’s greatest hits.”
3. “Alexa, call Computerwise.”
4. “Alexa can you give me the weather for tomorrow.”
5. “Alexa, remind me to make my bookings for my events at 8 pm.”

## Requirements

All dependencies are listed in `requirements.txt`. Key packages:

- PyTorch
- librosa
- transformers
- numpy
- scikit-learn
- pyttsx3
- soundfile
- scipy
- noisereduce
- pyloudnorm

Install everything in one step:

```bash
pip install -r requirements.txt
```

## Quick Start

1. Make sure the `output/combined_rf.joblib` and `output/combined_scaler.joblib` artifacts exist. If not, rerun `python train_combined_model.py`.
2. Run inference on a `.wav` file (16 kHz mono) using the combined model:

```bash
python inference_real.py test_data/cleaned_real_3c.wav
```

Expected output: the predicted phrase printed on screen and spoken aloud via `pyttsx3`.

## Model Performance

| Model            | Features                       | Test Accuracy | Notes                       |
|------------------|--------------------------------|---------------|-----------------------------|
| Duration-only    | 1 feature (clip length)        | 100 %         | Not robust to speed changes |
| MFCC-only        | 64 MFCC/chroma features        | 60 %          | Good speech-content signal  |
| Combined (prod)  | 64 MFCC/chroma + duration      | 80 %          | Balanced, production model  |

## Phrases (Strengths & Weaknesses)

- Works well: commands with distinctive wording, especially the longest (“remind me...”) and shortest (“call Computerwise”) clips.
- Struggles occasionally: Class 1 (“Elvis Presley”) can be confused if background music/noise is present; the two-stage model (`train_two_stage.py`) mitigates this.
- Best results when clips last between 3 s and 10 s with minimal background noise.

## Data Structure

| Folder                     | Description                                 |
|----------------------------|---------------------------------------------|
| `cleaned_real_data_v2/`    | Enhanced real recordings (post-processing)  |
| `augmented_real_data_v2/`  | Synthetic variants generated from cleaned data |
| `test_data/`               | Five held-out `.wav` files (one per class)  |
| `output/`                  | Saved model artifacts (RF, MFCC baselines, two-stage, etc.) |

## Training Your Own Model

1. Run the cleaning pipeline (if you add new audio):
   ```bash
   python clean_and_augment_real.py
   ```
   This populates `cleaned_real_data_v2/` and `augmented_real_data_v2/`.
2. Train the production model:
   ```bash
   python train_combined_model.py
   ```
   Training on ~85 samples finishes within a minute on a typical laptop.
3. (Optional) Train baselines or specialized models:
   - `python train_duration_only.py`
   - `python train_mfcc_model.py`
   - `python train_two_stage.py` (two-stage classifier for Class 1)

## Adding New Phrases

1. Record at least 3–5 clean clips per new phrase at 16 kHz. Aim for isolated speech with minimal background noise.
2. Place raw recordings in `real_data/` using a consistent naming scheme (e.g., `real_6a.wav`, `real_6b.wav`).
3. Update `real_phrases.py` with the new phrase mapping.
4. Rerun:
   ```bash
   python clean_and_augment_real.py
   python train_combined_model.py
   ```
   Expect accuracy to degrade slightly until enough recordings are available for the new class.

## Technical Details

- Features: 64 speech-content features (MFCC means/stds, delta MFCC means/stds, chroma means) + 1 duration feature.
- Classifier: RandomForestClassifier with 300 estimators, class-balanced weighting, `random_state=42`.
- Dataset: 15 cleaned recordings (3 per phrase) augmented 5× each → ~85 training samples.
- LUFS normalization, compression, and spectral denoising ensure consistent audio quality.

## Known Limitations

- Class 1 (“Elvis Presley”) remains the trickiest; the two-stage approach helps but longer term, more real recordings would improve robustness.
- Phrases shorter than ~3 s or longer than ~10 s can bounce accuracy.
- Model assumes reasonably clean audio (minimal echo/noise); heavy background interference confuses both the duration and MFCC features.

## Testing

- Evaluate the production model on held-out clips:
  ```bash
  python evaluate_real_model.py
  ```
- Verify speech-content robustness (tempo changes):
  ```bash
  python test_speed_robustness.py
  ```

## Maintenance

Keep `requirements.txt` in sync when adding new dependencies, and remove stale artifacts from `output/` before retraining if you want a clean slate. If accuracy drops, rerun data cleaning and augmentations, then retrain all models for a fresh comparison.

1) Project Overview section:
   - Brief description: Speech recognition for cerebral palsy users
   - 5 Alexa voice commands
   - 80% accuracy using traditional ML (MFCCs + duration)

2) Requirements section:
   - List key dependencies from requirements.txt
   - Installation command: pip install -r requirements.txt

3) Quick Start section:
   - How to test the model on a single audio file
   - Expected output
   - Example command

4) Model Performance section:
   - Table showing all 3 models:
     * Duration-only: 100% (not speed-robust)
     * MFCC-only: 60% (speed-robust)
     * Combined: 80% (production model)

5) The 5 Phrases section:
   - List the 5 Alexa commands
   - Show which phrases work well vs struggle

6) Data Structure section:
   - Explain folder structure:
     * cleaned_real_data_v2/ - cleaned audio
     * augmented_real_data_v2/ - augmented samples
     * test_data/ - held-out test set
     * output/ - trained models

7) Training Your Own Model section:
   - How to retrain if needed
   - Which script to run
   - Expected training time

8) Adding New Phrases section:
   - Requirements: 3-5 recordings per phrase
   - Steps to add new commands
   - Expected accuracy

9) Technical Details section:
   - Feature extraction: 64 MFCCs + 1 duration
   - Model: RandomForest with 300 trees
   - Training data: 85 samples (15 original + augmentation)

10) Known Limitations section:
    - Class 1 ("Elvis Presley") sometimes confused
    - Works best with 3-10 second phrases
    - Requires clean audio input
