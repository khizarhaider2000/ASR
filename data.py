import pyttsx3
import os
import librosa
import soundfile as sf
import numpy as np

# phrase config
phrases = [
    "I need water","I'm hungry","Please help me","Yes","No","Thank you","I'm tired",
    "I want to go outside","Stop","Go","More","Less","I need the bathroom",
    "I don't understand","Wait","Come here","Good morning","Good night","I'm happy",
    "I'm sad","I need medicine","Call someone","I feel sick","I like this","I don't like this"
]

output_dir = "fake_training_data"
os.makedirs(output_dir, exist_ok=True)

speech_rates = [150, 180]  # limit to 2 rates
augment_times = 5          # number of augmentations per base file
tts_voices_to_use = 4      # number of voices to use (from system list)

# initialize TTS
engine = pyttsx3.init()
all_voices = engine.getProperty('voices')
selected_voices = all_voices[:tts_voices_to_use]  # pick first N voices


def augment_audio(file_path, out_path):
    # Load audio
    y, sr = librosa.load(file_path, sr=16000)
    
    # Random pitch shift between -2 and +2 semitones (keyword args for librosa v0.10+)
    y_shifted = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=np.random.uniform(-2, 2))
    
    # Random speed change 0.9–1.1
    y_stretched = librosa.effects.time_stretch(y_shifted, rate=np.random.uniform(0.9, 1.1))
    
    # Mild Gaussian noise
    y_noisy = y_stretched + 0.005 * np.random.randn(len(y_stretched))
    
    # Save augmented file
    sf.write(out_path, y_noisy, sr)

# generate base audio
file_count = 0
for i, phrase in enumerate(phrases):
    for voice in selected_voices:
        for rate in speech_rates:
            engine.setProperty('rate', rate)
            engine.setProperty('voice', voice.id)
            base_filename = os.path.join(output_dir, f"phrase_{i:02d}_{file_count}.wav")
            engine.save_to_file(phrase, base_filename)
            file_count += 1

engine.runAndWait()

# apply augmentations
existing_files = [f for f in os.listdir(output_dir) if f.endswith(".wav")]
aug_count = 0
for f in existing_files:
    input_path = os.path.join(output_dir, f)
    for n in range(augment_times):
        aug_filename = os.path.join(output_dir, f"aug_{aug_count}_{f}")
        augment_audio(input_path, aug_filename)
        aug_count += 1

total_files = len(existing_files) + aug_count
print(f"Generated {len(existing_files)} base files and {aug_count} augmented files. Total: {total_files}")
