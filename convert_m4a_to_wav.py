import librosa
import soundfile as sf
import os

def convert_m4a_to_wav(input_file, output_file=None, sr=16000, delete_original=False):
    """Convert M4A to WAV using librosa"""
    if output_file is None:
        output_file = input_file.replace('.m4a', '.wav')
    
    # Load audio
    audio, sample_rate = librosa.load(input_file, sr=sr)
    
    # Save as WAV
    sf.write(output_file, audio, sample_rate)
    print(f"Converted: {input_file} -> {output_file}")
    
    # Delete original M4A file if requested
    if delete_original:
        os.remove(input_file)
        print(f"Deleted: {input_file}")
    
    return output_file

def convert_folder(input_folder, output_folder=None, sr=16000, delete_original=False):
    """Convert all M4A files in a folder"""
    if output_folder is None:
        output_folder = input_folder
    
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if filename.endswith('.m4a'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename.replace('.m4a', '.wav'))
            convert_m4a_to_wav(input_path, output_path, sr, delete_original)

if __name__ == "__main__":
    # Convert and DELETE original M4A files
    convert_folder("real_data", "real_data", delete_original=True)