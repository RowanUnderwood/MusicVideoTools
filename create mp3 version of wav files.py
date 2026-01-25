import os
from pydub import AudioSegment

def convert_wav_to_mp3():
    # Get the current working directory
    current_dir = os.getcwd()
    
    # List all files in the directory
    files = [f for f in os.listdir(current_dir) if f.lower().endswith('.wav')]
    
    if not files:
        print("No .wav files found in the current directory.")
        return

    print(f"Found {len(files)} WAV files. Starting conversion...\n")

    for file in files:
        try:
            # Define input and output names
            wav_path = os.path.join(current_dir, file)
            mp3_name = os.path.splitext(file)[0] + ".mp3"
            mp3_path = os.path.join(current_dir, mp3_name)

            # Skip if mp3 already exists to avoid overwriting (optional)
            if os.path.exists(mp3_path):
                print(f"Skipping {file} (MP3 already exists)")
                continue

            # Convert
            print(f"Converting: {file} -> {mp3_name}")
            audio = AudioSegment.from_wav(wav_path)
            
            # Export as mp3 (you can adjust bitrate e.g., bitrate="192k")
            audio.export(mp3_path, format="mp3", bitrate="192k")
            
        except Exception as e:
            print(f"Failed to convert {file}: {e}")

    print("\nBatch conversion complete.")

if __name__ == "__main__":
    convert_wav_to_mp3()