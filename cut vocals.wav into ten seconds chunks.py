from pydub import AudioSegment
import os

def chunk_audio(filename, chunk_length_ms=10000):
    # Load the audio file
    audio = AudioSegment.from_wav(filename)
    
    # Calculate total duration
    duration_ms = len(audio)
    
    # Iterate through the file in 10-second (10000ms) steps
    for i, start_ms in enumerate(range(0, duration_ms, chunk_length_ms)):
        chunk = audio[start_ms:start_ms + chunk_length_ms]
        
        # Define the output name (e.g., vocals_chunk_0.wav)
        chunk_name = f"vocals_chunk_{i}.wav"
        
        # Export the chunk to the current directory
        chunk.export(chunk_name, format="wav")
        print(f"Exported: {chunk_name}")

if __name__ == "__main__":
    target_file = "vocals.wav"
    
    if os.path.exists(target_file):
        chunk_audio(target_file)
    else:
        print(f"Error: {target_file} not found in the current directory.")