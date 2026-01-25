import os
import json
import subprocess
import glob
import re
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip

# ================= CONFIGURATION =================
BASE_NAME = "omens-in-the-rain"
TOTAL_SHOTS = 36
FPS = 25
TARGET_DURATION = 5.0  # seconds
SELECTIONS_FILE = "video_selections.json"
OUTPUT_FILENAME = "final_music_video.mp4"
# =================================================

def load_selections():
    if os.path.exists(SELECTIONS_FILE):
        with open(SELECTIONS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_selection(shot_num, filename):
    selections = load_selections()
    selections[str(shot_num)] = filename
    with open(SELECTIONS_FILE, 'w') as f:
        json.dump(selections, f, indent=4)

def get_files_for_shot(shot_num):
    # Pattern to match: omens-in-the-rain-shot[N]-v[M]...mp4
    pattern = f"{BASE_NAME}-shot{shot_num}-v*.mp4"
    files = glob.glob(pattern)
    # Sort alphabetically so v1, v2, v3 appear in order
    files.sort()
    return files

def select_shots():
    selections = load_selections()
    
    for shot_num in range(1, TOTAL_SHOTS + 1):
        if str(shot_num) in selections:
            print(f"Shot {shot_num} already selected: {selections[str(shot_num)]}")
            continue

        versions = get_files_for_shot(shot_num)
        
        if not versions:
            print(f" [!] Warning: No files found for Shot {shot_num}. Skipping...")
            continue

        print(f"\n--- Reviewing Shot {shot_num} ({len(versions)} versions) ---")
        
        # Launch VLC with all versions for this shot
        # VLC flag '--play-and-exit' will close after the playlist finishes
        try:
            print("Launching VLC playlist...")
            subprocess.Popen(['vlc', '--play-and-exit'] + versions)
        except FileNotFoundError:
            print("Error: VLC command not found. Please ensure VLC is in your PATH.")

        # Display numbered list in terminal
        for i, v in enumerate(versions):
            print(f"[{i+1}] {v}")

        while True:
            try:
                choice = int(input(f"Select version for Shot {shot_num} (1-{len(versions)}): "))
                if 1 <= choice <= len(versions):
                    selected_file = versions[choice - 1]
                    save_selection(shot_num, selected_file)
                    print(f"Selected: {selected_file}")
                    break
                else:
                    print("Invalid choice.")
            except ValueError:
                print("Please enter a number.")

def assemble_final_video():
    selections = load_selections()
    if len(selections) < TOTAL_SHOTS:
        print(f"\n[!] Only {len(selections)}/{TOTAL_SHOTS} shots selected. Proceed with partial assembly? (y/n)")
        if input().lower() != 'y': return

    # Find the audio file
    audio_files = glob.glob("*.mp3")
    if not audio_files:
        print("Error: No .mp3 file found in the current directory.")
        return
    audio_path = audio_files[0]
    print(f"Using audio: {audio_path}")

    clips = []
    print("\nProcessing and trimming clips...")
    
    # Sort keys to ensure sequence
    sorted_shot_nums = sorted([int(k) for k in selections.keys()])

    for shot_num in sorted_shot_nums:
        file_path = selections[str(shot_num)]
        print(f"Adding Shot {shot_num}: {file_path}")
        
        clip = VideoFileClip(file_path)
        
        # Force trimming to exactly 5 seconds
        if clip.duration > TARGET_DURATION:
            clip = clip.subclip(0, TARGET_DURATION)
        
        clips.append(clip)

    print("Concatenating video...")
    final_video = concatenate_videoclips(clips, method="compose")
    
    print("Attaching audio...")
    audio = AudioFileClip(audio_path)
    # Optional: Trim audio to match video or vice versa
    final_video = final_video.set_audio(audio)

    print(f"Rendering final video to {OUTPUT_FILENAME}...")
    final_video.write_videofile(OUTPUT_FILENAME, fps=FPS, codec="libx264", audio_codec="aac")
    
    # Close clips to free memory
    for c in clips: c.close()
    print("Done!")

if __name__ == "__main__":
    print("=== Music Video Assembly Script ===")
    
    # Step 1: User Selection Phase
    select_shots()
    
    # Step 2: Final Render Phase
    print("\nSelections complete. Starting final assembly...")
    assemble_final_video()