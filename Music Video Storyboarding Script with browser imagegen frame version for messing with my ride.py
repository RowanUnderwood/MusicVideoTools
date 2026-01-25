import os
import requests
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import csv
import time
import glob
import pyautogui
import pygetwindow as gw
import pyperclip
import sys

# ================= CONFIGURATION =================
CSV_FILE = "messing with my ride.csv"
LM_STUDIO_URL = "http://192.168.2.192:1234/v1/chat/completions"
MODEL_ID = "qwen3-vl-8b-instruct-abliterated-v2.0"
LOG_FILE = "storyboard_generation_log.csv"

# Directory to monitor
DOWNLOADS_DIR = r"C:\Users\Jacob\Downloads"

# Image Settings
VERSIONS_PER_SHOT = 4

# Browser / Automation Settings
BROWSER_WINDOW_TITLE = "Chrome"
GENERATION_WAIT_TIME = 30
DOWNLOAD_WAIT_TIME = 30

# Resolution Configuration
# Set this to "4k" or "1080p" to switch modes
RESOLUTION_MODE = "1080p"

# Coordinate Mappings
# 4k Mode: Assumes browser is in a 1920x1080 quadrant starting at x=1920
# 1080p Mode: Assumes browser is Maximized on a 1920x1080 screen
CLICK_COORDS = {
    "4k": {
        "llm_tab": (2048, 17),
        "chat_window": (2585, 560),
        "new_chat": (1954, 220),
        # Calculated by adding 1920 to the 1080p X coords
        "zoom_click": (2949, 519),  # 1029 + 1920
        "save_click": (3787, 160),  # 1867 + 1920
        "close_zoom": (1959, 153)   # 39 + 1920
    },
    "1080p": {
        "llm_tab": (128, 17),
        "chat_window": (889, 573),
        "new_chat": (34, 220),
        "zoom_click": (1029, 519),
        "save_click": (1867, 160),
        "close_zoom": (39, 153)
    }
}

# Safety
pyautogui.FAILSAFE = True
# =================================================

def get_existing_progress():
    """Reads the log file to determine which Shot/Version combos are done."""
    completed = set()
    if os.path.isfile(LOG_FILE):
        try:
            log_df = pd.read_csv(LOG_FILE)
            
            # Clean column names just in case
            log_df.columns = log_df.columns.str.strip()
            
            for _, row in log_df.iterrows():
                # Check for bad filename
                filename = str(row.get('Filename', ''))
                
                # If filename is UNKNOWN, we do NOT add it to completed set (so it regenerates)
                if filename == "UNKNOWN_FILENAME":
                    continue
                    
                # Otherwise, mark as done
                completed.add((row['Shot#'], row['Version']))
                
        except Exception as e:
            print(f"[!] Warning: Could not read existing log file: {e}")
    return completed

def log_task(data_dict):
    """Updates the CSV log. Overwrites the row if Shot#/Version exists, otherwise appends."""
    file_exists = os.path.isfile(LOG_FILE)
    columns = [
        "Timestamp", "Shot#", "Version", "Filename",
        "Shot start frame", "Shot end frame", "Visual Concept",
        "Original Prompt", "LLM Prompt"
    ]
    
    if not file_exists:
        # Create new file with header and first row
        df = pd.DataFrame([data_dict], columns=columns)
        df.to_csv(LOG_FILE, index=False)
    else:
        try:
            # Read existing log
            df = pd.read_csv(LOG_FILE)
            
            # Create a mask to find if this specific Shot/Version already exists
            # (e.g. replacing an 'UNKNOWN_FILENAME' entry)
            mask = (df['Shot#'] == data_dict['Shot#']) & (df['Version'] == data_dict['Version'])
            
            if mask.any():
                # Overwrite existing row(s) with new data
                for col in columns:
                    df.loc[mask, col] = data_dict[col]
            else:
                # Append new row
                new_row = pd.DataFrame([data_dict])
                df = pd.concat([df, new_row], ignore_index=True)
            
            # Save back to CSV
            df.to_csv(LOG_FILE, index=False)
            
        except Exception as e:
            print(f"[!] Error updating log file: {e}")
            # Fallback to simple append if pandas fails hard (unlikely)
            with open(LOG_FILE, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writerow(data_dict)

def get_detailed_description(concept, visual_category, version_num):
    """Queries local LLM for image description."""
    system_prompt = "You are a professional cinematographer and visual effects supervisor."
    user_prompt = (
        f"Design a unique visual variation (Variation #{version_num}) for a single frame "
        f"from a music video. \n"
        f"Context/Category: {visual_category}\n"
        f"Core Concept: {concept}\n\n"
        "Describe the image in a very detailed way. Always stay true to the core themes. "
        "The core themese are nighttime, cyberpunk, Japan, and purple color grading. Describe the image only - do not include any additional notes, "
        "captions, or subtitles. The image should have no overlays or text."
    )

    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.8
    }

    for attempt in range(2):
        try:
            response = requests.post(LM_STUDIO_URL, json=payload, timeout=120)
            return response.json()['choices'][0]['message']['content'].strip()
        except Exception as e:
            if attempt == 1:
                print(f"\n[!] Final Error calling LM Studio: {e}")
            continue
    return None

def activate_browser():
    try:
        windows = gw.getWindowsWithTitle(BROWSER_WINDOW_TITLE)
        if windows:
            win = windows[0]
            if win.isMinimized:
                win.restore()
            win.activate()

            # Logic: If 1080p, we force maximize to ensure coords line up.
            # If 4k, we respect the user's existing window placement (quadrant).
            if RESOLUTION_MODE == "1080p" and not win.isMaximized:
                win.maximize()

            time.sleep(0.5)
            return True
        else:
            print(f"[!] Could not find window '{BROWSER_WINDOW_TITLE}'")
            return False
    except Exception:
        return False

def get_latest_file(directory):
    """
    Returns the most recently modified file in the directory, 
    IGNORING browser temp files (.tmp, .crdownload).
    """
    files = glob.glob(os.path.join(directory, "*"))
    
    # Filter out files that are temporary downloads
    # .crdownload is for Chrome/Edge, .tmp is generic, .part is Firefox
    valid_files = [
        f for f in files 
        if not f.lower().endswith(('.tmp', '.crdownload', '.part'))
    ]
    
    if not valid_files:
        return None
        
    return max(valid_files, key=os.path.getmtime)

def wait_for_new_file(directory, baseline_file, timeout=15):
    """
    Waits for a new file to appear that is not the baseline file.
    Includes error handling for race conditions where files are renamed (WinError 2).
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            current_latest = get_latest_file(directory)
            
            if current_latest and current_latest != baseline_file:
                # Check size to ensure write is starting/complete
                # If file vanishes (renames) between get_latest and getsize, 
                # the Exception block catches it.
                if os.path.getsize(current_latest) > 0:
                    return os.path.basename(current_latest)
                    
        except (OSError, FileNotFoundError):
            # This catches WinError 2. The file probably got renamed from 
            # .tmp to .png at the exact moment we checked it. 
            # We just loop again and capture it under its new name.
            pass
            
        time.sleep(1)
        
    return "UNKNOWN_FILENAME"

def send_to_browser_via_pyautogui(prompt_text):
    """Controls browser using configured coordinates and Clipboard Paste."""
    try:
        latest_file_before = get_latest_file(DOWNLOADS_DIR)

        if not activate_browser():
            return None

        # Select coordinates based on mode
        coords = CLICK_COORDS.get(RESOLUTION_MODE)
        if not coords:
            print(f"[!] Error: Invalid RESOLUTION_MODE '{RESOLUTION_MODE}'")
            return None

        # 1. Navigation Clicks
        pyautogui.click(coords["llm_tab"][0], coords["llm_tab"][1])
        time.sleep(0.2)

        pyautogui.click(coords["new_chat"][0], coords["new_chat"][1])
        time.sleep(0.2)

        pyautogui.click(coords["chat_window"][0], coords["chat_window"][1])
        time.sleep(0.8)

        # 2. Paste Prompt (Ctrl+V)
        full_prompt = "Generate the following image in a 16x9 aspect ratio." + prompt_text
        pyperclip.copy(full_prompt)
        pyautogui.hotkey('ctrl', 'v')
        time.sleep(0.5)

        # 3. Start Generation
        pyautogui.press('enter')

        print(f"   -> Generating ({GENERATION_WAIT_TIME}s)...")
        time.sleep(GENERATION_WAIT_TIME)

        # 4. Download (New 2-Click Process)
        print("   -> Clicking Zoom...")
        pyautogui.click(coords["zoom_click"][0], coords["zoom_click"][1])
        
        # Wait for image modal/zoom to open
        time.sleep(1.0) 
        
        print("   -> Clicking Download...")
        pyautogui.click(coords["save_click"][0], coords["save_click"][1])

        # 5. Monitor File
        print(f"   -> Waiting for file ({DOWNLOAD_WAIT_TIME}s)...")
        new_filename = wait_for_new_file(DOWNLOADS_DIR, latest_file_before, timeout=DOWNLOAD_WAIT_TIME)

        if new_filename == "UNKNOWN_FILENAME":
            error_msg = f"FATAL ERROR: Download timed out after {DOWNLOAD_WAIT_TIME}s. File not detected."
            print(f"   [!!!] {error_msg}")
            sys.exit(error_msg)
        else:
            print(f"   -> Detected: {new_filename}")
            
            # 6. Reset UI (Back out of zoom)
            print("   -> Closing zoom view...")
            pyautogui.click(coords["close_zoom"][0], coords["close_zoom"][1])
            time.sleep(0.5)

        return new_filename

    except Exception as e:
        print(f"\n[!] Error during automation: {e}")
        return None

def main():
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found.")
        return

    if not os.path.exists(DOWNLOADS_DIR):
        print(f"Error: Downloads directory '{DOWNLOADS_DIR}' not found.")
        return

    df = pd.read_csv(CSV_FILE)
    df.columns = df.columns.str.strip()

    completed_shots = get_existing_progress()
    print(f"--- RESUMING AUTOMATION ({RESOLUTION_MODE} Mode) ---")
    print(f"Monitor: {DOWNLOADS_DIR}")
    print(f"Skipping {len(completed_shots)} previously completed tasks (ignoring UNKNOWNs).")

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Total Shots"):
        shot_id = row['Shot#']
        concept = row['Image Storyboard Prompt']
        visual_category = str(row['Visual Concept'])
        start_frame = row['Shot start frame']
        end_frame = row['Shot end frame']

        if "Live Performance" in visual_category:
            continue

        for v in range(1, VERSIONS_PER_SHOT + 1):

            if (shot_id, v) in completed_shots:
                continue

            detailed_prompt = get_detailed_description(concept, visual_category, v)

            if not detailed_prompt:
                print(f"\n[!] Skipping Shot {shot_id} v{v} due to LLM failure.")
                continue

            print(f"\nProcessing Shot {shot_id} [{visual_category}] v{v}...")

            detected_filename = send_to_browser_via_pyautogui(detailed_prompt)

            if detected_filename:
                log_data = {
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Shot#": shot_id,
                    "Version": v,
                    "Filename": detected_filename,
                    "Shot start frame": start_frame,
                    "Shot end frame": end_frame,
                    "Visual Concept": visual_category,
                    "Original Prompt": concept,
                    "LLM Prompt": detailed_prompt
                }
                log_task(log_data)
            else:
                print(f"\n[!] Automation Failed for Shot {shot_id} v{v}")

    print(f"\n--- PROCESS COMPLETE ---")

if __name__ == "__main__":
    main()