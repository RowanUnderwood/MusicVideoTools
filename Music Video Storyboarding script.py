import os
import json
import requests
import random
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import csv

# ================= CONFIGURATION =================
CSV_FILE = "omens in the rain.csv"
WORKFLOW_FILE = "ZImage_Poster_API.json"
LM_STUDIO_URL = "http://192.168.2.192:1234/v1/chat/completions"
COMFY_URL = "http://127.0.0.1:8188/prompt"
MODEL_ID = "qwen3-vl-8b-instruct-abliterated-v2.0"
LOG_FILE = "storyboard_generation_log.csv"

# Image Settings
VERSIONS_PER_SHOT = 4
BASE_NAME = "omens-in-the-rain"

# Aspect Ratio Config for Node 77 (WLSH)
TARGET_ASPECT_RATIO = "16:9" 
TARGET_DIRECTION = "landscape" 

# Reliability Settings
LM_TIMEOUT = 120
MAX_RETRIES = 2
# =================================================

def log_task(shot_num, version, ratio, prompt):
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Shot", "Version", "Ratio", "Prompt"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), shot_num, version, ratio, prompt])

def get_detailed_description(concept, version_num):
    """Queries Qwen-VL to expand the concept into a unique cinematic description."""
    system_prompt = "You are a professional cinematographer and visual effects supervisor."
    # Added variety instruction to the prompt to ensure each version is distinct
    user_prompt = (
        f"Design a unique visual variation (Variation #{version_num}) for a single frame "
        f"from a music video shot for MTV in the late 1990's based on this concept: ({concept})\n"
        "Describe the image in a very detailed way. Always stay true to the core themes for this video "
        "of rain, storms, sensuality, and omens.  Any characters present should be described in a sensual way. Describe the image only - do not include any additional notes, "
        "captions, or subtitles. The image should have no overlays or text."
    )
    
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.8 # Slightly increased temperature for more creative variation
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(LM_STUDIO_URL, json=payload, timeout=LM_TIMEOUT)
            return response.json()['choices'][0]['message']['content'].strip()
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                print(f"\n[!] Final Error calling LM Studio: {e}")
            continue
    return None

def send_to_comfy(workflow, prompt_text, filename, ratio, direction):
    """Injects parameters into the ZImage_Poster_API workflow."""
    try:
        # Update Positive Prompt (Node 6)
        if "6" in workflow:
            workflow["6"]["inputs"]["text"] = prompt_text
        
        # 1. Update Aspect Ratio parameters (Node 77)
        if "77" in workflow:
            workflow["77"]["inputs"]["aspect"] = str(ratio)
            workflow["77"]["inputs"]["direction"] = str(direction)
        
        # 2. CRITICAL: Redirect KSampler (Node 53) to use the WLSH Latent (Node 77)
        if "53" in workflow:
            workflow["53"]["inputs"]["latent_image"] = ["77", 0]
        
        # Update Seed (Node 57)
        if "57" in workflow:
            workflow["57"]["inputs"]["seed"] = random.randint(1, 10**15)
            
        # Update Filename Prefix (Node 73 - SaveImage)
        if "73" in workflow:
            workflow["73"]["inputs"]["filename_prefix"] = filename

        response = requests.post(COMFY_URL, json={"prompt": workflow}, timeout=15)
        return response.status_code == 200
    except Exception as e:
        print(f"\n[!] Error calling ComfyUI: {e}")
        return False

def main():
    if not os.path.exists(WORKFLOW_FILE):
        print(f"Error: {WORKFLOW_FILE} not found.")
        return

    with open(WORKFLOW_FILE, 'r') as f:
        workflow_template = json.load(f)

    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found.")
        return

    df = pd.read_csv(CSV_FILE)
    df.columns = df.columns.str.strip()
    
    print(f"--- STARTING MULTI-PROMPT GENERATION ---")
    print(f"Workflow: {WORKFLOW_FILE} | Versions per Shot: {VERSIONS_PER_SHOT}")
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Total Shots"):
        shot_id = row['Shot #']
        concept = row['Z-Image Storyboard Prompt']
        
        # Batch generate versions - LLM call is now INSIDE this loop
        for v in range(1, VERSIONS_PER_SHOT + 1):
            
            # 1. Expand concept via LLM for EVERY version
            detailed_prompt = get_detailed_description(concept, v)
            
            if not detailed_prompt:
                print(f"\n[!] Skipping Shot {shot_id} v{v} due to LLM failure.")
                continue
            
            filename = f"{BASE_NAME}-shot{shot_id}-v{v}"
            
            # 2. Send the unique description to ComfyUI
            success = send_to_comfy(
                workflow_template, 
                detailed_prompt, 
                filename, 
                TARGET_ASPECT_RATIO, 
                TARGET_DIRECTION
            )
            
            if success:
                log_task(shot_id, v, TARGET_ASPECT_RATIO, detailed_prompt)
            else:
                print(f"\n[!] Queue Failed for Shot {shot_id} v{v}")

    print(f"\n--- PROCESS COMPLETE ---")

if __name__ == "__main__":
    main()