import gradio as gr
import pandas as pd
import os
import sys
import json
import requests
import websocket
import uuid
import urllib.request
from pydub import AudioSegment, silence
from moviepy.editor import VideoFileClip, AudioFileClip, ColorClip, concatenate_videoclips, ImageClip
from datetime import datetime
import shutil
import math
import threading
import time
import random
import re
import glob
import io
import copy
import keyboard  # Requires: pip install keyboard

# ==========================================
# CONFIGURATION
# ==========================================
COMFY_URL = "127.0.0.1:8188"
LM_STUDIO_URL = "http://127.0.0.1:1234/v1"

# WORKFLOW FILES
WORKFLOW_IMG = "ZImage_Poster_API.json"
WORKFLOW_VID_ACTION = "wan22_i2v_v17_jakes version slow motion api version.json"
WORKFLOW_VID_VOCAL = "011426-LTX2-AudioSync-i2v-Ver2-Jakes Version API.json"

REQUIRED_COLUMNS = [
    "Shot_ID", "Type", 
    "Start_Time", "End_Time", "Duration", 
    "Start_Frame", "End_Frame", "Total_Frames",
    "Lyrics", "Image_Prompt", "Video_Prompt", "Image_Path", "Video_Path", "Status"
]

DEFAULT_CONCEPT_PROMPT = (
    "Context: The overarching plot is: {plot}\n"
    "Previous Shot Visual: {prev_shot}\n"
    "Current Shot Info: Timestamp {start}s, Duration {duration}s, Type: {type}.\n"
    "Task: Write a highly detailed visual description of the first frame for this specific shot. "
    "Include details on lighting, texture, composition, and mood. "
    "Focus on the narrative flow from the previous shot."
)

DEFAULT_VOCAL_PROMPT = (
    "Context: This is a performance shot, visually distinct from the overarching narrative plot.\n"
    "Singer, Band, and Venue Info: {performance_desc}\n"
    "Current Shot Info: Timestamp {start}s, Duration {duration}s, Type: {type}.\n"
    "Task: Write a highly detailed visual description of the first frame for this specific performance shot. "
    "Include details on lighting, texture, composition, and mood. "
    "CRITICAL: This shot MUST be a tight close-up on the singer (e.g., extreme close-up on the face or microphone). "
    "Do not use medium or wide shots, as highly detailed facial features are required for proper lip-syncing."
)

DEFAULT_VIDEO_PROMPT = (
    "Generate a prompt based on this image description for a video model to encompass 5 seconds of action. "
    "Describe the scene, camera motion, emotions, lighting and performance only. "
    "Pay special attention to the camera's motion. Do not include any additional notes or titles in your description."
)

# ==========================================
# PRELOAD WORKFLOWS
# ==========================================
PRELOADED_WORKFLOW_IMG = {}
if os.path.exists(WORKFLOW_IMG):
    with open(WORKFLOW_IMG, 'r') as f:
        PRELOADED_WORKFLOW_IMG = json.load(f)

PRELOADED_WORKFLOW_VID_ACTION = {}
if os.path.exists(WORKFLOW_VID_ACTION):
    with open(WORKFLOW_VID_ACTION, 'r') as f:
        PRELOADED_WORKFLOW_VID_ACTION = json.load(f)

PRELOADED_WORKFLOW_VID_VOCAL = {}
if os.path.exists(WORKFLOW_VID_VOCAL):
    with open(WORKFLOW_VID_VOCAL, 'r') as f:
        PRELOADED_WORKFLOW_VID_VOCAL = json.load(f)

# ==========================================
# SYSTEM UTILITIES
# ==========================================

def get_file_path(file_obj):
    """Safely extracts a file path from a Gradio file component."""
    if file_obj is None: return None
    if isinstance(file_obj, str): return file_obj
    if hasattr(file_obj, 'name'): return file_obj.name
    if isinstance(file_obj, dict) and 'name' in file_obj: return file_obj['name']
    return None

def restart_application():
    """Restarts the current python process."""
    print("♻️ Restarting application via hotkey...")
    python = sys.executable
    os.execl(python, python, *sys.argv)

def snap_to_frame(seconds, fps=24):
    frame_dur = 1.0 / fps
    return round(seconds / frame_dur) * frame_dur

def get_ltx_frame_count(target_seconds, fps=24):
    min_frames = math.ceil(target_seconds * fps)
    valid_buckets = [1 + 8*k for k in range(60)] 
    for bucket in valid_buckets:
        if bucket >= min_frames:
            return bucket
    return valid_buckets[-1]

def get_ltx_duration(seconds, fps=24):
    frames = get_ltx_frame_count(seconds, fps)
    return frames / fps

# ==========================================
# BACKEND UTILITIES
# ==========================================

class ComfyBridge:
    def __init__(self, server_address=COMFY_URL):
        self.server_address = server_address
        self.client_id = str(uuid.uuid4())
        self.ws = websocket.WebSocket()

    def connect(self):
        try:
            self.ws.connect(f"ws://{self.server_address}/ws?clientId={self.client_id}")
            return True
        except Exception as e:
            print(f"ComfyUI Connection Failed: {e}")
            return False
            
    def close(self):
        try:
            self.ws.close()
        except Exception:
            pass

    def upload_image(self, filepath):
        url = f"http://{self.server_address}/upload/image"
        try:
            with open(filepath, 'rb') as f:
                files = {'image': f}
                data = {'overwrite': 'true'}
                response = requests.post(url, files=files, data=data)
            return response.json().get('name')
        except Exception as e:
            print(f"Upload failed: {e}")
            return None

    def queue_prompt(self, workflow):
        p = {"prompt": workflow, "client_id": self.client_id}
        data = json.dumps(p).encode('utf-8')
        req = urllib.request.Request(f"http://{self.server_address}/prompt", data=data)
        try:
            return json.loads(urllib.request.urlopen(req).read())
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8')
            print(f"🔴 ComfyUI API Error: {e.code} {e.reason}")
            raise Exception(f"ComfyUI Error: {error_body}")

    def track_progress(self, prompt_id):
        while True:
            try:
                out = self.ws.recv()
                if isinstance(out, str):
                    message = json.loads(out)
                    if message['type'] == 'executing':
                        data = message['data']
                        if data['node'] is None and data['prompt_id'] == prompt_id:
                            break
            except Exception:
                break

    def get_history(self, prompt_id):
        with urllib.request.urlopen(f"http://{self.server_address}/history/{prompt_id}") as response:
            return json.loads(response.read())

    def download_file(self, filename, subfolder, file_type, save_path):
        url = f"http://{self.server_address}/view?filename={filename}&subfolder={subfolder}&type={file_type}"
        urllib.request.urlretrieve(url, save_path)

class LLMBridge:
    def __init__(self, base_url=LM_STUDIO_URL):
        self.base_url = base_url

    def get_models(self):
        try:
            resp = requests.get(f"{self.base_url}/models", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                return [m['id'] for m in data['data']]
        except Exception as e:
            pass
        return ["qwen3-vl-8b-instruct-abliterated-v2.0"]

    def query(self, system_prompt, user_prompt, model, temperature=0.7):
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_tokens": 1000
        }
        try:
            resp = requests.post(url, json=payload, timeout=120)
            if resp.status_code != 200:
                return f"Error {resp.status_code} from LLM: {resp.text}"
            return resp.json()['choices'][0]['message']['content'].strip()
        except Exception as e:
            return f"Error: {str(e)}"

# ==========================================
# PROJECT MANAGER
# ==========================================

class ProjectManager:
    def __init__(self):
        self.current_project = None
        self.base_dir = "projects"
        os.makedirs(self.base_dir, exist_ok=True)
        self.df = pd.DataFrame(columns=REQUIRED_COLUMNS)
        self.stop_generation = False 
        self.stop_image_generation = False
        self.stop_video_generation = False
        self.is_generating = False # Generation concurrency lock
        
        # Time Tracking Variables
        self.total_time_spent = 0
        self.session_start_time = None

    def sanitize_name(self, name):
        return re.sub(r'[\\/*?:"<>|]', "", name).strip().replace(" ", "_")
        
    def get_current_total_time(self):
        """Calculates total active project time, auto-saves it, and returns it."""
        if self.session_start_time and self.current_project:
            elapsed = time.time() - self.session_start_time
            self.session_start_time = time.time()  # Reset session marker
            self.total_time_spent += elapsed
            
            # Auto-save time to settings
            settings = self.load_project_settings()
            settings["total_time_spent"] = self.total_time_spent
            path = os.path.join(self.base_dir, self.current_project, "settings.json")
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(settings, f, indent=4)
            except Exception:
                pass
                
        return self.total_time_spent

    def create_project(self, name):
        if not name: return "Invalid name"
        clean_name = self.sanitize_name(name)
        path = os.path.join(self.base_dir, clean_name)
        folders = ["assets", "audio_chunks", "images", "videos", "renders"]
        
        if os.path.exists(path):
            return f"Project '{clean_name}' already exists."

        for f in folders:
            os.makedirs(os.path.join(path, f), exist_ok=True)
        
        self.df = pd.DataFrame(columns=REQUIRED_COLUMNS)
        self.df.to_csv(os.path.join(path, "shot_list.csv"), index=False)
        self.current_project = clean_name
        
        self.total_time_spent = 0
        self.session_start_time = time.time()
        
        with open(os.path.join(path, "lyrics.txt"), "w") as f:
            f.write("")
        return f"Project '{clean_name}' created."

    def load_project(self, name):
        path = os.path.join(self.base_dir, name)
        csv_path = os.path.join(path, "shot_list.csv")
        if os.path.exists(csv_path):
            self.df = pd.read_csv(csv_path)
            for col in REQUIRED_COLUMNS:
                if col not in self.df.columns:
                    self.df[col] = ""
            self.current_project = name
            
            # Load stored time tracking
            settings = self.load_project_settings()
            self.total_time_spent = settings.get("total_time_spent", 0)
            self.session_start_time = time.time()
            
            return f"Loaded '{name}'", self.df
        return "Project not found.", pd.DataFrame()

    def import_csv(self, file_obj):
        if not self.current_project:
            return "No project loaded.", pd.DataFrame()
        
        try:
            new_df = pd.read_csv(get_file_path(file_obj))
            missing_cols = [c for c in REQUIRED_COLUMNS if c not in new_df.columns]
            if missing_cols:
                for c in missing_cols:
                    new_df[c] = ""
            
            self.df = new_df
            self.save_data()
            return "✅ CSV Uploaded & Verified", self.df
        except Exception as e:
            return f"❌ Error reading CSV: {e}", self.df

    def export_csv(self):
        if not self.current_project or self.df.empty:
            return None
        return os.path.join(self.base_dir, self.current_project, "shot_list.csv")

    def save_data(self):
        if self.current_project:
            path = os.path.join(self.base_dir, self.current_project, "shot_list.csv")
            self.df.to_csv(path, index=False)

    def get_path(self, subfolder):
        return os.path.join(self.base_dir, self.current_project, subfolder)
        
    def save_lyrics(self, text):
        if not self.current_project: return
        path = os.path.join(self.base_dir, self.current_project, "lyrics.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
            
    def get_lyrics(self):
        if not self.current_project: return ""
        path = os.path.join(self.base_dir, self.current_project, "lyrics.txt")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        return ""

    def save_project_settings(self, settings_dict):
        if not self.current_project: return "No project loaded."
        
        # Make sure we carry over total time when resaving settings
        settings_dict["total_time_spent"] = self.get_current_total_time()
        
        path = os.path.join(self.base_dir, self.current_project, "settings.json")
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(settings_dict, f, indent=4)
            return f"Settings saved."
        except Exception as e:
            return f"Error saving settings: {e}"

    def load_project_settings(self):
        if not self.current_project: return {}
        path = os.path.join(self.base_dir, self.current_project, "settings.json")
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
        
    def save_asset(self, source_path, filename):
        if not self.current_project or not source_path: return None
        dest = os.path.join(self.get_path("assets"), filename)
        if os.path.abspath(source_path) == os.path.abspath(dest): return dest
        shutil.copy(source_path, dest)
        return dest

    def get_asset_path_if_exists(self, filename):
        if not self.current_project: return None
        path = os.path.join(self.get_path("assets"), filename)
        return path if os.path.exists(path) else None


# ==========================================
# LOGIC: TIMELINE & CONCEPTS
# ==========================================

def get_existing_projects():
    base_dir = "projects"
    if not os.path.exists(base_dir): return []
    projects = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    return sorted(projects)

def scan_vocals_advanced(vocals_file_path, project_name, min_silence, silence_thresh, shot_mode, min_dur, max_dur, pm):
    if not project_name or not vocals_file_path or not os.path.exists(vocals_file_path): return pd.DataFrame()

    try:
        audio = AudioSegment.from_file(vocals_file_path)
    except Exception as e:
        print(f"Error loading audio: {e}")
        return pd.DataFrame()

    total_duration = audio.duration_seconds
    nonsilent_ranges = silence.detect_nonsilent(audio, min_silence_len=int(min_silence), silence_thresh=silence_thresh)
    
    new_rows = []
    current_cursor = 0.0
    shot_counter = 1
    fps = 24.0
    frame_step = 1.0 / fps

    def create_row(sType, start, end):
        dur = end - start
        start_frame = round(start * fps)
        end_frame = round(end * fps)
        total_frames = end_frame - start_frame
        
        return {
            "Shot_ID": f"S{shot_counter:03d}",
            "Type": sType,
            "Start_Time": float(f"{start:.4f}"),
            "End_Time": float(f"{end:.4f}"),
            "Duration": float(f"{dur:.4f}"),
            "Start_Frame": int(start_frame),
            "End_Frame": int(end_frame),
            "Total_Frames": int(total_frames),
            "Status": "Pending"
        }

    for (start_ms, end_ms) in nonsilent_ranges:
        voc_start = start_ms / 1000.0
        voc_end = end_ms / 1000.0
        
        aligned_start = math.floor(voc_start * fps) / fps
        aligned_end = math.ceil(voc_end * fps) / fps
        
        if aligned_start < current_cursor:
            aligned_start = current_cursor
            
        while (aligned_start - current_cursor) >= frame_step: 
            gap_duration = aligned_start - current_cursor
            chosen_raw = random.uniform(min_dur, max_dur)
            if chosen_raw > 5.0: chosen_raw = 5.0
            if chosen_raw > gap_duration: chosen_raw = gap_duration

            chosen_dur = get_ltx_duration(chosen_raw, fps)
            action_end = current_cursor + chosen_dur
            
            new_rows.append(create_row("Action", current_cursor, action_end))
            shot_counter += 1
            current_cursor = action_end
            
            if current_cursor >= aligned_start: break

        vocal_cursor = current_cursor
        while vocal_cursor < (aligned_end - frame_step):
            remaining = aligned_end - vocal_cursor
            chunk_raw = 5.0 if remaining > 5.0 else remaining
            chunk_size = get_ltx_duration(chunk_raw, fps)
            
            new_rows.append(create_row("Vocal", vocal_cursor, vocal_cursor + chunk_size))
            shot_counter += 1
            vocal_cursor += chunk_size
            
        current_cursor = vocal_cursor

    remaining_time = total_duration - current_cursor
    while remaining_time > frame_step:
        chosen_raw = random.uniform(min_dur, max_dur)
        if chosen_raw > 5.0: chosen_raw = 5.0
        if chosen_raw > remaining_time: chosen_raw = remaining_time
        
        chosen_dur = get_ltx_duration(chosen_raw, fps)
        
        new_rows.append(create_row("Action", current_cursor, current_cursor + chosen_dur))
        shot_counter += 1
        current_cursor += chosen_dur
        remaining_time = total_duration - current_cursor
    
    new_df = pd.DataFrame(new_rows)
    for col in REQUIRED_COLUMNS:
        if col not in new_df.columns: new_df[col] = ""
            
    pm.df = new_df
    pm.save_data()
    return pm.df

def generate_overarching_plot(concept, lyrics, llm_model, pm):
    llm = LLMBridge()
    df = pm.df
    if df.empty: return "Error: Timeline is empty."

    timeline_str = ""
    for idx, row in df.iterrows():
        if row['Type'] == 'Vocal':
            timeline_str += f"[{row['Start_Time']:.2f}s - {row['End_Time']:.2f}s: SINGING]\n"
    
    sys_prompt = "You are a creative writer for music videos."
    user_prompt = (
        f"Rough Concept: {concept}\n\nLyrics:\n{lyrics}\n\nTimeline:\n{timeline_str}\n\n"
        "Task: Write a cohesive linear plot summary for this video (max 300 words)."
    )
    return llm.query(sys_prompt, user_prompt, llm_model)

def generate_performance_description(concept, plot, llm_model):
    llm = LLMBridge()
    sys_prompt = "You are a casting director and set designer."
    user_prompt = (
        f"Concept: {concept}\nPlot: {plot}\n\n"
        "Task: Describe the physical appearance and style of the lead singer, the backing band, and the performance venue for this music video. "
        "Keep it concise (2-3 sentences). Focus on visual descriptors."
    )
    return llm.query(sys_prompt, user_prompt, llm_model)

def generate_concepts_logic(overarching_plot, prompt_template, vocal_prompt_template, llm_model, rough_concept, performance_desc, video_prompt_template, specific_shot_id, pm):
    llm = LLMBridge()
    df = pm.df
    pm.stop_generation = False
    
    if df.empty: 
        yield df, "Error: Timeline is empty."
        return

    if specific_shot_id == "ALL":
        mask = pd.Series(True, index=df.index)
    elif specific_shot_id:
        mask = (df['Shot_ID'].astype(str).str.upper() == str(specific_shot_id).upper())
    else:
        mask = (df['Image_Prompt'].isna() | (df['Image_Prompt'] == ""))
        
    indices_to_process = df[mask].index.tolist()
    
    yield df, f"🚀 Starting generation for {len(indices_to_process)} shots..."
    
    for count, index in enumerate(indices_to_process, 1):
        if pm.stop_generation: 
            yield df, "🛑 Stopped. Waiting for current task to complete..."
            break
            
        row = df.loc[index]
        yield df, f"⏳ Generating ({count}/{len(indices_to_process)}): Prompts for {row['Shot_ID']}..."
        
        prev_shot_text = "None (Start of video)"
        if index > 0:
             prev_shot_text = df.loc[index - 1, 'Image_Prompt']
             if pd.isna(prev_shot_text): prev_shot_text = "N/A"

        if row['Type'] == 'Vocal':
            prompt_input = vocal_prompt_template.replace("{performance_desc}", performance_desc)\
                .replace("{type}", row['Type'])\
                .replace("{start}", f"{row['Start_Time']:.1f}")\
                .replace("{duration}", f"{row['Duration']:.1f}")
        else:
            filled_prompt = prompt_template.replace("{plot}", overarching_plot)\
                .replace("{type}", row['Type'])\
                .replace("{start}", f"{row['Start_Time']:.1f}")\
                .replace("{duration}", f"{row['Duration']:.1f}")\
                .replace("{prev_shot}", prev_shot_text)
            prompt_input = filled_prompt
        
        image_prompt_text = llm.query("Describe the scene in high detail using plain English.", prompt_input, llm_model)
        
        vid_prompt_in = f"Visual Description: {image_prompt_text}\n\nTask: {video_prompt_template}"
        final_vid_prompt = llm.query("You are a video prompt expert.", vid_prompt_in, llm_model)

        df.at[index, 'Image_Prompt'] = image_prompt_text
        df.at[index, 'Video_Prompt'] = final_vid_prompt
        
        pm.df = df
        pm.save_data()
        
    if not pm.stop_generation:
        yield df, "🎉 Concept Generation Complete!"

def stop_gen(pm):
    pm.stop_generation = True
    pm.stop_image_generation = True
    pm.stop_video_generation = True
    return "🛑 Stopping... Waiting for current task to complete..."

# ==========================================
# LOGIC: IMAGE GENERATION & GALLERY
# ==========================================

def get_project_images(pm, project_name=None):
    proj = project_name if project_name else pm.current_project
    if not proj: return []

    img_dir = os.path.join(pm.base_dir, proj, "images")
    if not os.path.exists(img_dir): return []
    
    files = glob.glob(os.path.join(img_dir, "*.png"))
    
    # Sort files by Shot_ID first, then by filename/time to group versions chronologically
    def sort_key(filepath):
        fname = os.path.basename(filepath)
        parts = fname.split("_")
        shot_id = parts[0].upper() if len(parts) > 0 else fname
        return (shot_id, filepath)
        
    files = sorted(files, key=sort_key)
    gallery_data = []
    
    for f in files:
        fname = os.path.basename(f)
        parts = fname.split("_")
        caption = f"{parts[0]}" if len(parts) >= 2 else fname
        gallery_data.append((f, caption))
        
    return gallery_data

def get_image_count_for_shot(shot_id, img_list):
    count = 0
    for path, caption in img_list:
        if os.path.basename(path).upper().startswith(f"{str(shot_id).upper()}_"):
            count += 1
    return count

def generate_image_for_shot(shot_id, pm):
    comfy = ComfyBridge()
    if not comfy.connect(): return None
    
    try:
        row_idx = pm.df.index[pm.df['Shot_ID'].astype(str).str.upper() == str(shot_id).upper()].tolist()
        if not row_idx: return None
        row = pm.df.loc[row_idx[0]]
        
        prompt_text = row.get('Image_Prompt')
        if not prompt_text or pd.isna(prompt_text): return None

        print(f"🎬 Sending Image Prompt for {shot_id}: {prompt_text}")

        # Load from preloaded JSON dict
        wf = copy.deepcopy(PRELOADED_WORKFLOW_IMG)
        if not wf: return None
        
        # Inject random seed safely across ALL possible seed nodes to prevent ComfyUI caching bugs
        new_seed = random.randint(1, 100000000000000)
        for node_id, node_data in wf.items():
            if isinstance(node_data, dict) and "inputs" in node_data:
                if "seed" in node_data["inputs"]:
                    node_data["inputs"]["seed"] = new_seed
                if "noise_seed" in node_data["inputs"]:
                    node_data["inputs"]["noise_seed"] = new_seed
                    
        if "6" in wf: wf["6"]["inputs"]["text"] = prompt_text
        if "60" in wf:
            wf["60"]["inputs"]["custom_ratio"] = True
            wf["60"]["inputs"]["custom_aspect_ratio"] = "16:9"
        if "73" in wf:
            clean_proj = pm.sanitize_name(pm.current_project)
            wf["73"]["inputs"]["filename_prefix"] = f"{clean_proj}/{shot_id}"

        resp = comfy.queue_prompt(wf)
        prompt_id = resp['prompt_id']
        comfy.track_progress(prompt_id)
        
        history = comfy.get_history(prompt_id)[prompt_id]
        outputs = history['outputs']
        img_node = next((v for k,v in outputs.items() if 'images' in v), None)
        
        if img_node:
            fname = img_node['images'][0]['filename']
            subfolder = img_node['images'][0]['subfolder']
            file_type = img_node['images'][0]['type']
            
            save_name = f"{shot_id}_v{int(time.time())}.png"
            local_path = os.path.join(pm.get_path("images"), save_name)
            comfy.download_file(fname, subfolder, file_type, local_path)
            
            pm.df.at[row_idx[0], 'Image_Path'] = local_path
            pm.save_data()
            return local_path
    except Exception as e:
        print(f"❌ GENERATION FAILED: {e}")
        return None
    finally:
        comfy.close()
    return None

def advanced_batch_image_generation(mode, target_versions, pm):
    """Handles various generation modes with clear interruption checks."""
    if pm.is_generating:
        yield [], None, "❌ Error: A generation process is already actively running."
        return

    pm.stop_image_generation = False
    pm.is_generating = True
    
    try:
        if pm.current_project: pm.load_project(pm.current_project)
        
        df = pm.df
        if df.empty: 
            yield [], None, "No shots found."
            return

        current_gallery = get_project_images(pm)
        yield current_gallery, None, f"🚀 Starting image generation ({mode})..."

        # Filter logic based on dropdown selection
        if mode == "Generate all Action Shots":
            shot_ids = df[df['Type'] == 'Action']['Shot_ID'].tolist()
        elif mode == "Generate all Vocal Shots":
            shot_ids = df[df['Type'] == 'Vocal']['Shot_ID'].tolist()
        else:
            shot_ids = df['Shot_ID'].tolist()
        
        for shot_id in shot_ids:
            if pm.stop_image_generation: break
                
            row = df[df['Shot_ID'] == shot_id].iloc[0]
            if pd.isna(row.get('Image_Prompt')) or not str(row.get('Image_Prompt')).strip():
                yield current_gallery, None, f"⚠️ Skipped {shot_id}: Missing image prompt."
                continue

            # If Regenerate is selected, explicitly clear the slate for each matching shot ID
            if mode == "Regenerate all Shots":
                img_dir = pm.get_path("images")
                if os.path.exists(img_dir):
                    for f in glob.glob(os.path.join(img_dir, f"{shot_id}_*.png")):
                        try: os.remove(f)
                        except Exception: pass
                current_gallery = get_project_images(pm)
                
            current_count = get_image_count_for_shot(shot_id, current_gallery)
            
            while current_count < target_versions:
                if pm.stop_image_generation: break
                
                status_msg = f"⏳ Working... Generating Image for {shot_id} (Version {current_count + 1}/{target_versions})"
                yield current_gallery, None, status_msg
                
                new_img_path = generate_image_for_shot(shot_id, pm)
                
                if new_img_path:
                    current_gallery = get_project_images(pm)
                    current_count += 1
                    yield current_gallery, new_img_path, f"✅ Finished {shot_id}"
                else:
                    yield current_gallery, None, f"❌ Failed to generate image for {shot_id}."
                    break
                    
        if pm.stop_image_generation:
            yield current_gallery, None, "🛑 Generation Stopped by User."
        else:
            yield current_gallery, None, "🎉 Batch Image Generation Complete."
    finally:
        pm.is_generating = False

def delete_image_file(path, project_name, pm):
    if not path or not os.path.exists(path):
        return get_project_images(pm, project_name), None
    try:
        os.remove(path)
    except Exception as e:
        print(f"Error deleting file: {e}")
    return get_project_images(pm, project_name), None


# ==========================================
# LOGIC: VIDEO GENERATION
# ==========================================

def get_project_videos(pm, project_name=None):
    proj = project_name if project_name else pm.current_project
    if not proj: return []

    vid_dir = os.path.join(pm.base_dir, proj, "videos")
    if not os.path.exists(vid_dir): return []
    
    files = glob.glob(os.path.join(vid_dir, "*.mp4"))
    
    # Sort files by Shot_ID first, then by filename/time to group versions chronologically
    def sort_key(filepath):
        fname = os.path.basename(filepath)
        parts = fname.split("_")
        shot_id = parts[0].upper() if len(parts) > 0 else fname
        return (shot_id, filepath)
        
    files = sorted(files, key=sort_key)
    gallery_data = []
    
    for f in files:
        fname = os.path.basename(f)
        parts = fname.split("_")
        caption = f"{parts[0]}" if len(parts) >= 2 else fname
        gallery_data.append((f, caption))
        
    return gallery_data

def delete_video_file(path, project_name, pm):
    if not path or not os.path.exists(path):
        return get_project_videos(pm, project_name), None
    try:
        os.remove(path)
    except Exception as e:
        print(f"Error deleting file: {e}")
    return get_project_videos(pm, project_name), None

def get_video_count_for_shot(shot_id, vid_list):
    count = 0
    for path, caption in vid_list:
        if os.path.basename(path).upper().startswith(f"{str(shot_id).upper()}_"):
            count += 1
    return count

def generate_video_for_shot(shot_id, pm):
    comfy = ComfyBridge()
    if not comfy.connect(): return None
    
    try:
        row_idx = pm.df.index[pm.df['Shot_ID'].astype(str).str.upper() == str(shot_id).upper()].tolist()
        if not row_idx: return None
        row = pm.df.loc[row_idx[0]]

        img_path = row.get('Image_Path')
        if not img_path or pd.isna(img_path) or not os.path.exists(str(img_path)): 
            print(f"No image found for {shot_id}")
            return None

        vid_prompt = row.get('Video_Prompt', '')
        if pd.isna(vid_prompt) or not vid_prompt:
            vid_prompt = row.get('Image_Prompt', '')

        print(f"\n🎬 === START VIDEO GENERATION ===")
        print(f"🎬 Shot ID: {shot_id} | Type: {row['Type']}")
        print(f"🎬 Video Prompt:\n{vid_prompt}\n=================================\n")

        wf = {}
        
        
        if row['Type'] == "Action":
            wf = copy.deepcopy(PRELOADED_WORKFLOW_VID_ACTION)
            if not wf: return None
            
            server_img = comfy.upload_image(img_path)
            if not server_img: return None
            
            needed_duration = row['Duration']
            ltx_frames = get_ltx_frame_count(needed_duration, fps=24)
            
            # Explicit WAN Targeting
            if "102:98" in wf: wf["102:98"]["inputs"]["image"] = server_img
            if "8" in wf: wf["8"]["inputs"]["length"] = ltx_frames
            if "6" in wf: wf["6"]["inputs"]["text"] = vid_prompt # Positive Prompt
            if "70" in wf:
                clean_proj = pm.sanitize_name(pm.current_project)
                wf["70"]["inputs"]["filename_prefix"] = f"{clean_proj}/videos/{shot_id}"

            new_seed = random.randint(1, 100000000000000)
            if "93:10" in wf: wf["93:10"]["inputs"]["noise_seed"] = new_seed
            if "93:12" in wf: wf["93:12"]["inputs"]["noise_seed"] = new_seed
                
        else:
            wf = copy.deepcopy(PRELOADED_WORKFLOW_VID_VOCAL)
            if not wf: return None
            
            vocals_path = pm.get_asset_path_if_exists("vocals.mp3")
            if not vocals_path: return None

            audio = AudioSegment.from_file(vocals_path)
            chunk = audio[row['Start_Time']*1000 : row['End_Time']*1000]
            chunk_path = os.path.join(pm.get_path("audio_chunks"), f"{shot_id}_audio.mp3")
            chunk.export(chunk_path, format="mp3")
            
            server_audio = comfy.upload_image(chunk_path)
            server_img = comfy.upload_image(img_path)
            
            # Explicit LTX Targeting
            if "12" in wf: wf["12"]["inputs"]["audio"] = server_audio
            if "102" in wf: wf["102"]["inputs"]["value"] = row['Duration']
            if "62" in wf: wf["62"]["inputs"]["image"] = server_img
            if "85" in wf: wf["85"]["inputs"]["text"] = vid_prompt # Text Multiline Positive Prompt

        # Universal Fallback Injector for WAN and LTX text prompts
        if "6" in wf: 
            wf["6"]["inputs"]["text"] = vid_prompt
            
        for node_id, node_data in wf.items():
            if isinstance(node_data, dict) and "inputs" in node_data:
                if "text" in node_data["inputs"] and isinstance(node_data["inputs"]["text"], str):
                    class_type = node_data.get("class_type", "")
                    if "TextEncode" in class_type or class_type == "CLIPTextEncode" or "Prompt" in class_type:
                        node_data["inputs"]["text"] = vid_prompt

        resp = comfy.queue_prompt(wf)
        prompt_id = resp['prompt_id']
        comfy.track_progress(prompt_id)
        
        history = comfy.get_history(prompt_id)[prompt_id]
        outputs = history['outputs']
        
        vid_node = None
        for k, v in outputs.items():
            if 'gifs' in v: vid_node = v['gifs']
            elif 'videos' in v: vid_node = v['videos']
            
            if vid_node:
                fname = vid_node[0]['filename']
                sub = vid_node[0]['subfolder']
                ftype = vid_node[0]['type']
                
                save_name = f"{shot_id}_vid_v{int(time.time())}.mp4"
                local_path = os.path.join(pm.get_path("videos"), save_name)
                
                comfy.download_file(fname, sub, ftype, local_path)
                
                pm.df.at[row_idx[0], 'Video_Path'] = local_path
                pm.save_data()
                return local_path
                
    except Exception as e: 
        print(f"Error: {e}")
        return None
    finally:
        comfy.close()
    return None

def advanced_batch_video_generation(mode, target_versions, pm):
    """Handles various video generation modes with clear interruption and missing asset checks."""
    if pm.is_generating:
        yield [], None, "❌ Error: A generation process is already actively running."
        return

    pm.stop_video_generation = False
    pm.is_generating = True
    
    try:
        if pm.current_project: pm.load_project(pm.current_project)
        
        df = pm.df
        if df.empty: 
            yield [], None, "No shots found."
            return

        current_gallery = get_project_videos(pm)
        yield current_gallery, None, f"🚀 Starting video generation ({mode})..."

        if mode == "Generate all Action Shots":
            shot_ids = df[df['Type'] == 'Action']['Shot_ID'].tolist()
        elif mode == "Generate all Vocal Shots":
            shot_ids = df[df['Type'] == 'Vocal']['Shot_ID'].tolist()
        else:
            shot_ids = df['Shot_ID'].tolist()
        
        for shot_id in shot_ids:
            if pm.stop_video_generation: break
            
            row = df[df['Shot_ID'] == shot_id].iloc[0]

            if mode == "Regenerate all Shots":
                vid_dir = pm.get_path("videos")
                if os.path.exists(vid_dir):
                    for f in glob.glob(os.path.join(vid_dir, f"{shot_id}_*.mp4")):
                        try: os.remove(f)
                        except Exception: pass
                current_gallery = get_project_videos(pm)
                
            # Graceful warning for missing images
            if pd.isna(row.get('Image_Path')) or not str(row.get('Image_Path')).strip() or not os.path.exists(str(row.get('Image_Path'))):
                yield current_gallery, None, f"⚠️ Skipped {shot_id}: Missing generated image. Please generate its image first."
                continue

            current_count = get_video_count_for_shot(shot_id, current_gallery)
            
            while current_count < target_versions:
                if pm.stop_video_generation: break
                
                status_msg = f"⏳ Working... Generating Video for {shot_id} (Version {current_count + 1}/{target_versions})"
                yield current_gallery, None, status_msg
                
                new_vid_path = generate_video_for_shot(shot_id, pm)
                
                if new_vid_path:
                    current_gallery = get_project_videos(pm)
                    current_count += 1
                    yield current_gallery, new_vid_path, f"✅ Finished {shot_id}"
                else:
                    yield current_gallery, None, f"❌ Failed to generate video for {shot_id}."
                    break
                    
        if pm.stop_video_generation:
            yield current_gallery, None, "🛑 Generation Stopped by User."
        else:
            yield current_gallery, None, "🎉 Batch Video Generation Complete."
    finally:
        pm.is_generating = False

def assemble_video(full_song_path, pm, fallback_mode=False):
    df = pm.df
    clips = []
    if df.empty: return "No shots to assemble."

    df = df.sort_values(by="Start_Time")
    
    for index, row in df.iterrows():
        vid_path = row.get('Video_Path')
        img_path = row.get('Image_Path')
        dur = row['Duration']
        
        clip = None
        
        if vid_path and pd.notna(vid_path) and os.path.exists(str(vid_path)):
            try:
                clip = VideoFileClip(vid_path)
                if clip.duration > dur: clip = clip.subclip(0, dur)
            except Exception as e:
                print(f"Error loading clip {vid_path}: {e}")
        
        if clip is None and fallback_mode and img_path and pd.notna(img_path) and os.path.exists(str(img_path)):
            try:
                clip = ImageClip(str(img_path)).set_duration(dur)
            except Exception as e:
                print(f"Error loading image fallback {img_path}: {e}")

        if clip is None:
            clip = ColorClip(size=(1280, 720), color=(0,0,0), duration=dur)
            
        clips.append(clip)
            
    if not clips: return "No valid clips found."

    final = concatenate_videoclips(clips, method="compose")
    
    audio_path = full_song_path if (full_song_path and os.path.exists(full_song_path)) else pm.get_asset_path_if_exists("full_song.mp3")
    if not audio_path: audio_path = pm.get_asset_path_if_exists("vocals.mp3")
    
    if audio_path and os.path.exists(audio_path):
        try:
            audio = AudioFileClip(audio_path)
            if audio.duration > final.duration: audio = audio.subclip(0, final.duration)
            final = final.set_audio(audio)
        except Exception as e: print(f"Audio attach failed: {e}")
        
    # Format current total project time and append to the final video filename
    total_seconds = pm.get_current_total_time()
    m, s = divmod(int(total_seconds), 60)
    h, m = divmod(m, 60)
    time_str = f"{h:02d}h{m:02d}m{s:02d}s"
    
    out_path = os.path.join(pm.get_path("renders"), f"final_cut_{time_str}.mp4")
    final.write_videofile(out_path, fps=24, codec='libx264', audio_codec='aac')
    return out_path

# ==========================================
# GRADIO UI
# ==========================================

css = """
/* Enable reliable internal scrolling for specific gallery wrappers */
.scrollable-gallery {
    overflow-y: auto !important;
    max-height: 600px !important;
}
"""

with gr.Blocks(title="Music Video AI Studio", theme=gr.themes.Default(), css=css) as app:
    pm_state = gr.State(ProjectManager)
    
    gr.Markdown("# 🎬 AI Music Video Director")
    current_proj_var = gr.State("")
    
# --- TAB 1: SETUP ---
    with gr.Tab("1. Project & Assets"):
        gr.Markdown("### Create or Load")
        with gr.Row():
            with gr.Column():
                proj_name = gr.Textbox(label="New Project Name", placeholder="MyMusicVideo_v1")
                create_btn = gr.Button("Create New Project")
            with gr.Column():
                with gr.Row():
                    project_dropdown = gr.Dropdown(choices=get_existing_projects(), label="Select Existing Project", interactive=True)
                    refresh_proj_btn = gr.Button("🔄", size="sm")
                with gr.Row():
                    load_btn = gr.Button("Load Selected Project")
                    delete_proj_btn = gr.Button("Delete Selected Project", variant="stop")
        
        proj_status = gr.Textbox(label="System Status", interactive=False)
        gr.Markdown("### Assets")
        with gr.Row():
            # Upgraded audio upload components for better previewing
            vocals_up = gr.Audio(label="Upload Vocals (Audio)", type="filepath")
            song_up = gr.Audio(label="Upload Full Song (Audio)", type="filepath")
            lyrics_in = gr.Textbox(label="Lyrics", lines=5)

# --- TAB 2: STORYBOARD ---
    with gr.Tab("2. Storyboard"):
        with gr.Accordion("Step 1: Timeline Settings", open=True):
            with gr.Row():
                min_silence_sl = gr.Slider(500, 2000, value=700, label="Min Silence (ms)")
                silence_thresh_sl = gr.Slider(-60, -20, value=-45, label="Silence Threshold (dB)")
            with gr.Row():
                shot_mode_drp = gr.Dropdown(["Fixed", "Random"], value="Random", label="Action Shot Mode")
                min_shot_dur = gr.Slider(1, 5, value=2, label="Min Duration (s)")
                max_shot_dur = gr.Slider(1, 5, value=4, label="Max Duration (s)")
            with gr.Row():
                scan_btn = gr.Button("1. Scan Vocals & Build Timeline", variant="primary")
                scan_status = gr.Textbox(label="Build Status", interactive=False)
        
        with gr.Accordion("Step 2: Plot & Concept Generation", open=True):
            with gr.Row():
                llm_dropdown = gr.Dropdown(choices=["qwen3-vl-8b-instruct-abliterated-v2.0"], label="Select LLM Model", interactive=True)
                refresh_llm_btn = gr.Button("🔄", size="sm")
            
            with gr.Row():
                rough_concept_in = gr.Textbox(label="Rough User Concept / Vibe", placeholder="e.g. A cyberpunk rainstorm...", scale=2, lines=5)
                with gr.Column(scale=1):
                    gen_performance_btn = gr.Button("Generate Singer, Band & Venue Desc")
                    performance_desc_in = gr.Textbox(label="Singer, Band, and Venue Description", placeholder="Short description of the singer, band, and venue setup", lines=2)
            
            gen_plot_btn = gr.Button("2. Generate Overarching Plot")
            plot_out = gr.Textbox(label="Overarching Plot", lines=4, interactive=True)
            
            with gr.Accordion("Advanced: Prompt Templates", open=False):
                prompt_template_in = gr.Textbox(value=DEFAULT_CONCEPT_PROMPT, label="Action Shot Visual Prompt Template", lines=4)
                vocal_prompt_template_in = gr.Textbox(value=DEFAULT_VOCAL_PROMPT, label="Vocal Shot Visual Prompt Template", lines=4)
                video_prompt_template_in = gr.Textbox(value=DEFAULT_VIDEO_PROMPT, label="Video Model Prompt Template", lines=3)
            
            with gr.Row():
                gen_concepts_btn = gr.Button("3. Generate Prompts", variant="primary")
                stop_concepts_btn = gr.Button("Stop Generation", variant="stop")
            
            concept_gen_status = gr.Textbox(label="Concept Generation Status", interactive=False)
        
        with gr.Row():
            gr.Markdown("### 📂 Data Management")
            with gr.Row():
                export_csv_btn = gr.Button("Export CSV")
                csv_downloader = gr.File(label="Download Shot List", interactive=False)
            with gr.Row():
                import_csv_btn = gr.UploadButton("Import CSV (Overwrite)", file_types=[".csv"])
                import_status = gr.Textbox(label="Import Status", interactive=False)

        with gr.Row():
            regen_shot_id = gr.Textbox(label="Shot ID to Regenerate", placeholder="S005")
            regen_single_btn = gr.Button("Regenerate Single Shot")
            regen_all_btn = gr.Button("Regenerate All Shots")
        shot_table = gr.Dataframe(headers=REQUIRED_COLUMNS, interactive=True, wrap=True)

# --- TAB 3: IMAGE GENERATION ---
    with gr.Tab("3. Image Generation"):
        selected_img_path = gr.State("")
        
        with gr.Row():
            img_gen_mode_dropdown = gr.Dropdown(choices=["Generate Remaining Shots", "Regenerate all Shots", "Generate all Action Shots", "Generate all Vocal Shots"], value="Generate Remaining Shots", label="Generation Mode")
            img_versions_dropdown = gr.Dropdown(choices=[1, 2, 3, 4], value=1, label="Versions per Shot")
            img_gen_start_btn = gr.Button("Start Generation", variant="primary")
            img_gen_stop_btn = gr.Button("Stop Generation", variant="stop", visible=False)
        
        img_gen_status = gr.Textbox(label="Generation Status (Displays Progress Here)", interactive=False)
        
        with gr.Row():
            with gr.Column(scale=1):
                img_gallery = gr.Gallery(label="Generated Thumbnails", columns=4, elem_classes=["scrollable-gallery"], allow_preview=False, interactive=True)
            
            with gr.Column(scale=1):
                img_large_view = gr.Image(label="Selected Shot", interactive=False)
                with gr.Row():
                    sel_shot_info_img = gr.Textbox(label="Selected Shot ID", interactive=False)
                with gr.Row():
                    gen_vid_from_img_btn = gr.Button("Generate This Video")
                
                with gr.Row():
                    del_img_btn = gr.Button("🗑️ Delete This Version", variant="stop")
                    regen_img_btn = gr.Button("♻️ Regenerate This Shot")

        # --- Tab 3 Events ---
        def on_img_gallery_select(evt: gr.SelectData, proj, pm):
            gal_data = get_project_images(pm, proj)
            if evt.index < len(gal_data):
                fpath = gal_data[evt.index][0]
                fname = os.path.basename(fpath)
                shot_id = fname.split('_')[0] if '_' in fname else "Unknown"
                return fpath, shot_id, fpath 
            return None, "", ""

        img_gallery.select(on_img_gallery_select, inputs=[current_proj_var, pm_state], outputs=[img_large_view, sel_shot_info_img, selected_img_path])
        
        # Swapping Start/Stop Logic
        start_img_evt = img_gen_start_btn.click(
            lambda: (gr.update(visible=False), gr.update(visible=True)), outputs=[img_gen_start_btn, img_gen_stop_btn]
        ).then(
            advanced_batch_image_generation, inputs=[img_gen_mode_dropdown, img_versions_dropdown, pm_state], outputs=[img_gallery, img_large_view, img_gen_status], show_progress="hidden"
        ).then(
            lambda: (gr.update(visible=True), gr.update(visible=False)), outputs=[img_gen_start_btn, img_gen_stop_btn]
        )
        
        img_gen_stop_btn.click(
            stop_gen, inputs=[pm_state], outputs=[img_gen_status], cancels=[start_img_evt]
        ).then(
            lambda: (gr.update(visible=True), gr.update(visible=False)), outputs=[img_gen_start_btn, img_gen_stop_btn]
        )

        def handle_img_delete(path_to_del, proj, pm):
            new_gal, _ = delete_image_file(path_to_del, proj, pm)
            return new_gal, None, "", "" 
            
        del_img_btn.click(handle_img_delete, inputs=[selected_img_path, current_proj_var, pm_state], outputs=[img_gallery, img_large_view, sel_shot_info_img, selected_img_path])

        def handle_regen_img(shot_id_txt, selected_path, proj, pm):
            if pm.is_generating:
                yield gr.update(), gr.update(), "❌ Error: A generation process is already actively running."
                return
            if not shot_id_txt: 
                yield gr.update(), gr.update(), "❌ No Shot ID selected"
                return
                
            pm.is_generating = True
            try:
                if selected_path and os.path.exists(selected_path):
                    try:
                        os.remove(selected_path)
                    except Exception as e:
                        print(f"Could not delete file {selected_path}: {e}")

                yield gr.update(), gr.update(), f"⏳ Regenerating Image for {shot_id_txt}..."
                path = generate_image_for_shot(shot_id_txt, pm)
                yield get_project_images(pm, proj), path, f"✅ Finished regenerating {shot_id_txt}"
            finally:
                pm.is_generating = False

        regen_img_btn.click(handle_regen_img, inputs=[sel_shot_info_img, selected_img_path, current_proj_var, pm_state], outputs=[img_gallery, img_large_view, img_gen_status], show_progress="hidden")

# --- TAB 4: VIDEO GENERATION ---
    with gr.Tab("4. Video Generation"):
        selected_vid_path = gr.State("")
        
        with gr.Row():
            vid_gen_mode_dropdown = gr.Dropdown(choices=["Generate Remaining Shots", "Regenerate all Shots", "Generate all Action Shots", "Generate all Vocal Shots"], value="Generate Remaining Shots", label="Generation Mode")
            vid_versions_dropdown = gr.Dropdown(choices=[1, 2, 3], value=1, label="Versions per Shot")
            vid_gen_start_btn = gr.Button("Start Generation", variant="primary")
            vid_gen_stop_btn = gr.Button("Stop Generation", variant="stop", visible=False)

        vid_gen_status = gr.Textbox(label="Generation Status (Displays Progress Here)", interactive=False)

        with gr.Row():
            with gr.Column(scale=1):
                vid_gallery = gr.Gallery(label="Generated Video Thumbnails", columns=4, elem_classes=["scrollable-gallery"], allow_preview=False, interactive=True)
            
            with gr.Column(scale=1):
                vid_large_view = gr.Video(label="Selected Video", interactive=False)
                with gr.Row():
                    sel_shot_info_vid = gr.Textbox(label="Selected Shot ID", interactive=False)
                
                with gr.Row():
                    del_vid_btn = gr.Button("🗑️ Delete This Video", variant="stop")
                    regen_vid_btn = gr.Button("♻️ Regenerate Video")

        # --- Updated Event to bridge Tab 3 click into Tab 4 updating ---
        def handle_gen_vid_from_tab3(shot_id, proj, pm):
            if pm.is_generating:
                yield gr.update(), "❌ Error: A generation process is already actively running.", gr.update(), gr.update()
                return
            if not shot_id:
                yield gr.update(), "❌ No Shot ID selected", gr.update(), gr.update()
                return

            pm.is_generating = True
            try:
                gen_msg = f"⏳ Generating video for {shot_id}..."
                yield gr.update(value="Video is generating in Tab 4..."), gen_msg, gen_msg, get_project_videos(pm, proj)
                generate_video_for_shot(shot_id, pm)
                
                done_msg = f"✅ Finished generating video for {shot_id}."
                yield gr.update(value="Generate This Video"), done_msg, done_msg, get_project_videos(pm, proj)
            finally:
                pm.is_generating = False

        gen_vid_from_img_btn.click(
            handle_gen_vid_from_tab3, 
            inputs=[sel_shot_info_img, current_proj_var, pm_state], 
            outputs=[gen_vid_from_img_btn, img_gen_status, vid_gen_status, vid_gallery], 
            show_progress="hidden"
        )

        # --- Tab 4 Events ---
        def on_vid_gallery_select(evt: gr.SelectData, proj, pm):
            gal_data = get_project_videos(pm, proj)
            if evt.index < len(gal_data):
                fpath = gal_data[evt.index][0]
                fname = os.path.basename(fpath)
                shot_id = fname.split('_')[0] if '_' in fname else "Unknown"
                return fpath, shot_id, fpath 
            return None, "", ""

        vid_gallery.select(on_vid_gallery_select, inputs=[current_proj_var, pm_state], outputs=[vid_large_view, sel_shot_info_vid, selected_vid_path])
        
        start_vid_evt = vid_gen_start_btn.click(
            lambda: (gr.update(visible=False), gr.update(visible=True)), outputs=[vid_gen_start_btn, vid_gen_stop_btn]
        ).then(
            advanced_batch_video_generation, inputs=[vid_gen_mode_dropdown, vid_versions_dropdown, pm_state], outputs=[vid_gallery, vid_large_view, vid_gen_status], show_progress="hidden"
        ).then(
            lambda: (gr.update(visible=True), gr.update(visible=False)), outputs=[vid_gen_start_btn, vid_gen_stop_btn]
        )
        
        vid_gen_stop_btn.click(
            stop_gen, inputs=[pm_state], outputs=[vid_gen_status], cancels=[start_vid_evt]
        ).then(
            lambda: (gr.update(visible=True), gr.update(visible=False)), outputs=[vid_gen_start_btn, vid_gen_stop_btn]
        )

        def handle_vid_delete(path_to_del, proj, pm):
            new_gal, _ = delete_video_file(path_to_del, proj, pm)
            return new_gal, None, "", "" 
            
        del_vid_btn.click(handle_vid_delete, inputs=[selected_vid_path, current_proj_var, pm_state], outputs=[vid_gallery, vid_large_view, sel_shot_info_vid, selected_vid_path])

        def handle_regen_vid(shot_id_txt, selected_path, proj, pm):
            if pm.is_generating:
                yield gr.update(), gr.update(), "❌ Error: A generation process is already actively running."
                return
            if not shot_id_txt: 
                yield gr.update(), gr.update(), "❌ No Shot ID selected"
                return
                
            pm.is_generating = True
            try:
                if selected_path and os.path.exists(selected_path):
                    try:
                        os.remove(selected_path)
                    except Exception as e:
                        print(f"Could not delete file {selected_path}: {e}")

                yield gr.update(), gr.update(), f"⏳ Regenerating Video for {shot_id_txt}..."
                path = generate_video_for_shot(shot_id_txt, pm)
                yield get_project_videos(pm, proj), path, f"✅ Finished regenerating {shot_id_txt}"
            finally:
                pm.is_generating = False

        regen_vid_btn.click(handle_regen_vid, inputs=[sel_shot_info_vid, selected_vid_path, current_proj_var, pm_state], outputs=[vid_gallery, vid_large_view, vid_gen_status], show_progress="hidden")

# --- TAB 5: ASSEMBLY ---
    with gr.Tab("5. Assembly"):
        with gr.Row():
            assemble_btn = gr.Button("Assemble Final Video (Strictly Videos)", variant="secondary")
            assemble_current_btn = gr.Button("Assemble with Current Assets (Videos > Images > Black)", variant="primary")
        final_video_out = gr.Video(label="Final Cut")
        
        assemble_btn.click(lambda s, pm: assemble_video(get_file_path(s), pm, fallback_mode=False), inputs=[song_up, pm_state], outputs=[final_video_out])
        assemble_current_btn.click(lambda s, pm: assemble_video(get_file_path(s), pm, fallback_mode=True), inputs=[song_up, pm_state], outputs=[final_video_out])

# ==========================================
# GLOBAL LOGIC & WIRING
# ==========================================

    def handle_create(name, pm):
        msg = pm.create_project(name)
        return msg, gr.Dropdown(choices=get_existing_projects()), pm.sanitize_name(name)

    def handle_load(name, pm):
        msg, df = pm.load_project(name)
        lyrics = pm.get_lyrics()
        v_path = pm.get_asset_path_if_exists("vocals.mp3")
        s_path = pm.get_asset_path_if_exists("full_song.mp3")
        settings = pm.load_project_settings()
        
        gal_imgs = get_project_images(pm, name)
        gal_vids = get_project_videos(pm, name)
        
        return (
            msg, df, lyrics, v_path, s_path, 
            settings.get("min_silence", 700), settings.get("silence_thresh", -45), 
            settings.get("shot_mode", "Random"), settings.get("min_dur", 2), settings.get("max_dur", 4),
            settings.get("llm_model", "qwen3-vl-8b-instruct-abliterated-v2.0"), settings.get("rough_concept", ""), 
            settings.get("plot", ""), 
            settings.get("prompt_template", DEFAULT_CONCEPT_PROMPT),
            settings.get("vocal_prompt_template", DEFAULT_VOCAL_PROMPT),
            settings.get("performance_desc", ""), 
            settings.get("video_prompt_template", DEFAULT_VIDEO_PROMPT),
            name,
            gal_imgs, gal_vids, gr.Button(value="Start Generation", variant="primary")
        )

    def handle_delete_project(name, pm):
        if not name: return "No project selected.", gr.update()
        path = os.path.join(pm.base_dir, name)
        if os.path.exists(path):
            try:
                shutil.rmtree(path)
                if pm.current_project == name:
                    pm.current_project = None
                    pm.df = pd.DataFrame(columns=REQUIRED_COLUMNS)
                return f"Deleted project '{name}'.", gr.update(choices=get_existing_projects(), value=None)
            except Exception as e:
                return f"Error deleting project: {e}", gr.update()
        return "Project not found.", gr.update()

    # Auto Save functions mapping
    def auto_save_lyrics(proj_name, text, pm):
        if proj_name:
            pm.current_project = proj_name
            pm.save_lyrics(text)

    def auto_save_files(proj_name, v_file, s_file, pm):
        if proj_name:
            v_src = get_file_path(v_file)
            s_src = get_file_path(s_file)
            if v_src: pm.save_asset(v_src, "vocals.mp3")
            if s_src: pm.save_asset(s_src, "full_song.mp3")

    def auto_save_tab2(proj_name, min_sil, sil_thresh, mode, min_d, max_d, llm, concept, plot, template, vocal_template, performance_d, vid_temp, pm):
        if proj_name:
            pm.current_project = proj_name
            settings = {
                "min_silence": min_sil, "silence_thresh": sil_thresh, "shot_mode": mode,
                "min_dur": min_d, "max_dur": max_d, "llm_model": llm,
                "rough_concept": concept, "plot": plot, "prompt_template": template,
                "vocal_prompt_template": vocal_template,
                "performance_desc": performance_d, "video_prompt_template": vid_temp
            }
            pm.save_project_settings(settings)

    create_btn.click(handle_create, inputs=[proj_name, pm_state], outputs=[proj_status, project_dropdown, current_proj_var])
    refresh_proj_btn.click(lambda: gr.Dropdown(choices=get_existing_projects()), outputs=project_dropdown)

    load_btn.click(
        handle_load, 
        inputs=[project_dropdown, pm_state], 
        outputs=[
            proj_status, shot_table, lyrics_in, vocals_up, song_up, 
            min_silence_sl, silence_thresh_sl, shot_mode_drp, min_shot_dur, max_shot_dur,
            llm_dropdown, rough_concept_in, plot_out, prompt_template_in, vocal_prompt_template_in, 
            performance_desc_in, video_prompt_template_in,
            current_proj_var,
            img_gallery, vid_gallery, img_gen_start_btn 
        ]
    )

    delete_proj_btn.click(handle_delete_project, inputs=[project_dropdown, pm_state], outputs=[proj_status, project_dropdown])

    # Tie listeners into the UI for dynamic, manual-free saving (outputs=[] removed)
    lyrics_in.change(auto_save_lyrics, inputs=[current_proj_var, lyrics_in, pm_state])
    
    for file_comp in [vocals_up, song_up]:
        file_comp.upload(auto_save_files, inputs=[current_proj_var, vocals_up, song_up, pm_state])
        file_comp.clear(auto_save_files, inputs=[current_proj_var, vocals_up, song_up, pm_state])
        
    t2_inputs = [current_proj_var, min_silence_sl, silence_thresh_sl, shot_mode_drp, min_shot_dur, max_shot_dur, llm_dropdown, rough_concept_in, plot_out, prompt_template_in, vocal_prompt_template_in, performance_desc_in, video_prompt_template_in, pm_state]
    for tab2_comp in [min_silence_sl, silence_thresh_sl, shot_mode_drp, min_shot_dur, max_shot_dur, llm_dropdown, rough_concept_in, plot_out, prompt_template_in, vocal_prompt_template_in, performance_desc_in, video_prompt_template_in]:
        tab2_comp.change(auto_save_tab2, inputs=t2_inputs)
        
    export_csv_btn.click(lambda pm: pm.export_csv(), inputs=[pm_state], outputs=csv_downloader)
    import_csv_btn.upload(lambda f, pm: pm.import_csv(f), inputs=[import_csv_btn, pm_state], outputs=[import_status, shot_table])
    
    # Save manual user edits inside the Gradio dataframe back to the CSV dynamically
    def save_manual_df_edits(new_df, pm):
        if pm.current_project:
            pm.df = new_df
            pm.save_data()
            
    shot_table.change(save_manual_df_edits, inputs=[shot_table, pm_state])

    refresh_llm_btn.click(lambda: gr.Dropdown(choices=LLMBridge().get_models()), outputs=llm_dropdown)
    
    def run_scan(v_file, p_name, m_sil, s_thr, s_mode, min_d, max_d, pm):
        yield "⏳ Initializing...", pm.df
        if not p_name: 
            yield "❌ Error: No project selected.", pm.df
            return
        pm.current_project = p_name
        
        final_v_path = get_file_path(v_file) or pm.get_asset_path_if_exists("vocals.mp3")
        
        if not final_v_path or not os.path.exists(final_v_path):
             yield "❌ Error: No vocals file found.", pm.df
             return
             
        yield "⏳ Detecting silence and building timeline (this may take a moment)...", pm.df
        df = scan_vocals_advanced(final_v_path, p_name, m_sil, s_thr, s_mode, min_d, max_d, pm)
        
        if df.empty:
            yield "❌ Error: Could not build timeline. Check audio file.", pm.df
        else:
            yield "✅ Timeline Built Successfully!", df

    scan_btn.click(run_scan, inputs=[vocals_up, current_proj_var, min_silence_sl, silence_thresh_sl, shot_mode_drp, min_shot_dur, max_shot_dur, pm_state], outputs=[scan_status, shot_table])
    
    gen_performance_btn.click(generate_performance_description, inputs=[rough_concept_in, plot_out, llm_dropdown], outputs=performance_desc_in)
    gen_plot_btn.click(generate_overarching_plot, inputs=[rough_concept_in, lyrics_in, llm_dropdown, pm_state], outputs=plot_out)
    
    gen_concepts_btn.click(generate_concepts_logic, inputs=[plot_out, prompt_template_in, vocal_prompt_template_in, llm_dropdown, rough_concept_in, performance_desc_in, video_prompt_template_in, gr.State(None), pm_state], outputs=[shot_table, concept_gen_status])
    stop_concepts_btn.click(stop_gen, inputs=[pm_state], outputs=[concept_gen_status]) 
    regen_single_btn.click(generate_concepts_logic, inputs=[plot_out, prompt_template_in, vocal_prompt_template_in, llm_dropdown, rough_concept_in, performance_desc_in, video_prompt_template_in, regen_shot_id, pm_state], outputs=[shot_table, concept_gen_status])
    regen_all_btn.click(generate_concepts_logic, inputs=[plot_out, prompt_template_in, vocal_prompt_template_in, llm_dropdown, rough_concept_in, performance_desc_in, video_prompt_template_in, gr.State("ALL"), pm_state], outputs=[shot_table, concept_gen_status])

if __name__ == "__main__":
    try:
        keyboard.add_hotkey('ctrl+r', restart_application)
        print("⌨️  Hotkey Ctrl+R registered for restarting the application. (Ensure your terminal has focus to use)")
    except Exception as e:
        print(f"⚠️ Could not register hotkey 'ctrl+r'. Run script as admin or ensure 'keyboard' module is installed. Error: {e}")
        
    app.launch()