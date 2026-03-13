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
from moviepy.editor import VideoFileClip, AudioFileClip, ColorClip, concatenate_videoclips
from datetime import datetime
import shutil
import math
import threading
import time
import random
import re
import glob
import io

# ==========================================
# CONFIGURATION
# ==========================================
COMFY_URL = "127.0.0.1:8188"
LM_STUDIO_URL = "http://127.0.0.1:1234/v1"

# WORKFLOW FILES
WORKFLOW_IMG = "ZImage_Poster_API.json"
WORKFLOW_VID_ACTION = "wan22_i2v_v17_jakes version slow motion api version.json"
WORKFLOW_VID_VOCAL = "011426-LTX2-AudioSync-i2v-Ver2-Jakes Version API.json"

# CHANGED: Replaced Concept/Visual_Prompt with Image_Prompt
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

DEFAULT_VIDEO_PROMPT = (
    "Generate a prompt based on this image description for a video model to encompass 5 seconds of action. "
    "Describe the scene, camera motion, emotions, lighting and performance only. "
    "Pay special attention to the camera's motion. Do not include any additional notes or titles in your description."
)

# ==========================================
# SYSTEM UTILITIES
# ==========================================

def restart_application():
    """Restarts the current python process."""
    print("♻️ Restarting application...")
    python = sys.executable
    os.execl(python, python, *sys.argv)

def snap_to_frame(seconds, fps=24):
    """Rounds seconds to the nearest 1/24th of a second frame boundary."""
    frame_dur = 1.0 / fps
    return round(seconds / frame_dur) * frame_dur

def get_ltx_frame_count(target_seconds, fps=24):
    """Returns the nearest valid LTX frame count (8n + 1)."""
    min_frames = math.ceil(target_seconds * fps)
    valid_buckets = [1 + 8*k for k in range(60)] 
    for bucket in valid_buckets:
        if bucket >= min_frames:
            return bucket
    return valid_buckets[-1]

def get_ltx_duration(seconds, fps=24):
    """Calculates the duration in seconds for the nearest LTX-compatible frame count."""
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
            print(f"Warning: Could not fetch models from LM Studio: {e}")
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

    def sanitize_name(self, name):
        return re.sub(r'[\\/*?:"<>|]', "", name).strip().replace(" ", "_")

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
        
        with open(os.path.join(path, "lyrics.txt"), "w") as f:
            f.write("")
        return f"Project '{clean_name}' created."

    def load_project(self, name):
        path = os.path.join(self.base_dir, name)
        csv_path = os.path.join(path, "shot_list.csv")
        if os.path.exists(csv_path):
            self.df = pd.read_csv(csv_path)
            # Ensure columns exist if loading old projects
            for col in REQUIRED_COLUMNS:
                if col not in self.df.columns:
                    self.df[col] = ""
            self.current_project = name
            return f"Loaded '{name}'", self.df
        return "Project not found.", pd.DataFrame()

    def import_csv(self, file_obj):
        if not self.current_project:
            return "No project loaded.", pd.DataFrame()
        
        try:
            new_df = pd.read_csv(file_obj.name)
            
            # Validation
            missing_cols = [c for c in REQUIRED_COLUMNS if c not in new_df.columns]
            if missing_cols:
                # Add missing columns instead of rejecting
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

pm = ProjectManager()

# ==========================================
# LOGIC: TIMELINE & CONCEPTS
# ==========================================

def get_existing_projects():
    if not os.path.exists(pm.base_dir): return []
    projects = [d for d in os.listdir(pm.base_dir) if os.path.isdir(os.path.join(pm.base_dir, d))]
    return sorted(projects)

def scan_vocals_advanced(vocals_file_path, project_name, min_silence, silence_thresh, shot_mode, min_dur, max_dur):
    if not project_name or not vocals_file_path or not os.path.exists(vocals_file_path): return pd.DataFrame()

    try:
        audio = AudioSegment.from_mp3(vocals_file_path)
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

    def get_random_action_dur():
        raw = random.uniform(min_dur, max_dur)
        return get_ltx_duration(raw, fps)

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

def generate_overarching_plot(concept, lyrics, llm_model):
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

def generate_singer_description(concept, plot, llm_model):
    llm = LLMBridge()
    sys_prompt = "You are a casting director."
    user_prompt = (
        f"Concept: {concept}\nPlot: {plot}\n\n"
        "Task: Describe the physical appearance and style of the lead singer for this video. "
        "Keep it concise (1-2 sentences). Focus on visual descriptors."
    )
    return llm.query(sys_prompt, user_prompt, llm_model)

def generate_concepts_logic(overarching_plot, prompt_template, llm_model, rough_concept, singer_desc, video_prompt_template, specific_shot_id=None, progress=gr.Progress()):
    llm = LLMBridge()
    df = pm.df
    pm.stop_generation = False
    if df.empty: return df

    mask = (df['Shot_ID'] == specific_shot_id) if specific_shot_id else (df['Image_Prompt'].isna() | (df['Image_Prompt'] == ""))
    indices_to_process = df[mask].index.tolist()
    
    for index in progress.tqdm(indices_to_process, desc="Generating Prompts"):
        if pm.stop_generation: 
            print("Stop signal received.")
            break
            
        row = df.loc[index]
        
        # Determine Context
        prev_shot_text = "None (Start of video)"
        if index > 0:
             # Look at previous Image_Prompt
             prev_shot_text = df.loc[index - 1, 'Image_Prompt']
             if pd.isna(prev_shot_text): prev_shot_text = "N/A"

        # === 1. Generate Single Image Prompt ===
        if row['Type'] == 'Vocal':
            prompt_input = (
                f"Global Concept: {rough_concept}\n"
                f"Singer Description: {singer_desc}\n"
                f"Shot Info: Vocal Shot at {row['Start_Time']:.1f}s.\n"
                "Task: Write a detailed plain English description of this shot for an AI image generator. "
                "Focus on the singer performing."
            )
        else:
            # Action Logic: Pass Plot + Previous Shot + User Template
            filled_prompt = prompt_template.replace("{plot}", overarching_plot)\
                .replace("{type}", row['Type'])\
                .replace("{start}", f"{row['Start_Time']:.1f}")\
                .replace("{duration}", f"{row['Duration']:.1f}")\
                .replace("{prev_shot}", prev_shot_text)
            prompt_input = filled_prompt
        
        # CHANGED: Request detailed description, no sentence limit
        image_prompt_text = llm.query("Describe the scene in high detail using plain English.", prompt_input, llm_model)
        
        # === 2. Generate Video Prompt ===
        # Derived from Image_Prompt
        vid_prompt_in = f"Visual Description: {image_prompt_text}\n\nTask: {video_prompt_template}"
        final_vid_prompt = llm.query("You are a video prompt expert.", vid_prompt_in, llm_model)

        df.at[index, 'Image_Prompt'] = image_prompt_text
        df.at[index, 'Video_Prompt'] = final_vid_prompt
        
        pm.df = df
        pm.save_data()
        
    return df

def stop_gen():
    pm.stop_generation = True
    pm.stop_image_generation = True
    pm.stop_video_generation = True
    return "🛑 Stopping... Waiting for current task to complete..."

# ==========================================
# LOGIC: IMAGE GENERATION & GALLERY
# ==========================================

def get_project_images(project_name=None):
    proj = project_name if project_name else pm.current_project
    if not proj: return []

    img_dir = os.path.join(pm.base_dir, proj, "images")
    if not os.path.exists(img_dir): return []
    
    files = sorted(glob.glob(os.path.join(img_dir, "*.png")), key=os.path.getmtime, reverse=True)
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
        if os.path.basename(path).startswith(f"{shot_id}_"):
            count += 1
    return count

def generate_image_for_shot(shot_id):
    comfy = ComfyBridge()
    if not comfy.connect(): return None
    
    try:
        row_idx = pm.df.index[pm.df['Shot_ID'] == shot_id].tolist()
        if not row_idx: return None
        row = pm.df.loc[row_idx[0]]
    except IndexError: return None
        
    # CHANGED: Use Image_Prompt instead of Visual_Prompt
    prompt_text = row['Image_Prompt']
    if not prompt_text: return None

    try:
        with open(WORKFLOW_IMG, 'r') as f: wf = json.load(f)
    except FileNotFoundError: return None
        
    if "6" in wf: wf["6"]["inputs"]["text"] = prompt_text
    if "60" in wf:
        wf["60"]["inputs"]["custom_ratio"] = True
        wf["60"]["inputs"]["custom_aspect_ratio"] = "16:9"
    if "73" in wf:
        clean_proj = pm.sanitize_name(pm.current_project)
        wf["73"]["inputs"]["filename_prefix"] = f"{clean_proj}/{shot_id}"

    try:
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
    return None

def batch_image_generation(target_versions, progress=gr.Progress()):
    pm.stop_image_generation = False
    
    if pm.current_project: pm.load_project(pm.current_project)
    
    df = pm.df
    if df.empty: 
        yield [], None, "No shots found."
        return

    # CHANGED: Check Image_Prompt column
    if df['Image_Prompt'].isna().any() or (df['Image_Prompt'] == "").any():
        missing_count = len(df[df['Image_Prompt'].isna() | (df['Image_Prompt'] == "")])
        pm.stop_image_generation = True
        yield get_project_images(), None, f"⚠️ STOPPED: {missing_count} shots are missing prompts. Go to Tab 2 first."
        return

    current_gallery = get_project_images()
    yield current_gallery, None, "Starting generation..."

    shot_ids = df['Shot_ID'].tolist()
    
    for shot_id in progress.tqdm(shot_ids, desc="Processing Shots"):
        if pm.stop_image_generation: break
            
        current_count = get_image_count_for_shot(shot_id, current_gallery)
        
        while current_count < target_versions:
            if pm.stop_image_generation: break
            
            status_msg = f"🎨 Generating {shot_id} (Version {current_count + 1}/{target_versions})..."
            yield current_gallery, None, status_msg
            
            new_img_path = generate_image_for_shot(shot_id)
            
            if new_img_path:
                current_gallery = get_project_images()
                current_count += 1
                yield current_gallery, new_img_path, f"✅ Finished {shot_id}"
            else:
                break
                
    if pm.stop_image_generation:
        yield current_gallery, None, "🛑 Generation Stopped by User."
    else:
        yield current_gallery, None, "🎉 Batch Generation Complete."

def delete_image_file(path, project_name):
    if not path or not os.path.exists(path):
        return get_project_images(project_name), None
    try:
        os.remove(path)
    except Exception as e:
        print(f"Error deleting file: {e}")
    return get_project_images(project_name), None

def update_gen_btn_state(project_name):
    imgs = get_project_images(project_name)
    if len(imgs) > 0:
        return gr.Button(value="Resume Generation", variant="primary")
    return gr.Button(value="Generate All", variant="primary")

# ==========================================
# LOGIC: VIDEO GENERATION
# ==========================================

def get_project_videos(project_name=None):
    proj = project_name if project_name else pm.current_project
    if not proj: return []

    vid_dir = os.path.join(pm.base_dir, proj, "videos")
    if not os.path.exists(vid_dir): return []
    
    files = sorted(glob.glob(os.path.join(vid_dir, "*.mp4")), key=os.path.getmtime, reverse=True)
    gallery_data = []
    
    for f in files:
        fname = os.path.basename(f)
        parts = fname.split("_")
        caption = f"{parts[0]}" if len(parts) >= 2 else fname
        gallery_data.append((f, caption))
        
    return gallery_data

def delete_video_file(path, project_name):
    if not path or not os.path.exists(path):
        return get_project_videos(project_name), None
    try:
        os.remove(path)
    except Exception as e:
        print(f"Error deleting file: {e}")
    return get_project_videos(project_name), None

def get_video_count_for_shot(shot_id, vid_list):
    count = 0
    for path, caption in vid_list:
        if os.path.basename(path).startswith(f"{shot_id}_"):
            count += 1
    return count

def generate_video_for_shot(shot_id):
    comfy = ComfyBridge()
    if not comfy.connect(): return None
    
    try:
        row_idx = pm.df.index[pm.df['Shot_ID'] == shot_id].tolist()
        if not row_idx: return None
        row = pm.df.loc[row_idx[0]]
    except IndexError: return None

    img_path = row['Image_Path']
    if not img_path or not os.path.exists(img_path): 
        print(f"No image found for {shot_id}")
        return None

    # Use VIDEO prompt, fallback to Image_Prompt if missing
    vid_prompt = row.get('Video_Prompt', '')
    if pd.isna(vid_prompt) or not vid_prompt:
        vid_prompt = row['Image_Prompt']

    wf = {}
    
    if row['Type'] == "Action":
        try:
            with open(WORKFLOW_VID_ACTION, 'r') as f: wf = json.load(f)
        except FileNotFoundError: 
            print("Workflow file not found")
            return None
        
        server_img = comfy.upload_image(img_path)
        if not server_img: return None
        
        needed_duration = row['Duration']
        ltx_frames = get_ltx_frame_count(needed_duration, fps=24)
        
        if "102:98" in wf: wf["102:98"]["inputs"]["image"] = server_img
        if "6" in wf: wf["6"]["inputs"]["text"] = vid_prompt
        if "8" in wf: wf["8"]["inputs"]["length"] = ltx_frames
        if "70" in wf:
            clean_proj = pm.sanitize_name(pm.current_project)
            wf["70"]["inputs"]["filename_prefix"] = f"{clean_proj}/videos/{shot_id}"

        new_seed = random.randint(1, 100000000000000)
        if "93:10" in wf: wf["93:10"]["inputs"]["noise_seed"] = new_seed
        if "93:12" in wf: wf["93:12"]["inputs"]["noise_seed"] = new_seed
            
    else:
        try:
            with open(WORKFLOW_VID_VOCAL, 'r') as f: wf = json.load(f)
        except FileNotFoundError: return None
        
        vocals_path = pm.get_asset_path_if_exists("vocals.mp3")
        if not vocals_path: return None

        audio = AudioSegment.from_mp3(vocals_path)
        chunk = audio[row['Start_Time']*1000 : row['End_Time']*1000]
        chunk_path = os.path.join(pm.get_path("audio_chunks"), f"{shot_id}_audio.mp3")
        chunk.export(chunk_path, format="mp3")
        
        server_audio = comfy.upload_image(chunk_path)
        server_img = comfy.upload_image(img_path)
        
        if "12" in wf: wf["12"]["inputs"]["audio"] = server_audio
        if "102" in wf: wf["102"]["inputs"]["value"] = row['Duration']
        if "62" in wf: wf["62"]["inputs"]["image"] = server_img

    try:
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
    return None

def batch_video_generation(target_versions=1, progress=gr.Progress()):
    pm.stop_video_generation = False
    
    if pm.current_project: pm.load_project(pm.current_project)
    
    df = pm.df
    if df.empty: 
        yield [], None, "No shots found."
        return

    current_gallery = get_project_videos()
    yield current_gallery, None, "Starting video generation..."

    shot_ids = df['Shot_ID'].tolist()
    
    for shot_id in progress.tqdm(shot_ids, desc="Processing Videos"):
        if pm.stop_video_generation: break
        
        row = df[df['Shot_ID'] == shot_id].iloc[0]
        if not row['Image_Path'] or not os.path.exists(row['Image_Path']):
            continue

        current_count = get_video_count_for_shot(shot_id, current_gallery)
        
        while current_count < target_versions:
            if pm.stop_video_generation: break
            
            status_msg = f"🎬 Generating Video for {shot_id} (Version {current_count + 1}/{target_versions})..."
            yield current_gallery, None, status_msg
            
            new_vid_path = generate_video_for_shot(shot_id)
            
            if new_vid_path:
                current_gallery = get_project_videos()
                current_count += 1
                yield current_gallery, new_vid_path, f"✅ Finished {shot_id}"
            else:
                break
                
    if pm.stop_video_generation:
        yield current_gallery, None, "🛑 Generation Stopped by User."
    else:
        yield current_gallery, None, "🎉 Batch Video Generation Complete."

def assemble_video(full_song_path):
    df = pm.df
    clips = []
    if df.empty: return "No shots to assemble."

    df = df.sort_values(by="Start_Time")
    
    for index, row in df.iterrows():
        vid_path = row['Video_Path']
        dur = row['Duration']
        
        if vid_path and os.path.exists(vid_path):
            try:
                clip = VideoFileClip(vid_path)
                if clip.duration > dur: clip = clip.subclip(0, dur)
                clips.append(clip)
            except Exception as e:
                print(f"Error loading clip {vid_path}: {e}")
                clips.append(ColorClip(size=(1280, 720), color=(0,0,0), duration=dur))
        else:
            clips.append(ColorClip(size=(1280, 720), color=(0,0,0), duration=dur))
            
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
        
    out_path = os.path.join(pm.get_path("renders"), "final_cut.mp4")
    final.write_videofile(out_path, fps=24, codec='libx264', audio_codec='aac')
    return out_path

# ==========================================
# GRADIO UI
# ==========================================

with gr.Blocks(title="Music Video AI Studio", theme=gr.themes.Default()) as app:
    gr.Markdown("# 汐 AI Music Video Director")
    current_proj_var = gr.State("")
    
# --- TAB 1: SETUP ---
    with gr.Tab("1. Project & Assets"):
        with gr.Row():
            gr.Markdown("### 🛠️ Developer Tools")
            restart_btn = gr.Button("♻️ RESTART APP", variant="stop", size="sm")
        gr.Markdown("### Create or Load")
        with gr.Row():
            with gr.Column():
                proj_name = gr.Textbox(label="New Project Name", placeholder="MyMusicVideo_v1")
                create_btn = gr.Button("Create New Project")
            with gr.Column():
                with gr.Row():
                    project_dropdown = gr.Dropdown(choices=get_existing_projects(), label="Select Existing Project", interactive=True)
                    refresh_proj_btn = gr.Button("売", size="sm")
                load_btn = gr.Button("Load Selected Project")
        
        proj_status = gr.Textbox(label="System Status", interactive=False)
        gr.Markdown("### Assets")
        with gr.Row():
            vocals_up = gr.File(label="Upload Vocals (MP3)", file_types=[".mp3"])
            song_up = gr.File(label="Upload Full Song (MP3)", file_types=[".mp3"])
            lyrics_in = gr.Textbox(label="Lyrics", lines=5)
        save_proj_btn = gr.Button("沈 Save Project Changes", variant="secondary")

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
            scan_btn = gr.Button("1. Scan Vocals & Build Timeline", variant="primary")
        
        with gr.Accordion("Step 2: Plot & Concept Generation", open=True):
            with gr.Row():
                llm_dropdown = gr.Dropdown(choices=["qwen3-vl-8b-instruct-abliterated-v2.0"], label="Select LLM Model", interactive=True)
                refresh_llm_btn = gr.Button("売", size="sm")
            
            with gr.Row():
                rough_concept_in = gr.Textbox(label="Rough User Concept / Vibe", placeholder="e.g. A cyberpunk rainstorm...", scale=2)
                with gr.Column(scale=1):
                    gen_singer_btn = gr.Button("Generate Singer Desc")
                    singer_desc_in = gr.Textbox(label="Singer Description", placeholder="Short description of the singer's look/style", lines=2)
            
            gen_plot_btn = gr.Button("2. Generate Overarching Plot")
            plot_out = gr.Textbox(label="Overarching Plot", lines=4, interactive=True)
            
            with gr.Accordion("Advanced: Prompt Templates", open=False):
                prompt_template_in = gr.Textbox(value=DEFAULT_CONCEPT_PROMPT, label="Shot Visual Prompt Template", lines=4)
                video_prompt_template_in = gr.Textbox(value=DEFAULT_VIDEO_PROMPT, label="Video Model Prompt Template", lines=3)
            
            with gr.Row():
                gen_concepts_btn = gr.Button("3. Generate Prompts", variant="primary")
                stop_concepts_btn = gr.Button("Stop Generation", variant="stop")
            
            save_tab2_btn = gr.Button("Save Tab 2 Settings", variant="secondary")
        
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
        shot_table = gr.Dataframe(headers=REQUIRED_COLUMNS, interactive=True, wrap=True)

# --- TAB 3: IMAGE GENERATION ---
    with gr.Tab("3. Image Generation"):
        selected_img_path = gr.State("")
        
        with gr.Row():
            img_gen_all_btn = gr.Button("Generate All", variant="primary")
            img_stop_btn = gr.Button("Stop", variant="stop")
            img_versions_dropdown = gr.Dropdown(choices=[1, 2, 3, 4], value=1, label="Versions per Shot")
        
        img_gen_status = gr.Textbox(label="Generation Status", interactive=False)
        
        with gr.Row():
            with gr.Column(scale=1):
                img_gallery = gr.Gallery(label="Generated Thumbnails", columns=4, height=600, allow_preview=False, interactive=True)
            
            with gr.Column(scale=1):
                img_large_view = gr.Image(label="Selected Shot", interactive=False)
                with gr.Row():
                    sel_shot_info_img = gr.Textbox(label="Selected Shot ID", interactive=False)
                
                with gr.Row():
                    del_img_btn = gr.Button("🗑️ Delete This Version", variant="stop")
                    regen_img_btn = gr.Button("♻️ Regenerate This Shot")

        # --- Tab 3 Events ---
        def on_img_gallery_select(evt: gr.SelectData, proj):
            gal_data = get_project_images(proj)
            if evt.index < len(gal_data):
                fpath = gal_data[evt.index][0]
                fname = os.path.basename(fpath)
                shot_id = fname.split('_')[0] if '_' in fname else "Unknown"
                return fpath, shot_id, fpath 
            return None, "", ""

        img_gallery.select(on_img_gallery_select, inputs=[current_proj_var], outputs=[img_large_view, sel_shot_info_img, selected_img_path])
        img_gen_all_btn.click(batch_image_generation, inputs=[img_versions_dropdown], outputs=[img_gallery, img_large_view, img_gen_status])
        img_stop_btn.click(stop_gen, outputs=[img_gen_status])

        def handle_img_delete(path_to_del, proj):
            new_gal, _ = delete_image_file(path_to_del, proj)
            return new_gal, None, "", "" 
            
        del_img_btn.click(handle_img_delete, inputs=[selected_img_path, current_proj_var], outputs=[img_gallery, img_large_view, sel_shot_info_img, selected_img_path])

        def handle_regen_img(shot_id_txt):
            if not shot_id_txt: return None
            path = generate_image_for_shot(shot_id_txt)
            return get_project_images(), path

        regen_img_btn.click(handle_regen_img, inputs=[sel_shot_info_img], outputs=[img_gallery, img_large_view])

# --- TAB 4: VIDEO GENERATION ---
    with gr.Tab("4. Video Generation"):
        selected_vid_path = gr.State("")
        
        with gr.Row():
            vid_gen_all_btn = gr.Button("Generate All Videos", variant="primary")
            vid_stop_btn = gr.Button("Stop", variant="stop")
            vid_versions_dropdown = gr.Dropdown(choices=[1, 2, 3], value=1, label="Versions per Shot")
        
        with gr.Row():
             single_vid_id = gr.Textbox(label="Single Shot ID", placeholder="S001")
             gen_single_vid_btn = gr.Button("Generate Single Video")

        vid_gen_status = gr.Textbox(label="Generation Status", interactive=False)

        with gr.Row():
            with gr.Column(scale=1):
                vid_gallery = gr.Gallery(label="Generated Video Thumbnails", columns=4, height=600, allow_preview=False, interactive=True)
            
            with gr.Column(scale=1):
                vid_large_view = gr.Video(label="Selected Video", interactive=False)
                with gr.Row():
                    sel_shot_info_vid = gr.Textbox(label="Selected Shot ID", interactive=False)
                
                with gr.Row():
                    del_vid_btn = gr.Button("🗑️ Delete This Video", variant="stop")
                    regen_vid_btn = gr.Button("♻️ Regenerate Video")

        # --- Tab 4 Events ---
        def on_vid_gallery_select(evt: gr.SelectData, proj):
            gal_data = get_project_videos(proj)
            if evt.index < len(gal_data):
                fpath = gal_data[evt.index][0]
                fname = os.path.basename(fpath)
                shot_id = fname.split('_')[0] if '_' in fname else "Unknown"
                return fpath, shot_id, fpath 
            return None, "", ""

        vid_gallery.select(on_vid_gallery_select, inputs=[current_proj_var], outputs=[vid_large_view, sel_shot_info_vid, selected_vid_path])
        vid_gen_all_btn.click(batch_video_generation, inputs=[vid_versions_dropdown], outputs=[vid_gallery, vid_large_view, vid_gen_status])
        vid_stop_btn.click(stop_gen, outputs=[vid_gen_status])
        
        def handle_single_vid_gen(shot_id):
             path = generate_video_for_shot(shot_id)
             return get_project_videos(), path
        
        gen_single_vid_btn.click(handle_single_vid_gen, inputs=[single_vid_id], outputs=[vid_gallery, vid_large_view])

        def handle_vid_delete(path_to_del, proj):
            new_gal, _ = delete_video_file(path_to_del, proj)
            return new_gal, None, "", "" 
            
        del_vid_btn.click(handle_vid_delete, inputs=[selected_vid_path, current_proj_var], outputs=[vid_gallery, vid_large_view, sel_shot_info_vid, selected_vid_path])

        def handle_regen_vid(shot_id_txt):
            if not shot_id_txt: return None
            path = generate_video_for_shot(shot_id_txt)
            return get_project_videos(), path

        regen_vid_btn.click(handle_regen_vid, inputs=[sel_shot_info_vid], outputs=[vid_gallery, vid_large_view])

# --- TAB 5: ASSEMBLY ---
    with gr.Tab("5. Assembly"):
        with gr.Row():
            assemble_btn = gr.Button("Assemble Final Video", variant="primary")
        final_video_out = gr.Video(label="Final Cut")
        assemble_btn.click(lambda s: assemble_video(s.name if s else None), inputs=[song_up], outputs=[final_video_out])

# ==========================================
# GLOBAL LOGIC & WIRING
# ==========================================

    def handle_create(name):
        msg = pm.create_project(name)
        return msg, gr.Dropdown(choices=get_existing_projects()), pm.sanitize_name(name)

    def handle_load(name):
        msg, df = pm.load_project(name)
        lyrics = pm.get_lyrics()
        v_path = pm.get_asset_path_if_exists("vocals.mp3")
        s_path = pm.get_asset_path_if_exists("full_song.mp3")
        settings = pm.load_project_settings()
        
        gal_imgs = get_project_images(name)
        gal_vids = get_project_videos(name)
        gen_btn = update_gen_btn_state(name)
        
        return (
            msg, df, lyrics, v_path, s_path, 
            settings.get("min_silence", 700), settings.get("silence_thresh", -45), 
            settings.get("shot_mode", "Random"), settings.get("min_dur", 2), settings.get("max_dur", 4),
            settings.get("llm_model", "qwen3-vl-8b-instruct-abliterated-v2.0"), settings.get("rough_concept", ""), 
            settings.get("plot", ""), 
            settings.get("prompt_template", DEFAULT_CONCEPT_PROMPT),
            settings.get("singer_desc", ""), 
            settings.get("video_prompt_template", DEFAULT_VIDEO_PROMPT),
            name,
            gal_imgs, gal_vids, gen_btn 
        )

    def handle_save_changes(project_state_name, lyrics_text, v_file, s_file):
        if not project_state_name: return "No project active.", None, None
        pm.current_project = project_state_name
        pm.save_lyrics(lyrics_text)
        v_path = pm.save_asset(v_file.name, "vocals.mp3") if v_file and hasattr(v_file, 'name') else None
        s_path = pm.save_asset(s_file.name, "full_song.mp3") if s_file and hasattr(s_file, 'name') else None
        return f"Saved assets for '{project_state_name}'.", v_path, s_path

    def handle_save_tab2(project_state_name, min_sil, sil_thresh, mode, min_d, max_d, llm, concept, plot, template, singer_d, vid_temp):
        if not project_state_name: return "No active project."
        pm.current_project = project_state_name
        settings = {
            "min_silence": min_sil, "silence_thresh": sil_thresh, "shot_mode": mode,
            "min_dur": min_d, "max_dur": max_d, "llm_model": llm,
            "rough_concept": concept, "plot": plot, "prompt_template": template,
            "singer_desc": singer_d, "video_prompt_template": vid_temp
        }
        return pm.save_project_settings(settings)

    create_btn.click(handle_create, inputs=proj_name, outputs=[proj_status, project_dropdown, current_proj_var])
    refresh_proj_btn.click(lambda: gr.Dropdown(choices=get_existing_projects()), outputs=project_dropdown)

    load_btn.click(
        handle_load, 
        inputs=project_dropdown, 
        outputs=[
            proj_status, shot_table, lyrics_in, vocals_up, song_up, 
            min_silence_sl, silence_thresh_sl, shot_mode_drp, min_shot_dur, max_shot_dur,
            llm_dropdown, rough_concept_in, plot_out, prompt_template_in, 
            singer_desc_in, video_prompt_template_in,
            current_proj_var,
            img_gallery, vid_gallery, img_gen_all_btn 
        ]
    )

    save_proj_btn.click(handle_save_changes, inputs=[current_proj_var, lyrics_in, vocals_up, song_up], outputs=[proj_status, vocals_up, song_up])
    save_tab2_btn.click(handle_save_tab2, inputs=[current_proj_var, min_silence_sl, silence_thresh_sl, shot_mode_drp, min_shot_dur, max_shot_dur, llm_dropdown, rough_concept_in, plot_out, prompt_template_in, singer_desc_in, video_prompt_template_in], outputs=proj_status)
    
    export_csv_btn.click(lambda: pm.export_csv(), outputs=csv_downloader)
    import_csv_btn.upload(lambda f: pm.import_csv(f), inputs=import_csv_btn, outputs=[import_status, shot_table])

    refresh_llm_btn.click(lambda: gr.Dropdown(choices=LLMBridge().get_models()), outputs=llm_dropdown)
    
    def run_scan(v_file, p_name, m_sil, s_thr, s_mode, min_d, max_d):
        if not p_name: return pd.DataFrame()
        pm.current_project = p_name
        final_v_path = v_file.name if (v_file and hasattr(v_file, 'name')) else pm.get_asset_path_if_exists("vocals.mp3")
        return scan_vocals_advanced(final_v_path, p_name, m_sil, s_thr, s_mode, min_d, max_d)

    scan_btn.click(run_scan, inputs=[vocals_up, current_proj_var, min_silence_sl, silence_thresh_sl, shot_mode_drp, min_shot_dur, max_shot_dur], outputs=shot_table)
    
    gen_singer_btn.click(generate_singer_description, inputs=[rough_concept_in, plot_out, llm_dropdown], outputs=singer_desc_in)
    gen_plot_btn.click(generate_overarching_plot, inputs=[rough_concept_in, lyrics_in, llm_dropdown], outputs=plot_out)
    
    gen_concepts_btn.click(generate_concepts_logic, inputs=[plot_out, prompt_template_in, llm_dropdown, rough_concept_in, singer_desc_in, video_prompt_template_in], outputs=shot_table)
    
    stop_concepts_btn.click(stop_gen, outputs=[proj_status]) 
    
    regen_single_btn.click(generate_concepts_logic, inputs=[plot_out, prompt_template_in, llm_dropdown, rough_concept_in, singer_desc_in, video_prompt_template_in, regen_shot_id], outputs=shot_table)
    restart_btn.click(restart_application, outputs=None)

if __name__ == "__main__":
    app.launch()