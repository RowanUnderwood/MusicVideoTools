import gradio as gr
import pandas as pd
import os
import sys
import json
import requests
import uuid
import asyncio
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
import copy
import subprocess
import keyboard  # Requires: pip install keyboard

# ==========================================
# WINDOWS ASYNCIO PATCH (Fixes WinError 10054)
# ==========================================
if sys.platform.lower() == "win32" or os.name.lower() == "nt":
    try:
        from asyncio.proactor_events import _ProactorBasePipeTransport
        def silence_event_loop_closed(func):
            def wrapper(self, *args, **kwargs):
                try:
                    return func(self, *args, **kwargs)
                except (RuntimeError, ConnectionResetError):
                    pass
            return wrapper
        _ProactorBasePipeTransport._call_connection_lost = silence_event_loop_closed(_ProactorBasePipeTransport._call_connection_lost)
    except ImportError:
        pass

# ==========================================
# CONFIGURATION
# ==========================================
LTX_BASE_URL = "http://127.0.0.1:8000/api"
LM_STUDIO_URL = "http://127.0.0.1:1234/v1"

REQUIRED_COLUMNS = [
    "Shot_ID", "Type", 
    "Start_Time", "End_Time", "Duration", 
    "Start_Frame", "End_Frame", "Total_Frames",
    "Lyrics", "Video_Prompt", "Video_Path", "All_Video_Paths", "Status"
]

RESOLUTION_MAP = {
    "540p": (960, 540),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "1440p": (2560, 1440)
}

DEFAULT_CONCEPT_PROMPT = (
    "Context: The overarching plot is: {plot}\n"
    "Previous Shot Visual: {prev_shot}\n"
    "Current Shot Info: Timestamp {start}s, Duration {duration}s, Type: {type}.\n"
    "Task: Write a highly detailed visual description for this video shot encompassing the action. "
    "Describe the scene, camera motion, emotions, lighting and performance only. "
    "Pay special attention to the camera's motion. Do not include any additional notes or titles."
)

LTX_SYSTEM_PROMPT = """You are an expert cinematography AI director writing video generation prompts. Adhere strictly to these rules:

1. Establish the shot: Use cinematography terms that match the preferred film genre. Include aspects like scale or specific category characteristics.
2. Set the scene: Describe lighting conditions, color palette, surface textures, and atmosphere to shape the mood.
3. Describe the action: Write the core action as a natural sequence, flowing from beginning to end.
4. Define your character(s): Include age, hairstyle, clothing, and distinguishing details. Express emotions through physical cues.
5. Identify camera movement(s): Specify when the view should shift and how. Include how subjects or objects appear after the camera motion.
6. Format: Keep your prompt in a SINGLE flowing paragraph.
7. Grammar: Use present tense verbs to describe movement and action.
8. Detail scale: Match your detail to the shot scale (Closeups need more precise detail than wide shots).
9. Camera focus: When describing camera movement, focus on the camera’s relationship to the subject.
10. Length: Write 4 to 8 descriptive sentences to cover all key aspects."""

# Global cache for ffprobe frame counts to speed up preview loading in Tab 3
FRAME_COUNT_CACHE = {}

# ==========================================
# GLOBAL MEMORY
# ==========================================
GLOBAL_SETTINGS_FILE = "global_settings.json"

def get_global_llm():
    try:
        if os.path.exists(GLOBAL_SETTINGS_FILE):
            with open(GLOBAL_SETTINGS_FILE, "r") as f:
                return json.load(f).get("last_llm", None)
    except:
        pass
    return None

def save_global_llm(model_id):
    try:
        with open(GLOBAL_SETTINGS_FILE, "w") as f:
            json.dump({"last_llm": model_id}, f)
    except:
        pass

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
    """
    Calculates LTX Desktop-compliant frame counts (locked to whole seconds do not change this).
    1s = 25f, 2s = 49f, 3s = 73f, 4s = 97f, 5s = 121f.
    """
    target_int = int(math.ceil(target_seconds))
    
    if target_int < 1: 
        target_int = 1
    if target_int > 5: 
        target_int = 5
        
    total_frames = target_int * fps
    backend_frames = round((total_frames - 1) / 8) * 8 + 1
    return max(backend_frames, 9)

def get_ltx_duration(seconds, fps=24):
    """
    Returns the true floating-point timeline duration of the locked integer frame counts.
    """
    frames = get_ltx_frame_count(seconds, fps)
    return frames / fps

def format_time(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}h{m:02d}m{s:02d}s"

# ==========================================
# BACKEND UTILITIES
# ==========================================

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
            "max_tokens": 8000
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
        self.stop_video_generation = False
        self.is_generating = False 
        
        # Time Tracking Variables
        self.total_time_spent = 0
        self.session_start_time = None

    def sanitize_name(self, name):
        return re.sub(r'[\\/*?:"<>|]', "", name).strip().replace(" ", "_")
        
    def get_current_total_time(self):
        if self.session_start_time and self.current_project:
            elapsed = time.time() - self.session_start_time
            self.session_start_time = time.time()  
            self.total_time_spent += elapsed
            
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
        
        folders = ["assets", "audio_chunks", "videos", "renders", "cutting_room"]
        
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
            
            settings = self.load_project_settings()
            self.total_time_spent = settings.get("total_time_spent", 0)
            self.session_start_time = time.time()
            
            sync_video_directory(self)
            return f"Loaded '{name}'", self.df
        return "Project not found.", pd.DataFrame()

    def import_csv(self, file_obj):
        if not self.current_project:
            return "No project loaded.", self.df
        
        try:
            new_df = pd.read_csv(get_file_path(file_obj))
            
            if "Shot_ID" not in new_df.columns:
                return "❌ Error: Uploaded CSV is missing the 'Shot_ID' column.", self.df

            if len(new_df) != len(self.df):
                return f"❌ Error: Uploaded CSV has {len(new_df)} rows, but current project has {len(self.df)} rows.", self.df

            # Set indexes to ensure reliable alignment even if the user sorted the CSV
            new_df = new_df.set_index("Shot_ID")
            curr_df = self.df.set_index("Shot_ID")
            
            missing_shots = set(curr_df.index) - set(new_df.index)
            if missing_shots:
                return f"❌ Error: CSV is missing required Shot IDs: {', '.join(missing_shots)}", self.df

            if 'Type' not in new_df.columns:
                 return "❌ Error: CSV is missing 'Type' column.", self.df
                 
            type_mismatch = new_df['Type'] != curr_df['Type']
            if type_mismatch.any():
                bad_shots = curr_df[type_mismatch].index.tolist()
                return f"❌ Error: Shot 'Type' mismatch for shots: {', '.join(map(str, bad_shots))}", self.df

            if 'Video_Prompt' in new_df.columns:
                curr_df['Video_Prompt'] = new_df['Video_Prompt']
                self.df = curr_df.reset_index()
                self.save_data()
                return "✅ CSV Uploaded & Verified. Prompts successfully updated.", self.df
            else:
                return "❌ Error: 'Video_Prompt' column not found in uploaded CSV.", self.df

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
        if not self.current_project: return None
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
        
        existing_settings = self.load_project_settings()
        existing_settings.update(settings_dict)
        existing_settings["total_time_spent"] = self.get_current_total_time()
        
        path = os.path.join(self.base_dir, self.current_project, "settings.json")
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(existing_settings, f, indent=4)
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
# LOGIC: DIRECTORY SYNC
# ==========================================

def sync_video_directory(pm):
    if not pm.current_project: return "No project loaded."
    vid_dir = pm.get_path("videos")
    if not os.path.exists(vid_dir): return "No videos directory."
    
    mp4s = glob.glob(os.path.join(vid_dir, "*.mp4"))
    shot_vids = {}
    
    for v in mp4s:
        fname = os.path.basename(v)
        shot_id = fname.split("_")[0].upper()
        if shot_id not in shot_vids: shot_vids[shot_id] = []
        shot_vids[shot_id].append(v)
        
    if "All_Video_Paths" not in pm.df.columns:
        pm.df["All_Video_Paths"] = ""
        
    for idx, row in pm.df.iterrows():
        sid = str(row.get("Shot_ID", "")).upper()
        if sid in shot_vids:
            vids = sorted(shot_vids[sid], key=os.path.getmtime, reverse=True)
            pm.df.at[idx, "All_Video_Paths"] = ",".join(vids)
            
            curr_path_raw = row.get("Video_Path", "")
            curr_path = "" if pd.isna(curr_path_raw) else str(curr_path_raw)
            
            if not curr_path or not os.path.exists(curr_path) or curr_path not in vids:
                pm.df.at[idx, "Video_Path"] = vids[0]
                pm.df.at[idx, "Status"] = "Done"
        else:
            pm.df.at[idx, "All_Video_Paths"] = ""
            
            curr_path_raw = row.get("Video_Path", "")
            curr_path = "" if pd.isna(curr_path_raw) else str(curr_path_raw)
            
            if not curr_path or not os.path.exists(curr_path):
                pm.df.at[idx, "Video_Path"] = ""
                pm.df.at[idx, "Status"] = "Pending"
                
    pm.save_data()
    return "Directory sync complete."

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
    
    MIN_LTX_DUR = get_ltx_duration(1.0, fps) 

    def create_row(sType, start, end, current_count):
        dur = end - start
        start_frame = round(start * fps)
        end_frame = round(end * fps)
        total_frames = end_frame - start_frame
        
        return {
            "Shot_ID": f"S{current_count:03d}",
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
        
        gap = voc_start - current_cursor
        
        while gap >= MIN_LTX_DUR:
            max_safe_int = int(math.floor(gap))
            if max_safe_int < 1: 
                break 
            
            chosen_raw = min_dur if shot_mode == "Fixed" else random.uniform(min_dur, max_dur)
            chosen_int = int(math.ceil(chosen_raw))
            
            if chosen_int > max_safe_int: chosen_int = max_safe_int
            if chosen_int > 5: chosen_int = 5
            
            actual_dur = get_ltx_duration(chosen_int, fps)
            
            if actual_dur > gap: 
                break
                
            new_rows.append(create_row("Action", current_cursor, current_cursor + actual_dur, shot_counter))
            shot_counter += 1
            current_cursor += actual_dur
            gap = voc_start - current_cursor

        vocal_req_dur = voc_end - current_cursor
        
        while vocal_req_dur > 0:
            if vocal_req_dur > 5.0:
                chosen_int = 5
            else:
                chosen_int = int(math.ceil(vocal_req_dur))
                if chosen_int < 1: chosen_int = 1
                
            actual_dur = get_ltx_duration(chosen_int, fps)
            
            new_rows.append(create_row("Vocal", current_cursor, current_cursor + actual_dur, shot_counter))
            shot_counter += 1
            current_cursor += actual_dur
            vocal_req_dur = voc_end - current_cursor

    remaining_time = total_duration - current_cursor
    while remaining_time >= MIN_LTX_DUR:
        max_safe_int = int(math.floor(remaining_time))
        if max_safe_int < 1: break
        
        chosen_raw = min_dur if shot_mode == "Fixed" else random.uniform(min_dur, max_dur)
        chosen_int = int(math.ceil(chosen_raw))
        
        if chosen_int > max_safe_int: chosen_int = max_safe_int
        if chosen_int > 5: chosen_int = 5
            
        actual_dur = get_ltx_duration(chosen_int, fps)
        if actual_dur > remaining_time: break
        
        new_rows.append(create_row("Action", current_cursor, current_cursor + actual_dur, shot_counter))
        shot_counter += 1
        current_cursor += actual_dur
        remaining_time = total_duration - current_cursor
        
    if remaining_time > 0.1:
        chosen_int = max(1, min(int(math.ceil(remaining_time)), 5))
        actual_dur = get_ltx_duration(chosen_int, fps)
        new_rows.append(create_row("Action", current_cursor, current_cursor + actual_dur, shot_counter))

    new_df = pd.DataFrame(new_rows)
    for col in REQUIRED_COLUMNS:
        if col not in new_df.columns: new_df[col] = ""
            
    pm.df = new_df
    pm.save_data()
    return pm.df

def generate_overarching_plot(concept, lyrics, llm_model, pm):
    yield "⏳ Generating overarching plot... (Please wait)"
    llm = LLMBridge()
    df = pm.df
    if df.empty: 
        yield "Error: Timeline is empty."
        return

    timeline_str = ""
    for idx, row in df.iterrows():
        if row['Type'] == 'Vocal':
            timeline_str += f"[{row['Start_Time']:.2f}s - {row['End_Time']:.2f}s: SINGING]\n"
    
    sys_prompt = "You are a creative writer for music videos."
    user_prompt = (
        f"Rough Concept: {concept}\n\nLyrics:\n{lyrics}\n\nTimeline:\n{timeline_str}\n\n"
        "Task: Write a cohesive linear plot summary for this video (max 300 words)."
    )
    yield llm.query(sys_prompt, user_prompt, llm_model)

def generate_performance_description(concept, plot, llm_model):
    yield "⏳ Generating performance description... (Please wait)"
    llm = LLMBridge()
    sys_prompt = "You are a casting director and set designer."
    user_prompt = (
        f"Concept: {concept}\nPlot: {plot}\n\n"
        "Task: Describe the physical appearance and style of the lead singer, specificaly for an AI video generation model. Describe a close-up shot and the microphone.  Do not include any details about the character that would be out of view in a close-up shot."
        "Keep it concise (2-3 sentences)."
    )
    yield llm.query(sys_prompt, user_prompt, llm_model)

def generate_concepts_logic(overarching_plot, prompt_template, llm_model, rough_concept, performance_desc, specific_shot_id, pm):
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
        mask = (df['Video_Prompt'].isna() | (df['Video_Prompt'] == ""))
        
    indices_to_process = df[mask].index.tolist()
    
    yield df, f"🚀 Starting generation for {len(indices_to_process)} shots..."
    time.sleep(0.1) 
    
    for count, index in enumerate(indices_to_process, 1):
        if pm.stop_generation: 
            yield df, "🛑 Stopped. Waiting for current task to complete..."
            break
            
        row = df.loc[index]
        yield df, f"⏳ Generating ({count}/{len(indices_to_process)}): Prompts for {row['Shot_ID']}..."
        
        if row['Type'] == 'Vocal':
            final_vid_prompt = performance_desc
        else:
            loc_pos = df.index.get_loc(index)
            if loc_pos > 0:
                prev_index = df.index[loc_pos - 1]
                prev_shot_text = df.loc[prev_index, 'Video_Prompt']
                if pd.isna(prev_shot_text): prev_shot_text = "N/A"
            else:
                prev_shot_text = "None (Start of video)"

            filled_prompt = prompt_template.replace("{plot}", overarching_plot)\
                .replace("{type}", row['Type'])\
                .replace("{start}", f"{row['Start_Time']:.1f}")\
                .replace("{duration}", f"{row['Duration']:.1f}")\
                .replace("{prev_shot}", prev_shot_text)
            
            final_vid_prompt = llm.query(LTX_SYSTEM_PROMPT, filled_prompt, llm_model)

        df.at[index, 'Video_Prompt'] = final_vid_prompt
        pm.df = df
        pm.save_data()
        
    if not pm.stop_generation:
        yield df, "🎉 Concept Generation Complete!"

def stop_gen(pm):
    pm.stop_generation = True
    pm.stop_video_generation = True
    return "🛑 Stopping... Waiting for current task to complete..."

# ==========================================
# LOGIC: STORY EXPORTER
# ==========================================
def generate_story_file(pm):
    if not pm.current_project or pm.df.empty: return None
    story_content = ""
    for _, row in pm.df.iterrows():
        sid = row.get("Shot_ID", "Unknown")
        prompt = row.get("Video_Prompt", "No prompt generated.")
        story_content += f"Shot {sid}:\n{prompt}\n\n"
    
    path = os.path.join(pm.base_dir, pm.current_project, "story.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(story_content)
    return path

# ==========================================
# LOGIC: VIDEO GENERATION (LTX)
# ==========================================

def get_project_videos(pm, project_name=None):
    proj = project_name if project_name else pm.current_project
    if not proj: return []

    vid_dir = os.path.join(pm.base_dir, proj, "videos")
    if not os.path.exists(vid_dir): return []
    
    files = glob.glob(os.path.join(vid_dir, "*.mp4"))
    
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
        
        if f in FRAME_COUNT_CACHE:
            caption = FRAME_COUNT_CACHE[f]
        else:
            try:
                cmd = [
                    "ffprobe", "-v", "error", "-select_streams", "v:0",
                    "-count_frames", "-show_entries", "stream=nb_read_frames",
                    "-of", "default=nokey=1:noprint_wrappers=1", f
                ]
                output = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
                if output and output.isdigit():
                    caption = f"{caption} ({output} frames)"
                    FRAME_COUNT_CACHE[f] = caption
                else:
                    caption = f"{caption} (Error reading frames)"
                    FRAME_COUNT_CACHE[f] = caption
            except Exception as e:
                caption = f"{caption} (Error)"
                FRAME_COUNT_CACHE[f] = caption
            
        gallery_data.append((f, caption))
        
    return gallery_data

def delete_video_file(path, project_name, pm):
    if not path or not os.path.exists(path):
        return get_project_videos(pm, project_name), None
    try:
        os.remove(path)
        if path in FRAME_COUNT_CACHE:
            del FRAME_COUNT_CACHE[path]
        sync_video_directory(pm)
    except Exception as e:
        print(f"Error deleting file: {e}")
    return get_project_videos(pm, project_name), None

def get_video_count_for_shot(shot_id, vid_list):
    count = 0
    for path, caption in vid_list:
        if os.path.basename(path).upper().startswith(f"{str(shot_id).upper()}_"):
            count += 1
    return count

def generate_video_for_shot(shot_id, resolution, vocal_mode, pm):
    row_idx = pm.df.index[pm.df['Shot_ID'].astype(str).str.upper() == str(shot_id).upper()].tolist()
    if not row_idx:
        yield None, "Error: Shot not found in timeline."
        return
        
    row = pm.df.loc[row_idx[0]]
    vid_prompt = str(row.get('Video_Prompt', ''))
    
    if row.get('Type') == "Vocal" and vocal_mode == "Use Singer/Band Description":
        settings = pm.load_project_settings()
        perf_desc = settings.get("performance_desc", "")
        if perf_desc:
            vid_prompt = perf_desc

    if pd.isna(vid_prompt) or not vid_prompt.strip():
        yield None, "Error: Missing Video Prompt."
        return

    print(f"\n🎬 === START VIDEO GENERATION (LTX) ===")
    print(f"🎬 Shot ID: {shot_id} | Type: {row['Type']}")
    print(f"🎬 Video Prompt:\n{vid_prompt}\n=================================\n")

    payload = {
        "prompt": vid_prompt,
        "negativePrompt": "blurry, distorted, low quality, artifacts, watermark",
        "model": "pro",
        "resolution": resolution, 
        "aspectRatio": "16:9",
        "duration": str(row['Duration']),
        "fps": "24",
        "cameraMotion": "none",
        "audio": "false"
    }

    if row['Type'] == "Vocal":
        vocals_path = pm.get_asset_path_if_exists("vocals.mp3")
        if not vocals_path:
            yield None, "Error: Missing vocals file for vocal shot."
            return

        try:
            audio = AudioSegment.from_file(vocals_path)
            start_ms = round(float(row['Start_Time']) * 1000)
            end_ms = round(float(row['End_Time']) * 1000)
            
            chunk = audio[start_ms : end_ms]
            
            expected_len_ms = end_ms - start_ms
            if len(chunk) < expected_len_ms:
                deficit = expected_len_ms - len(chunk)
                silence_pad = AudioSegment.silent(duration=deficit)
                chunk = chunk + silence_pad
                
            chunk_path = os.path.join(pm.get_path("audio_chunks"), f"{shot_id}_audio.mp3")
            chunk.export(chunk_path, format="mp3")
            
            payload["audio"] = "true"
            payload["audioPath"] = os.path.abspath(chunk_path)
            
        except Exception as e:
            print(f"❌ AUDIO ERROR for {shot_id}: {e}")
            yield None, f"Error processing audio: {str(e)}"
            return

    result_container = {}

    def worker():
        try:
            resp = requests.post(f"{LTX_BASE_URL}/generate", json=payload)
            resp.raise_for_status()
            result_container['response'] = resp.json()
        except requests.exceptions.RequestException as e:
            err_msg = str(e)
            if e.response is not None:
                err_msg += f" - {e.response.text}"
            result_container['error'] = err_msg

    t = threading.Thread(target=worker)
    t.start()

    while t.is_alive():
        time.sleep(1)
        try:
            prog_resp = requests.get(f"{LTX_BASE_URL}/generation/progress", timeout=2)
            if prog_resp.status_code == 200:
                data = prog_resp.json()
                status_text = f"LTX Progress - Status: {data.get('status')} | Phase: {data.get('phase')} | {data.get('progress')}%"
                yield None, status_text
        except requests.exceptions.RequestException:
            pass 

    t.join()

    if 'error' in result_container:
        print(f"❌ GENERATION FAILED: {result_container['error']}")
        pm.df.at[row_idx[0], 'Status'] = 'Error'
        pm.save_data()
        yield None, f"Error: {result_container['error']}"
        return

    video_path = result_container['response'].get('video_path')
    if video_path and os.path.exists(video_path):
        save_name = f"{shot_id}_vid_v{int(time.time())}.mp4"
        local_path = os.path.join(pm.get_path("videos"), save_name)
        shutil.copy(video_path, local_path)
        
        pm.df.at[row_idx[0], 'Video_Path'] = local_path
        pm.df.at[row_idx[0], 'Status'] = 'Done'
        pm.save_data()
        yield local_path, "Done"
    else:
        pm.df.at[row_idx[0], 'Status'] = 'Error'
        pm.save_data()
        yield None, "Error: Completed but no valid video path returned."

def advanced_batch_video_generation(mode, target_versions, resolution, vocal_mode, pm):
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
        elif mode == "Generate Remaining Shots":
            completed_ids = {
                os.path.basename(p).split('_')[0].upper()
                for p, _ in get_project_videos(pm)
            }
            shot_ids = [sid for sid in df['Shot_ID'].tolist()
                        if str(sid).upper() not in completed_ids]
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

            if pd.isna(row.get('Video_Prompt')) or not str(row.get('Video_Prompt')).strip():
                yield current_gallery, None, f"⚠️ Skipped {shot_id}: Missing Video Prompt."
                continue

            current_count = get_video_count_for_shot(shot_id, current_gallery)
            
            while current_count < target_versions:
                if pm.stop_video_generation: break
                
                new_vid_path = None
                vid_generator = generate_video_for_shot(shot_id, resolution, vocal_mode, pm)
                
                for path, msg in vid_generator:
                    if pm.stop_video_generation: break
                    if path is None:
                        yield current_gallery, None, f"⏳ {shot_id} (Ver {current_count + 1}/{target_versions}): {msg}"
                    else:
                        new_vid_path = path

                if pm.stop_video_generation: break
                
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
        sync_video_directory(pm)
        pm.is_generating = False

def assemble_video(full_song_path, resolution, pm, fallback_mode=False):
    df = pm.df
    clips = []
    clips_to_close = [] 
    if df.empty: return "No shots to assemble."

    df = df.sort_values(by="Start_Time")
    expected_cursor = 0.0
    
    target_size = RESOLUTION_MAP.get(resolution, (1920, 1080))
    
    for index, row in df.iterrows():
        vid_path = row.get('Video_Path')
        if vid_path and pd.notna(vid_path) and os.path.exists(str(vid_path)):
            try:
                temp_clip = VideoFileClip(str(vid_path))
                target_size = tuple(temp_clip.size)
                temp_clip.close()
                break
            except Exception:
                pass

    for index, row in df.iterrows():
        vid_path = row.get('Video_Path')
        dur = float(row['Duration'])
        start_time = float(row['Start_Time'])
        snapped_dur = round(dur * 24) / 24 
        clip = None
        
        gap = round((start_time - expected_cursor) * 24) / 24
        if gap > 0.05:
            pad = ColorClip(size=target_size, color=(0,0,0), duration=gap).set_fps(24)
            clips.append(pad)
            clips_to_close.append(pad)
        
        if vid_path and pd.notna(vid_path) and os.path.exists(str(vid_path)):
            try:
                clip = VideoFileClip(str(vid_path)).without_audio().set_fps(24)
                
                if clip.duration > snapped_dur: 
                    clip = clip.subclip(0, snapped_dur)
                clip = clip.set_duration(snapped_dur)
                
                if tuple(clip.size) != tuple(target_size):
                    clip = clip.resize(newsize=target_size)
                    
            except Exception as e:
                print(f"Error loading clip {vid_path}: {e}")

        if clip is None:
            if fallback_mode:
                clip = ColorClip(size=target_size, color=(0,0,0), duration=snapped_dur).set_fps(24)
            else:
                for c in clips_to_close: c.close()
                return f"Error: Missing or corrupt video for shot at {start_time}s. Assembly stopped (Strict Mode)."
            
        if clip is not None:
            clips.append(clip)
            clips_to_close.append(clip)
            
        expected_cursor = start_time + snapped_dur

    if not clips: return "No valid clips found."

    final = concatenate_videoclips(clips, method="chain")
    audio = None
    
    audio_path = full_song_path if (full_song_path and os.path.exists(full_song_path)) else pm.get_asset_path_if_exists("full_song.mp3")
    if not audio_path: audio_path = pm.get_asset_path_if_exists("vocals.mp3")
    
    if audio_path and os.path.exists(audio_path):
        try:
            audio = AudioFileClip(audio_path)
            if audio.duration > final.duration: audio = audio.subclip(0, final.duration)
            final = final.set_audio(audio)
        except Exception as e: print(f"Audio attach failed: {e}")
        
    total_seconds = pm.get_current_total_time()
    time_str = format_time(total_seconds)
    
    out_path = os.path.join(pm.get_path("renders"), f"final_cut_{time_str}.mp4")
    
    final.write_videofile(
        out_path, fps=24, codec='libx264', audio_codec='aac',
        temp_audiofile=os.path.join(pm.get_path("renders"), "temp_audio.m4a"),
        remove_temp=True,
        ffmpeg_params=["-ar", "44100"]
    )
    
    final.close()
    if audio is not None:
        try: audio.close()
        except: pass
    for c in clips_to_close:
        try: c.close()
        except: pass
        
    return out_path

# ==========================================
# GRADIO UI
# ==========================================

css = """
.scrollable-gallery {
    overflow-y: auto !important;
    max-height: 600px !important;
}
"""

with gr.Blocks(title="Music Video AI Studio", theme=gr.themes.Default(), css=css) as app:
    pm_state = gr.State(ProjectManager()) 
    
    gr.Markdown("# 🎬 AI Music Video Director (LTX Engine)")
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
        
        with gr.Row():
            proj_status = gr.Textbox(label="System Status", interactive=False)
            time_spent_disp = gr.Textbox(label="Total Project Time", interactive=False) 

        gr.Markdown("### Assets")
        with gr.Row():
            vocals_up = gr.Audio(label="Upload Vocals (Audio)", type="filepath")
            song_up = gr.Audio(label="Upload Full Song (Audio)", type="filepath")
            lyrics_in = gr.Textbox(label="Lyrics", lines=5)

# --- TAB 2: STORYBOARD ---
    with gr.Tab("2. Storyboard") as tab2_ui:
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
                avail_models = LLMBridge().get_models()
                last_model = get_global_llm()
                if not last_model:
                    last_model = avail_models[0] if avail_models else "qwen3-vl-8b-instruct-abliterated-v2.0"
                    
                llm_dropdown = gr.Dropdown(choices=avail_models, value=last_model, label="Select LLM Model", interactive=True, allow_custom_value=True)
                refresh_llm_btn = gr.Button("🔄", size="sm")
                
                llm_dropdown.change(save_global_llm, inputs=[llm_dropdown])
            
            with gr.Row():
                rough_concept_in = gr.Textbox(label="Rough User Concept / Vibe", placeholder="e.g. A cyberpunk rainstorm...", scale=2, lines=5)
                with gr.Column(scale=1):
                    gen_performance_btn = gr.Button("Generate Singer, Band & Venue Desc")
                    performance_desc_in = gr.Textbox(label="Singer, Band, and Venue Description (Also used as Prompt for Vocal Shots)", placeholder="Short description of the singer, band, and venue setup", lines=2)
            
            gen_plot_btn = gr.Button("2. Generate Overarching Plot")
            plot_out = gr.Textbox(label="Overarching Plot", lines=4, interactive=True)
            
            with gr.Accordion("Advanced: Prompt Templates", open=False):
                prompt_template_in = gr.Textbox(value=DEFAULT_CONCEPT_PROMPT, label="Action Shot Prompt Template", lines=4)
            
            with gr.Row():
                gen_concepts_btn = gr.Button("3. Generate Video Prompts", variant="primary")
                stop_concepts_btn = gr.Button("Stop Generation", variant="stop")
            
            concept_gen_status = gr.Textbox(label="Concept Generation Status", interactive=False)
        
        with gr.Row():
            gr.Markdown("### 📂 Data Management")
            with gr.Row():
                export_csv_btn = gr.Button("Export CSV")
                csv_downloader = gr.File(label="Download Shot List", interactive=False)
            with gr.Row():
                download_story_btn = gr.Button("Download Story (.txt)")
                story_downloader = gr.File(label="Story Text File", interactive=False)
            with gr.Row():
                import_csv_btn = gr.UploadButton("Import CSV (Update Prompts)", file_types=[".csv"])
                import_status = gr.Textbox(label="Import Status", interactive=False)

        with gr.Row():
            regen_shot_id = gr.Textbox(label="Shot ID to Regenerate", placeholder="S005")
            regen_single_btn = gr.Button("Regenerate Single Shot")
            regen_all_btn = gr.Button("Regenerate All Shots")
        shot_table = gr.Dataframe(headers=REQUIRED_COLUMNS, interactive=True, wrap=True, type="pandas")

# --- TAB 3: VIDEO GENERATION ---
    with gr.Tab("3. Video Generation"):
        selected_vid_path = gr.State("")
        
        with gr.Row():
            vid_gen_mode_dropdown = gr.Dropdown(choices=["Generate Remaining Shots", "Regenerate all Shots", "Generate all Action Shots", "Generate all Vocal Shots"], value="Generate Remaining Shots", label="Generation Mode")
            vid_versions_dropdown = gr.Dropdown(choices=[1, 2, 3, 4, 5], value=1, label="Versions per Shot")
            vid_resolution_dropdown = gr.Dropdown(choices=["540p", "720p", "1080p", "1440p"], value="1080p", label="Resolution")
            vid_vocal_prompt_mode = gr.Dropdown(choices=["Use Singer/Band Description", "Use Storyboard Prompt"], value="Use Singer/Band Description", label="Vocal Shot Prompt Mode")
            vid_gen_start_btn = gr.Button("Start Batch Generation", variant="primary")
            vid_gen_stop_btn = gr.Button("Stop Batch Generation", variant="stop", visible=False)
            
        vid_gen_status = gr.Textbox(label="Batch Generation Status", interactive=False)
        
        gr.Markdown("### 🎯 Single Shot Generation")
        with gr.Row():
            single_shot_dropdown = gr.Dropdown(label="Select Shot to Generate", choices=[], interactive=True)
            refresh_shots_btn = gr.Button("🔄 Refresh Shots", size="sm")
            single_shot_btn = gr.Button("Generate Additional Version", variant="primary")
        single_shot_prompt_edit = gr.Textbox(label="Edit Video Prompt for Selected Shot", lines=3, interactive=True)
        single_shot_status = gr.Textbox(label="Single Shot Status", interactive=False)

        with gr.Row():
            with gr.Column(scale=1):
                vid_gallery = gr.Gallery(label="Generated Video Thumbnails", columns=4, elem_classes=["scrollable-gallery"], allow_preview=False, interactive=True)
            
            with gr.Column(scale=1):
                vid_large_view = gr.Video(label="Selected Video", interactive=False)
                with gr.Row():
                    sel_shot_info_vid = gr.Textbox(label="Selected Shot ID", interactive=False)
                
                with gr.Row():
                    del_vid_btn = gr.Button("🗑️ Delete This Video", variant="stop")
                with gr.Row():
                    regen_vid_same_prompt_btn = gr.Button("♻️ Regenerate Video (Same Prompt)")
                    regen_vid_new_prompt_btn = gr.Button("✨ Regenerate Video AND Prompt", variant="primary")

        # --- Tab 3 Events ---
        def load_single_shot_prompt(shot_id, pm):
            if not shot_id or pm.df.empty: return ""
            row_idx = pm.df.index[pm.df['Shot_ID'].astype(str).str.upper() == str(shot_id).upper()].tolist()
            if row_idx:
                return str(pm.df.loc[row_idx[0], 'Video_Prompt'])
            return ""

        single_shot_dropdown.change(load_single_shot_prompt, inputs=[single_shot_dropdown, pm_state], outputs=[single_shot_prompt_edit])

        def save_single_shot_prompt(shot_id, new_prompt, pm):
            if not shot_id or pm.df.empty: return
            row_idx = pm.df.index[pm.df['Shot_ID'].astype(str).str.upper() == str(shot_id).upper()].tolist()
            if row_idx:
                pm.df.at[row_idx[0], 'Video_Prompt'] = new_prompt
                pm.save_data()

        single_shot_prompt_edit.change(save_single_shot_prompt, inputs=[single_shot_dropdown, single_shot_prompt_edit, pm_state])

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
            advanced_batch_video_generation, inputs=[vid_gen_mode_dropdown, vid_versions_dropdown, vid_resolution_dropdown, vid_vocal_prompt_mode, pm_state], outputs=[vid_gallery, vid_large_view, vid_gen_status], show_progress="hidden"
        ).then(
            lambda: (gr.update(visible=True), gr.update(visible=False)), outputs=[vid_gen_start_btn, vid_gen_stop_btn]
        )
        
        vid_gen_stop_btn.click(
            stop_gen, inputs=[pm_state], outputs=[vid_gen_status], cancels=[start_vid_evt]
        ).then(
            lambda: (gr.update(visible=True), gr.update(visible=False)), outputs=[vid_gen_start_btn, vid_gen_stop_btn]
        )
        
        def update_single_shot_choices(pm):
            if pm.df.empty: return gr.update(choices=[])
            return gr.update(choices=pm.df['Shot_ID'].dropna().unique().tolist())
            
        refresh_shots_btn.click(update_single_shot_choices, inputs=[pm_state], outputs=[single_shot_dropdown])

        def handle_single_shot(shot_id, res, vocal_mode, proj, pm):
            if pm.is_generating:
                yield get_project_videos(pm, proj), "❌ Error: A generation process is already actively running."
                return
            if not shot_id:
                yield get_project_videos(pm, proj), "❌ Error: No shot selected."
                return
                
            pm.is_generating = True
            try:
                vid_gen = generate_video_for_shot(shot_id, res, vocal_mode, pm)
                final_path = None
                for path, msg in vid_gen:
                    if path is None:
                        yield get_project_videos(pm, proj), f"⏳ {shot_id}: {msg}"
                    else:
                        final_path = path

                if final_path:
                    sync_video_directory(pm)
                    yield get_project_videos(pm, proj), f"✅ Finished generating new version of {shot_id}"
                else:
                    yield get_project_videos(pm, proj), f"❌ Failed to generate {shot_id}"
            finally:
                pm.is_generating = False

        single_shot_btn.click(handle_single_shot, inputs=[single_shot_dropdown, vid_resolution_dropdown, vid_vocal_prompt_mode, current_proj_var, pm_state], outputs=[vid_gallery, single_shot_status])

        def handle_vid_delete(path_to_del, proj, pm):
            new_gal, _ = delete_video_file(path_to_del, proj, pm)
            return new_gal, None, "", "" 
            
        del_vid_btn.click(handle_vid_delete, inputs=[selected_vid_path, current_proj_var, pm_state], outputs=[vid_gallery, vid_large_view, sel_shot_info_vid, selected_vid_path])

        def handle_regen_vid(shot_id_txt, selected_path, resolution, vocal_mode, proj, pm):
            if pm.is_generating:
                yield gr.update(), gr.update(), "❌ Error: A generation process is already actively running."
                return
            if not shot_id_txt: 
                yield gr.update(), gr.update(), "❌ No Shot ID selected"
                return
                
            pm.is_generating = True
            try:
                if selected_path and os.path.exists(selected_path):
                    try: os.remove(selected_path)
                    except Exception as e: print(f"Could not delete file {selected_path}: {e}")

                vid_generator = generate_video_for_shot(shot_id_txt, resolution, vocal_mode, pm)
                final_path = None
                for path, msg in vid_generator:
                    if path is None:
                        yield gr.update(), gr.update(), f"⏳ {shot_id_txt}: {msg}"
                    else:
                        final_path = path

                if final_path:
                    sync_video_directory(pm)
                    yield get_project_videos(pm, proj), final_path, f"✅ Finished regenerating {shot_id_txt}"
                else:
                    yield get_project_videos(pm, proj), gr.update(), f"❌ Failed to regenerate {shot_id_txt}"
            finally:
                pm.is_generating = False

        def handle_regen_vid_and_prompt(shot_id_txt, selected_path, resolution, vocal_mode, proj, pm):
            if pm.is_generating:
                yield gr.update(), gr.update(), "❌ Error: A generation process is already actively running."
                return
            if not shot_id_txt: 
                yield gr.update(), gr.update(), "❌ No Shot ID selected"
                return
                
            pm.is_generating = True
            try:
                settings = pm.load_project_settings()
                llm_model = settings.get("llm_model", "qwen3-vl-8b-instruct-abliterated-v2.0")
                plot = settings.get("plot", "")
                prompt_template = settings.get("prompt_template", DEFAULT_CONCEPT_PROMPT)
                performance_desc = settings.get("performance_desc", "")
                
                yield get_project_videos(pm, proj), gr.update(), f"⏳ Generating new prompt for {shot_id_txt}..."
                time.sleep(0.1)
                
                llm = LLMBridge()
                row_idx = pm.df.index[pm.df['Shot_ID'].astype(str).str.upper() == str(shot_id_txt).upper()].tolist()
                if not row_idx:
                    yield get_project_videos(pm, proj), gr.update(), f"❌ Shot {shot_id_txt} not found."
                    return
                index = row_idx[0]
                row = pm.df.loc[index]
                
                if row['Type'] == 'Vocal':
                    final_vid_prompt = performance_desc
                else:
                    loc_pos = pm.df.index.get_loc(index)
                    if loc_pos > 0:
                        prev_index = pm.df.index[loc_pos - 1]
                        prev_shot_text = pm.df.loc[prev_index, 'Video_Prompt']
                        if pd.isna(prev_shot_text): prev_shot_text = "N/A"
                    else:
                        prev_shot_text = "None (Start of video)"

                    filled_prompt = prompt_template.replace("{plot}", plot)\
                        .replace("{type}", row['Type'])\
                        .replace("{start}", f"{row['Start_Time']:.1f}")\
                        .replace("{duration}", f"{row['Duration']:.1f}")\
                        .replace("{prev_shot}", prev_shot_text)
                    
                    final_vid_prompt = llm.query(LTX_SYSTEM_PROMPT, filled_prompt, llm_model)

                pm.df.at[index, 'Video_Prompt'] = final_vid_prompt
                pm.save_data()
                
                yield get_project_videos(pm, proj), gr.update(), f"⏳ Prompt generated. Starting video generation for {shot_id_txt}..."
                time.sleep(0.1)
                
                if selected_path and os.path.exists(selected_path):
                    try: os.remove(selected_path)
                    except Exception as e: pass

                vid_generator = generate_video_for_shot(shot_id_txt, resolution, vocal_mode, pm)
                final_path = None
                for path, msg in vid_generator:
                    if path is None:
                        yield get_project_videos(pm, proj), gr.update(), f"⏳ {shot_id_txt}: {msg}"
                    else:
                        final_path = path

                if final_path:
                    sync_video_directory(pm)
                    yield get_project_videos(pm, proj), final_path, f"✅ Finished regenerating prompt and video for {shot_id_txt}"
                else:
                    yield get_project_videos(pm, proj), gr.update(), f"❌ Failed to regenerate {shot_id_txt}"
            finally:
                pm.is_generating = False

        regen_vid_same_prompt_btn.click(handle_regen_vid, inputs=[sel_shot_info_vid, selected_vid_path, vid_resolution_dropdown, vid_vocal_prompt_mode, current_proj_var, pm_state], outputs=[vid_gallery, vid_large_view, vid_gen_status], show_progress="hidden")
        regen_vid_new_prompt_btn.click(handle_regen_vid_and_prompt, inputs=[sel_shot_info_vid, selected_vid_path, vid_resolution_dropdown, vid_vocal_prompt_mode, current_proj_var, pm_state], outputs=[vid_gallery, vid_large_view, vid_gen_status], show_progress="hidden")

# --- TAB 4: ASSEMBLY & CUTTING ROOM ---
    with gr.Tab("4. Assembly & Cutting Room") as tab4_ui:
        gr.Markdown("### ✂️ Cutting Room & Version Comparison")
        with gr.Row():
            compare_shot_dropdown = gr.Dropdown(label="Select Shot to Compare Versions")
            next_shot_btn = gr.Button("➡️ Next Shot") 
        
        compare_cols = []
        compare_vids = []
        compare_set_btns = []
        compare_cut_btns = []
        compare_paths = []
        
        with gr.Row():
            for i in range(5):
                with gr.Column(visible=False) as col:
                    cvid = gr.Video(label=f"Version {i+1}", loop=True, interactive=False)
                    cset = gr.Button("⭐ Set as Active", variant="primary")
                    ccut = gr.Button("✂️ Move to Cutting Room Floor", variant="stop")
                    cpath = gr.State("")
                    
                    compare_cols.append(col)
                    compare_vids.append(cvid)
                    compare_set_btns.append(cset)
                    compare_cut_btns.append(ccut)
                    compare_paths.append(cpath)
                    
        gr.Markdown("---")
        gr.Markdown("### 🎞️ Final Assembly")
        with gr.Row():
            assemble_btn = gr.Button("Assemble Final Video (Strictly Videos)", variant="secondary")
            assemble_current_btn = gr.Button("Assemble with Current Assets (Videos > Black Fallback)", variant="primary")
        final_video_out = gr.Video(label="Final Cut")
        
        # --- Tab 4 Logic Wiring ---
        def manual_sync_and_get_choices(pm, progress=gr.Progress()):
            progress(0, desc="Syncing Video Directory...")
            sync_video_directory(pm)
            progress(0.8, desc="Updating Shot List...")
            if pm.df.empty: return gr.update(choices=[]), pm.df
            choices = pm.df[pm.df["All_Video_Paths"] != ""]["Shot_ID"].dropna().unique().tolist()
            progress(1.0, desc="Complete!")
            return gr.update(choices=choices), pm.df

        tab4_ui.select(manual_sync_and_get_choices, inputs=[pm_state], outputs=[compare_shot_dropdown, shot_table])
        
        # Next shot cycling logic
        def get_next_shot(current_shot, pm):
            if pm.df.empty: return gr.update()
            
            choices = pm.df[pm.df["All_Video_Paths"] != ""]["Shot_ID"].dropna().unique().tolist()
            if not choices: return gr.update(value=None)
            
            if current_shot not in choices:
                all_shots = pm.df["Shot_ID"].dropna().unique().tolist()
                if current_shot in all_shots:
                    curr_idx = all_shots.index(current_shot)
                    for i in range(1, len(all_shots) + 1):
                        check_idx = (curr_idx + i) % len(all_shots)
                        if all_shots[check_idx] in choices:
                            return gr.update(value=all_shots[check_idx])
                return gr.update(value=choices[0])
                
            idx = choices.index(current_shot)
            next_idx = (idx + 1) % len(choices)
            return gr.update(value=choices[next_idx])

        next_shot_btn.click(get_next_shot, inputs=[compare_shot_dropdown, pm_state], outputs=[compare_shot_dropdown])
        
        def update_comparison_view(shot_id, pm):
            if not shot_id or pm.df.empty:
                return [gr.update(visible=False)] * 5 + [gr.update(value=None)] * 5 + [""] * 5
                
            row_idx = pm.df.index[pm.df['Shot_ID'].astype(str).str.upper() == str(shot_id).upper()].tolist()
            if not row_idx:
                return [gr.update(visible=False)] * 5 + [gr.update(value=None)] * 5 + [""] * 5
                
            paths_str = pm.df.loc[row_idx[0], "All_Video_Paths"]
            if not paths_str or pd.isna(paths_str): paths = []
            else: paths = paths_str.split(",")
            
            col_updates = []
            vid_updates = []
            path_updates = []
            
            active_path = pm.df.loc[row_idx[0], "Video_Path"]
            
            for i in range(5):
                if i < len(paths):
                    p = paths[i]
                    is_active = (p == active_path)
                    label = f"Version {i+1} {'(ACTIVE)' if is_active else ''}"
                    col_updates.append(gr.update(visible=True))
                    vid_updates.append(gr.update(value=p, label=label))
                    path_updates.append(p)
                else:
                    col_updates.append(gr.update(visible=False))
                    vid_updates.append(gr.update(value=None))
                    path_updates.append("")
                    
            return col_updates + vid_updates + path_updates
            
        compare_shot_dropdown.change(update_comparison_view, inputs=[compare_shot_dropdown, pm_state], outputs=compare_cols + compare_vids + compare_paths)
        
        def set_active_video(path, shot_id, pm):
            if not path or not os.path.exists(path): return update_comparison_view(shot_id, pm)
            row_idx = pm.df.index[pm.df['Shot_ID'].astype(str).str.upper() == str(shot_id).upper()].tolist()
            if row_idx:
                pm.df.at[row_idx[0], "Video_Path"] = path
                pm.save_data()
            return update_comparison_view(shot_id, pm)
            
        def move_to_cutting_room(path, shot_id, pm):
            if not path or not os.path.exists(path):
                choices = pm.df[pm.df["All_Video_Paths"] != ""]["Shot_ID"].dropna().unique().tolist() if not pm.df.empty else []
                return [gr.update(choices=choices, value=shot_id)] + update_comparison_view(shot_id, pm)

            cut_dir = pm.get_path("cutting_room")
            os.makedirs(cut_dir, exist_ok=True)
            fname = os.path.basename(path)
            dest = os.path.join(cut_dir, fname)
            shutil.move(path, dest)
            sync_video_directory(pm)

            # Recalculate options & drop fallback if the selected shot was depleted
            choices = pm.df[pm.df["All_Video_Paths"] != ""]["Shot_ID"].dropna().unique().tolist() if not pm.df.empty else []
            if shot_id not in choices:
                shot_id = choices[0] if choices else None

            return [gr.update(choices=choices, value=shot_id)] + update_comparison_view(shot_id, pm)
            
        for i in range(5):
            compare_set_btns[i].click(set_active_video, inputs=[compare_paths[i], compare_shot_dropdown, pm_state], outputs=compare_cols + compare_vids + compare_paths)
            compare_cut_btns[i].click(move_to_cutting_room, inputs=[compare_paths[i], compare_shot_dropdown, pm_state], outputs=[compare_shot_dropdown] + compare_cols + compare_vids + compare_paths)
        
        assemble_btn.click(lambda s, res, pm: assemble_video(get_file_path(s), res, pm, fallback_mode=False), inputs=[song_up, vid_resolution_dropdown, pm_state], outputs=[final_video_out])
        assemble_current_btn.click(lambda s, res, pm: assemble_video(get_file_path(s), res, pm, fallback_mode=True), inputs=[song_up, vid_resolution_dropdown, pm_state], outputs=[final_video_out])

# ==========================================
# GLOBAL LOGIC & WIRING
# ==========================================

    def handle_create(name, v_file, s_file, lyrics_text, pm):
        msg = pm.create_project(name)
        clean_name = pm.sanitize_name(name)
        
        if "already exists" in msg or "Invalid" in msg:
             return msg, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        
        if v_file:
            v_src = get_file_path(v_file)
            if v_src: pm.save_asset(v_src, "vocals.mp3")
            
        if s_file:
            s_src = get_file_path(s_file)
            if s_src: pm.save_asset(s_src, "full_song.mp3")
            
        if lyrics_text:
            pm.save_lyrics(lyrics_text)
            
        df = pd.DataFrame(columns=REQUIRED_COLUMNS)
        
        return (
            msg, 
            gr.update(choices=get_existing_projects(), value=clean_name), 
            clean_name, 
            "00h00m00s",
            df, 
            "", 
            "", 
            "", 
            []
        )

    def handle_load(name, pm):
        msg, df = pm.load_project(name)
        lyrics = pm.get_lyrics()
        v_path = pm.get_asset_path_if_exists("vocals.mp3")
        s_path = pm.get_asset_path_if_exists("full_song.mp3")
        settings = pm.load_project_settings()
        
        gal_vids = get_project_videos(pm, name)
        time_str = format_time(pm.total_time_spent)
        
        return (
            msg, time_str, df, lyrics, v_path, s_path, 
            settings.get("min_silence", 700), settings.get("silence_thresh", -45), 
            settings.get("shot_mode", "Random"), settings.get("min_dur", 2), settings.get("max_dur", 4),
            settings.get("llm_model", "qwen3-vl-8b-instruct-abliterated-v2.0"), settings.get("rough_concept", ""), 
            settings.get("plot", ""), 
            settings.get("prompt_template", DEFAULT_CONCEPT_PROMPT),
            settings.get("performance_desc", ""), 
            name,
            gal_vids, gr.update(value="Start Batch Generation", variant="primary")
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

    def auto_save_tab2(proj_name, min_sil, sil_thresh, mode, min_d, max_d, llm, concept, plot, template, performance_d, pm):
        if proj_name:
            pm.current_project = proj_name
            settings = {
                "min_silence": min_sil, "silence_thresh": sil_thresh, "shot_mode": mode,
                "min_dur": min_d, "max_dur": max_d, "llm_model": llm,
                "rough_concept": concept, "plot": plot, "prompt_template": template,
                "performance_desc": performance_d
            }
            pm.save_project_settings(settings)

    refresh_proj_btn.click(lambda: gr.update(choices=get_existing_projects()), outputs=[project_dropdown])

    create_btn.click(
        handle_create, 
        inputs=[proj_name, vocals_up, song_up, lyrics_in, pm_state], 
        outputs=[proj_status, project_dropdown, current_proj_var, time_spent_disp, shot_table, rough_concept_in, plot_out, performance_desc_in, vid_gallery]
    )

    load_btn.click(
        handle_load, 
        inputs=[project_dropdown, pm_state], 
        outputs=[
            proj_status, time_spent_disp, shot_table, lyrics_in, vocals_up, song_up, 
            min_silence_sl, silence_thresh_sl, shot_mode_drp, min_shot_dur, max_shot_dur,
            llm_dropdown, rough_concept_in, plot_out, prompt_template_in, 
            performance_desc_in,
            current_proj_var,
            vid_gallery, vid_gen_start_btn 
        ]
    )

    delete_proj_btn.click(handle_delete_project, inputs=[project_dropdown, pm_state], outputs=[proj_status, project_dropdown])

    lyrics_in.change(auto_save_lyrics, inputs=[current_proj_var, lyrics_in, pm_state])
    
    for file_comp in [vocals_up, song_up]:
        file_comp.upload(auto_save_files, inputs=[current_proj_var, vocals_up, song_up, pm_state])
        file_comp.clear(auto_save_files, inputs=[current_proj_var, vocals_up, song_up, pm_state])
        
    t2_inputs = [current_proj_var, min_silence_sl, silence_thresh_sl, shot_mode_drp, min_shot_dur, max_shot_dur, llm_dropdown, rough_concept_in, plot_out, prompt_template_in, performance_desc_in, pm_state]
    
    for tab2_comp in [min_silence_sl, silence_thresh_sl, shot_mode_drp, min_shot_dur, max_shot_dur, llm_dropdown]:
        tab2_comp.change(auto_save_tab2, inputs=t2_inputs)
        
    for tab2_text_comp in [rough_concept_in, plot_out, prompt_template_in, performance_desc_in]:
        tab2_text_comp.blur(auto_save_tab2, inputs=t2_inputs)
        
    export_csv_btn.click(lambda pm: pm.export_csv(), inputs=[pm_state], outputs=csv_downloader)
    import_csv_btn.upload(lambda f, pm: pm.import_csv(f), inputs=[import_csv_btn, pm_state], outputs=[import_status, shot_table])
    
    download_story_btn.click(generate_story_file, inputs=[pm_state], outputs=[story_downloader])
    
    def save_manual_df_edits(new_df, pm):
        if pm.current_project:
            # Reformat to Pandas DataFrame in case Gradio returns it as a list
            if isinstance(new_df, list):
                new_df = pd.DataFrame(new_df, columns=REQUIRED_COLUMNS)
            pm.df = new_df
            pm.save_data()
            
    shot_table.change(save_manual_df_edits, inputs=[shot_table, pm_state])

    refresh_llm_btn.click(lambda: gr.update(choices=LLMBridge().get_models()), outputs=llm_dropdown)
    
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
    
    gen_concepts_btn.click(generate_concepts_logic, inputs=[plot_out, prompt_template_in, llm_dropdown, rough_concept_in, performance_desc_in, gr.State(None), pm_state], outputs=[shot_table, concept_gen_status])
    stop_concepts_btn.click(stop_gen, inputs=[pm_state], outputs=[concept_gen_status]) 
    regen_single_btn.click(generate_concepts_logic, inputs=[plot_out, prompt_template_in, llm_dropdown, rough_concept_in, performance_desc_in, regen_shot_id, pm_state], outputs=[shot_table, concept_gen_status])
    regen_all_btn.click(generate_concepts_logic, inputs=[plot_out, prompt_template_in, llm_dropdown, rough_concept_in, performance_desc_in, gr.State("ALL"), pm_state], outputs=[shot_table, concept_gen_status])

    # Dynamic UI Refresh Event
    tab2_ui.select(lambda pm: pm.df, inputs=[pm_state], outputs=[shot_table])

if __name__ == "__main__":
    try:
        keyboard.add_hotkey('ctrl+r', restart_application)
        print("⌨️  Hotkey Ctrl+R registered for restarting the application. (Ensure your terminal has focus to use)")
    except Exception as e:
        print(f"⚠️ Could not register hotkey 'ctrl+r'. Run script as admin or ensure 'keyboard' module is installed. Error: {e}")
        
    app.launch()