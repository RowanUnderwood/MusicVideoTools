import gradio as gr
import pandas as pd
import os
import sys
import json
import requests
import uuid
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
import keyboard  # Requires: pip install keyboard

# ==========================================
# CONFIGURATION
# ==========================================
LTX_BASE_URL = "http://127.0.0.1:8000/api"
LM_STUDIO_URL = "http://127.0.0.1:1234/v1"

REQUIRED_COLUMNS = [
    "Shot_ID", "Type", 
    "Start_Time", "End_Time", "Duration", 
    "Start_Frame", "End_Frame", "Total_Frames",
    "Lyrics", "Video_Prompt", "Video_Path", "Status"
]

DEFAULT_CONCEPT_PROMPT = (
    "Context: The overarching plot is: {plot}\n"
    "Previous Shot Visual: {prev_shot}\n"
    "Current Shot Info: Timestamp {start}s, Duration {duration}s, Type: {type}.\n"
    "Task: Write a highly detailed visual description for this video shot encompassing the action. "
    "Describe the scene, camera motion, emotions, lighting and performance only. "
    "Pay special attention to the camera's motion. Do not include any additional notes or titles."
)

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
    min_frames = math.ceil(target_seconds * fps)
    valid_buckets = [1 + 8*k for k in range(60)] 
    for bucket in valid_buckets:
        if bucket >= min_frames:
            return bucket
    return valid_buckets[-1]

def get_ltx_duration(seconds, fps=24):
    frames = get_ltx_frame_count(seconds, fps)
    return frames / fps

def format_time(seconds):
    """Helper to format seconds into hh:mm:ss"""
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
        """Calculates total active project time, auto-saves it, and returns it."""
        if self.session_start_time and self.current_project:
            elapsed = time.time() - self.session_start_time
            self.session_start_time = time.time()  
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
        folders = ["assets", "audio_chunks", "videos", "renders"]
        
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
        
        aligned_start = math.floor(voc_start * fps) / fps
        aligned_end = math.ceil(voc_end * fps) / fps
        
        if aligned_start < current_cursor:
            aligned_start = current_cursor
            
        while (aligned_start - current_cursor) >= frame_step: 
            gap_duration = aligned_start - current_cursor
            
            chosen_raw = min_dur if shot_mode == "Fixed" else random.uniform(min_dur, max_dur)
            if chosen_raw > 5.0: chosen_raw = 5.0
            if chosen_raw > gap_duration: chosen_raw = gap_duration

            chosen_dur = get_ltx_duration(chosen_raw, fps)
            chosen_dur = min(chosen_dur, gap_duration) # BUG 4 FIX
            action_end = current_cursor + chosen_dur
            
            new_rows.append(create_row("Action", current_cursor, action_end, shot_counter))
            shot_counter += 1
            current_cursor = action_end
            
            if current_cursor >= aligned_start: break

        vocal_cursor = current_cursor
        while vocal_cursor < (aligned_end - frame_step):
            remaining = aligned_end - vocal_cursor
            chunk_raw = 5.0 if remaining > 5.0 else remaining
            chunk_size = get_ltx_duration(chunk_raw, fps)
            chunk_size = min(chunk_size, remaining) # BUG 4 FIX
            
            new_rows.append(create_row("Vocal", vocal_cursor, vocal_cursor + chunk_size, shot_counter))
            shot_counter += 1
            vocal_cursor += chunk_size
            
        current_cursor = vocal_cursor

    remaining_time = total_duration - current_cursor
    while remaining_time > frame_step:
        chosen_raw = min_dur if shot_mode == "Fixed" else random.uniform(min_dur, max_dur)
        if chosen_raw > 5.0: chosen_raw = 5.0
        if chosen_raw > remaining_time: chosen_raw = remaining_time
        
        chosen_dur = get_ltx_duration(chosen_raw, fps)
        chosen_dur = min(chosen_dur, remaining_time) # BUG 4 FIX
        
        new_rows.append(create_row("Action", current_cursor, current_cursor + chosen_dur, shot_counter))
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
            
            final_vid_prompt = llm.query("You are a creative visual director.", filled_prompt, llm_model)

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
        
        # Open the clip briefly to calculate the exact frame count
        try:
            clip = VideoFileClip(f)
            frame_count = round(clip.fps * clip.duration)
            clip.close()
            caption = f"{caption} ({frame_count} frames)"
        except Exception as e:
            print(f"Error reading frame count for {fname}: {e}")
            caption = f"{caption} (Error)"
            
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

def generate_video_for_shot(shot_id, resolution, pm):
    row_idx = pm.df.index[pm.df['Shot_ID'].astype(str).str.upper() == str(shot_id).upper()].tolist()
    if not row_idx:
        yield None, "Error: Shot not found in timeline."
        return
        
    row = pm.df.loc[row_idx[0]]
    vid_prompt = row.get('Video_Prompt', '')
    if pd.isna(vid_prompt) or not vid_prompt:
        yield None, "Error: Missing Video Prompt."
        return

    print(f"\n🎬 === START VIDEO GENERATION (LTX) ===")
    print(f"🎬 Shot ID: {shot_id} | Type: {row['Type']}")
    print(f"🎬 Video Prompt:\n{vid_prompt}\n=================================\n")

    payload = {
        "prompt": vid_prompt,
        "negativePrompt": "blurry, distorted, low quality, artifacts, watermark",
        "model": "pro",
        "resolution": resolution,  # <--- FIXED BUG 2
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
            # BUG 5 FIX: Use round() instead of int()
            start_ms = round(float(row['Start_Time']) * 1000)
            end_ms = round(float(row['End_Time']) * 1000)
            
            chunk = audio[start_ms : end_ms]
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

def advanced_batch_video_generation(mode, target_versions, resolution, pm): # BUG 2 FIX
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
            # BUG 7 FIX: Actually filter for remaining shots
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
                vid_generator = generate_video_for_shot(shot_id, resolution, pm)
                
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
        pm.is_generating = False

def assemble_video(full_song_path, pm, fallback_mode=False):
    df = pm.df
    clips = []
    clips_to_close = [] 
    if df.empty: return "No shots to assemble."

    df = df.sort_values(by="Start_Time")
    
    expected_cursor = 0.0

    for index, row in df.iterrows():
        vid_path = row.get('Video_Path')
        dur = float(row['Duration'])
        start_time = float(row['Start_Time'])
        clip = None
        
        # BUG 3 FIX: Snap to exact frame
        gap = round((start_time - expected_cursor) * 24) / 24
        if gap > 0.05:
            pad = ColorClip(size=(1920, 1080), color=(0,0,0), duration=gap).set_fps(24)
            clips.append(pad)
            clips_to_close.append(pad)
        
        if vid_path and pd.notna(vid_path) and os.path.exists(str(vid_path)):
            try:
                clip = VideoFileClip(str(vid_path)).without_audio().set_fps(24)
                
                # BUG 3 FIX: Enforce exact snapped duration
                snapped_dur = round(dur * 24) / 24
                if clip.duration > snapped_dur: 
                    clip = clip.subclip(0, snapped_dur)
                clip = clip.set_duration(snapped_dur)
            except Exception as e:
                print(f"Error loading clip {vid_path}: {e}")

        if clip is None:
            snapped_dur = round(dur * 24) / 24
            clip = ColorClip(size=(1920, 1080), color=(0,0,0), duration=snapped_dur).set_fps(24)
            
        if clip is not None:
            clips.append(clip)
            clips_to_close.append(clip)
            
        expected_cursor = start_time + dur

    if not clips: return "No valid clips found."

    # BUG 1 FIX: method="chain" prevents progressive offset drift
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
    
    # BUG 8 FIX: Temp audio path stability
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
                import_csv_btn = gr.UploadButton("Import CSV (Overwrite)", file_types=[".csv"])
                import_status = gr.Textbox(label="Import Status", interactive=False)

        with gr.Row():
            regen_shot_id = gr.Textbox(label="Shot ID to Regenerate", placeholder="S005")
            regen_single_btn = gr.Button("Regenerate Single Shot")
            regen_all_btn = gr.Button("Regenerate All Shots")
        shot_table = gr.Dataframe(headers=REQUIRED_COLUMNS, interactive=True, wrap=True)

# --- TAB 3: VIDEO GENERATION ---
    with gr.Tab("3. Video Generation"):
        selected_vid_path = gr.State("")
        
        with gr.Row():
            vid_gen_mode_dropdown = gr.Dropdown(choices=["Generate Remaining Shots", "Regenerate all Shots", "Generate all Action Shots", "Generate all Vocal Shots"], value="Generate Remaining Shots", label="Generation Mode")
            vid_versions_dropdown = gr.Dropdown(choices=[1, 2, 3], value=1, label="Versions per Shot")
            # BUG 2 FIX: Add resolution dropdown
            vid_resolution_dropdown = gr.Dropdown(choices=["720p", "1080p", "1440p"], value="1080p", label="Resolution")
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

        # --- Tab 3 Events ---
        def on_vid_gallery_select(evt: gr.SelectData, proj, pm):
            gal_data = get_project_videos(pm, proj)
            if evt.index < len(gal_data):
                fpath = gal_data[evt.index][0]
                fname = os.path.basename(fpath)
                shot_id = fname.split('_')[0] if '_' in fname else "Unknown"
                return fpath, shot_id, fpath 
            return None, "", ""

        vid_gallery.select(on_vid_gallery_select, inputs=[current_proj_var, pm_state], outputs=[vid_large_view, sel_shot_info_vid, selected_vid_path])
        
        # BUG 2 FIX: Include vid_resolution_dropdown in inputs
        start_vid_evt = vid_gen_start_btn.click(
            lambda: (gr.update(visible=False), gr.update(visible=True)), outputs=[vid_gen_start_btn, vid_gen_stop_btn]
        ).then(
            advanced_batch_video_generation, inputs=[vid_gen_mode_dropdown, vid_versions_dropdown, vid_resolution_dropdown, pm_state], outputs=[vid_gallery, vid_large_view, vid_gen_status], show_progress="hidden"
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

        # BUG 2 FIX: Add resolution param
        def handle_regen_vid(shot_id_txt, selected_path, resolution, proj, pm):
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

                vid_generator = generate_video_for_shot(shot_id_txt, resolution, pm)
                final_path = None
                for path, msg in vid_generator:
                    if path is None:
                        yield gr.update(), gr.update(), f"⏳ {shot_id_txt}: {msg}"
                    else:
                        final_path = path

                if final_path:
                    yield get_project_videos(pm, proj), final_path, f"✅ Finished regenerating {shot_id_txt}"
                else:
                    yield get_project_videos(pm, proj), gr.update(), f"❌ Failed to regenerate {shot_id_txt}"
            finally:
                pm.is_generating = False

        # BUG 2 FIX: Add vid_resolution_dropdown to inputs
        regen_vid_btn.click(handle_regen_vid, inputs=[sel_shot_info_vid, selected_vid_path, vid_resolution_dropdown, current_proj_var, pm_state], outputs=[vid_gallery, vid_large_view, vid_gen_status], show_progress="hidden")

# --- TAB 4: ASSEMBLY ---
    with gr.Tab("4. Assembly"):
        with gr.Row():
            assemble_btn = gr.Button("Assemble Final Video (Strictly Videos)", variant="secondary")
            assemble_current_btn = gr.Button("Assemble with Current Assets (Videos > Black Fallback)", variant="primary")
        final_video_out = gr.Video(label="Final Cut")
        
        assemble_btn.click(lambda s, pm: assemble_video(get_file_path(s), pm, fallback_mode=False), inputs=[song_up, pm_state], outputs=[final_video_out])
        assemble_current_btn.click(lambda s, pm: assemble_video(get_file_path(s), pm, fallback_mode=True), inputs=[song_up, pm_state], outputs=[final_video_out])

# ==========================================
# GLOBAL LOGIC & WIRING
# ==========================================

    def handle_create(name, v_file, s_file, lyrics_text, pm):
        msg = pm.create_project(name)
        clean_name = pm.sanitize_name(name)
        
        if v_file:
            v_src = get_file_path(v_file)
            if v_src: pm.save_asset(v_src, "vocals.mp3")
            
        if s_file:
            s_src = get_file_path(s_file)
            if s_src: pm.save_asset(s_src, "full_song.mp3")
            
        if lyrics_text:
            pm.save_lyrics(lyrics_text)
            
        return msg, gr.update(choices=get_existing_projects(), value=clean_name), clean_name, "00h00m00s"

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
            gal_vids, gr.Button(value="Start Generation", variant="primary")
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
        outputs=[proj_status, project_dropdown, current_proj_var, time_spent_disp]
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
    for tab2_comp in [min_silence_sl, silence_thresh_sl, shot_mode_drp, min_shot_dur, max_shot_dur, llm_dropdown, rough_concept_in, plot_out, prompt_template_in, performance_desc_in]:
        tab2_comp.change(auto_save_tab2, inputs=t2_inputs)
        
    export_csv_btn.click(lambda pm: pm.export_csv(), inputs=[pm_state], outputs=csv_downloader)
    import_csv_btn.upload(lambda f, pm: pm.import_csv(f), inputs=[import_csv_btn, pm_state], outputs=[import_status, shot_table])
    
    def save_manual_df_edits(new_df, pm):
        if pm.current_project:
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

if __name__ == "__main__":
    try:
        keyboard.add_hotkey('ctrl+r', restart_application)
        print("⌨️  Hotkey Ctrl+R registered for restarting the application. (Ensure your terminal has focus to use)")
    except Exception as e:
        print(f"⚠️ Could not register hotkey 'ctrl+r'. Run script as admin or ensure 'keyboard' module is installed. Error: {e}")
        
    app.launch()