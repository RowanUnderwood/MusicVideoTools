import gradio as gr
import pandas as pd
import os
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

# ==========================================
# CONFIGURATION
# ==========================================
COMFY_URL = "127.0.0.1:8188"
LM_STUDIO_URL = "http://127.0.0.1:1234/v1"  # Base URL for LM Studio

# WORKFLOW FILES
WORKFLOW_IMG = "workflows/ZImage_Poster_API.json"
WORKFLOW_VID_ACTION = "workflows/wan2.2_infinite_video_lightning edition-painter jakes version x.json"
WORKFLOW_VID_VOCAL = "workflows/011426-LTX2-AudioSync-i2v-Ver2-Jakes Version API.json"

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
        return json.loads(urllib.request.urlopen(req).read())

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
        return ["qwen3-vl-8b-instruct-abliterated-v2.0"] # Default fallback

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
        except requests.exceptions.JSONDecodeError:
            return f"Error: LLM response was not valid JSON. Raw response: {resp.text}"
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
        self.df = pd.DataFrame()
        self.stop_generation = False 

    def sanitize_name(self, name):
        # Remove invalid characters and prevent directory traversal
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
        
        self.df = pd.DataFrame(columns=[
            "Shot_ID", "Type", "Start_Time", "End_Time", "Duration", 
            "Lyrics", "Concept", "Visual_Prompt", "Image_Path", "Video_Path", "Status"
        ])
        self.df.to_csv(os.path.join(path, "shot_list.csv"), index=False)
        self.current_project = clean_name
        
        # Create empty lyrics file
        with open(os.path.join(path, "lyrics.txt"), "w") as f:
            f.write("")
            
        return f"Project '{clean_name}' created."

    def load_project(self, name):
        path = os.path.join(self.base_dir, name)
        csv_path = os.path.join(path, "shot_list.csv")
        if os.path.exists(csv_path):
            self.df = pd.read_csv(csv_path)
            self.current_project = name
            return f"Loaded '{name}'", self.df
        return "Project not found.", pd.DataFrame()

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
        if not self.current_project:
            return "No project loaded. Cannot save settings."
        
        path = os.path.join(self.base_dir, self.current_project, "settings.json")
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(settings_dict, f, indent=4)
            return f"Settings saved to {self.current_project}/settings.json"
        except Exception as e:
            return f"Error saving settings: {e}"

    def load_project_settings(self):
        if not self.current_project:
            return {}
        
        path = os.path.join(self.base_dir, self.current_project, "settings.json")
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {} # Return empty if corrupt
        return {}
        
    def save_asset(self, source_path, filename):
        """Copies an asset to the project folder."""
        if not self.current_project or not source_path: return None
        dest = os.path.join(self.get_path("assets"), filename)
        
        # Don't copy if it's the same file
        if os.path.abspath(source_path) == os.path.abspath(dest):
            return dest
            
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
    if not os.path.exists(pm.base_dir):
        return []
    projects = [d for d in os.listdir(pm.base_dir) if os.path.isdir(os.path.join(pm.base_dir, d))]
    return sorted(projects)

def scan_vocals_advanced(vocals_file_path, project_name, min_silence, silence_thresh, 
                        shot_mode, min_dur, max_dur):
    if not project_name or not vocals_file_path or not os.path.exists(vocals_file_path):
        return pd.DataFrame()

    try:
        # Load Audio
        audio = AudioSegment.from_mp3(vocals_file_path)
    except Exception as e:
        print(f"Error loading audio: {e}")
        return pd.DataFrame()

    total_duration = audio.duration_seconds
    
    nonsilent_ranges = silence.detect_nonsilent(
        audio, 
        min_silence_len=int(min_silence), 
        silence_thresh=silence_thresh
    )
    
    new_rows = []
    current_ms = 0
    
    def get_next_dur_ms():
        if shot_mode == "Fixed":
            return max_dur * 1000
        else:
            return random.uniform(min_dur, max_dur) * 1000

    shot_counter = 1

    for (start_ms, end_ms) in nonsilent_ranges:
        gap_ms = start_ms - current_ms
        while gap_ms > 0:
            chosen_dur = get_next_dur_ms()
            if chosen_dur > 5000: chosen_dur = 5000
            if chosen_dur > gap_ms: chosen_dur = gap_ms
            if gap_ms - chosen_dur < 500 and gap_ms > 500: chosen_dur = gap_ms 
            
            new_rows.append({
                "Shot_ID": f"S{shot_counter:03d}",
                "Type": "Action",
                "Start_Time": current_ms / 1000.0,
                "End_Time": (current_ms + chosen_dur) / 1000.0,
                "Duration": chosen_dur / 1000.0,
                "Status": "Pending"
            })
            shot_counter += 1
            current_ms += chosen_dur
            gap_ms -= chosen_dur

        vocal_dur_ms = end_ms - start_ms
        vocal_cursor = start_ms
        while vocal_dur_ms > 0:
            chunk_size = 5000 if vocal_dur_ms > 5000 else vocal_dur_ms
            new_rows.append({
                "Shot_ID": f"S{shot_counter:03d}",
                "Type": "Vocal",
                "Start_Time": vocal_cursor / 1000.0,
                "End_Time": (vocal_cursor + chunk_size) / 1000.0,
                "Duration": chunk_size / 1000.0,
                "Status": "Pending"
            })
            shot_counter += 1
            vocal_cursor += chunk_size
            vocal_dur_ms -= chunk_size
        
        current_ms = end_ms

    remaining_ms = (total_duration * 1000) - current_ms
    while remaining_ms > 500: 
        chosen_dur = get_next_dur_ms()
        if chosen_dur > 5000: chosen_dur = 5000
        if chosen_dur > remaining_ms: chosen_dur = remaining_ms
        
        new_rows.append({
            "Shot_ID": f"S{shot_counter:03d}",
            "Type": "Action",
            "Start_Time": current_ms / 1000.0,
            "End_Time": (current_ms + chosen_dur) / 1000.0,
            "Duration": chosen_dur / 1000.0,
            "Status": "Pending"
        })
        shot_counter += 1
        current_ms += chosen_dur
        remaining_ms -= chosen_dur
    # Create the dataframe from the new rows
    new_df = pd.DataFrame(new_rows)
    
    # Ensure all standard columns exist, even if empty
    required_columns = ["Lyrics", "Concept", "Visual_Prompt", "Image_Path", "Video_Path"]
    for col in required_columns:
        if col not in new_df.columns:
            new_df[col] = ""  # Add missing column with empty string values
            
    pm.df = new_df
    pm.save_data()
    return pm.df

def generate_overarching_plot(concept, lyrics, llm_model):
    llm = LLMBridge()
    df = pm.df
    
    if df.empty:
        return "Error: Timeline is empty. Please scan vocals first."

    timeline_str = ""
    for idx, row in df.iterrows():
        if row['Type'] == 'Vocal':
            timeline_str += f"[{row['Start_Time']:.1f}s - {row['End_Time']:.1f}s: SINGING]\n"
    
    sys_prompt = "You are a creative writer for music videos."
    user_prompt = (
        f"Rough Concept: {concept}\n\n"
        f"Song Lyrics:\n{lyrics}\n\n"
        f"Timeline of Singing Parts:\n{timeline_str}\n\n"
        "Task: Write a cohesive, linear plot summary for this video. "
        "Analyze the lyrics to ensure the visual narrative aligns with the song's meaning. "
        "Describe what happens during the singing parts and what happens during the instrumental breaks. "
        "Include approximate timestamps in your text. Keep it under 300 words."
    )
    
    return llm.query(sys_prompt, user_prompt, llm_model)

def generate_concepts_logic(overarching_plot, prompt_template, llm_model, specific_shot_id=None, progress=gr.Progress()):
    llm = LLMBridge()
    df = pm.df
    pm.stop_generation = False

    if df.empty: return df

    if specific_shot_id:
        # Create a boolean mask for filtering
        mask = df['Shot_ID'] == specific_shot_id
    else:
        mask = df['Concept'].isna() | (df['Concept'] == "")

    # We iterate over the indices of the filtered rows
    indices_to_process = df[mask].index.tolist()
    
    for index in progress.tqdm(indices_to_process, desc="Generating Concepts"):
        if pm.stop_generation:
            print("Generation Stopped by User.")
            break
            
        row = df.loc[index]
        shot_type = row['Type']
        start_t = row['Start_Time']
        duration = row['Duration']
        
        filled_prompt = prompt_template.replace("{plot}", overarching_plot)
        filled_prompt = filled_prompt.replace("{type}", shot_type)
        filled_prompt = filled_prompt.replace("{start}", f"{start_t:.1f}")
        filled_prompt = filled_prompt.replace("{duration}", f"{duration:.1f}")
        
        sys_prompt = (
            "You are a Director of Photography. "
            "Describe ONLY the visual of the FIRST frame of this shot. "
            "Do not describe camera movement. Do not describe the rest of the clip. "
            "Output max 2 sentences."
        )
        
        concept_text = llm.query(sys_prompt, filled_prompt, llm_model)
        vis_sys_prompt = "You are a detailed image prompt generator. Describe this scene in keywords and visual details for an AI image generator. No text overlays."
        vis_prompt = llm.query(vis_sys_prompt, f"Scene: {concept_text}", llm_model)
        
        # Update the DataFrame
        df.at[index, 'Concept'] = concept_text
        df.at[index, 'Visual_Prompt'] = vis_prompt
        
        # Save per iteration
        pm.df = df
        pm.save_data()
        
    return df

def stop_gen():
    pm.stop_generation = True
    return "Stopping..."

# ==========================================
# LOGIC: GENERATION & ASSEMBLY
# ==========================================

def generate_image_for_shot(shot_id):
    comfy = ComfyBridge()
    if not comfy.connect(): return "ComfyUI Offline"
    
    try:
        # Find the row by Shot_ID
        row_idx = pm.df.index[pm.df['Shot_ID'] == shot_id].tolist()
        if not row_idx:
             return "Shot ID not found"
        
        row = pm.df.loc[row_idx[0]]
        
    except IndexError:
        return "Shot ID not found"
        
    prompt_text = row['Visual_Prompt']
    if not prompt_text: return "No Visual Prompt found"

    try:
        with open(WORKFLOW_IMG, 'r') as f:
            wf = json.load(f)
    except FileNotFoundError:
        return f"Workflow file not found: {WORKFLOW_IMG}"
        
    if "6" in wf: wf["6"]["inputs"]["text"] = prompt_text
    
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
            
            save_name = f"{shot_id}_v{int(time.time())}.png"
            local_path = os.path.join(pm.get_path("images"), save_name)
            comfy.download_file(fname, subfolder, "image", local_path)
            
            # Update DF
            pm.df.at[row_idx[0], 'Image_Path'] = local_path
            pm.save_data()
            
            return local_path
            
    except Exception as e:
        return f"Error: {e}"
    return "Failed"

def generate_video_for_shot(shot_id):
    comfy = ComfyBridge()
    if not comfy.connect(): return "ComfyUI Offline"
    
    try:
        row_idx = pm.df.index[pm.df['Shot_ID'] == shot_id].tolist()
        if not row_idx: return "Shot ID not found"
        row = pm.df.loc[row_idx[0]]
    except IndexError:
        return "Shot ID not found"

    img_path = row['Image_Path']
    if not img_path or not os.path.exists(img_path):
        return "No Image Generated Yet"

    if row['Type'] == "Action":
        try:
            with open(WORKFLOW_VID_ACTION, 'r') as f:
                wf = json.load(f)
        except FileNotFoundError:
            return f"Workflow file not found: {WORKFLOW_VID_ACTION}"
        
        server_img = comfy.upload_image(img_path)
        if not server_img: return "Failed to upload image to ComfyUI"

        if "113" in wf: wf["113"]["inputs"]["image"] = server_img
        if "195" in wf: wf["195"]["inputs"]["text"] = row['Visual_Prompt']
    
    else:
        try:
            with open(WORKFLOW_VID_VOCAL, 'r') as f:
                wf = json.load(f)
        except FileNotFoundError:
            return f"Workflow file not found: {WORKFLOW_VID_VOCAL}"
            
        # Get path from assets. If vocals not mapped properly, try default name.
        vocals_path = pm.get_asset_path_if_exists("vocals.mp3")
        if not vocals_path: return "Vocals file missing in assets"

        audio = AudioSegment.from_mp3(vocals_path)
        start_ms = row['Start_Time'] * 1000
        end_ms = row['End_Time'] * 1000
        chunk = audio[start_ms:end_ms]
        
        chunk_name = f"{shot_id}_audio.mp3"
        chunk_path = os.path.join(pm.get_path("audio_chunks"), chunk_name)
        chunk.export(chunk_path, format="mp3")
        
        server_audio = comfy.upload_image(chunk_path)
        server_img = comfy.upload_image(img_path)

        if not server_audio or not server_img: return "Failed to upload assets to ComfyUI"
        
        if "12" in wf: wf["12"]["inputs"]["audio"] = server_audio
        if "102" in wf: wf["102"]["inputs"]["value"] = row['Duration']
        if "62" in wf: wf["62"]["inputs"]["image"] = server_img

    try:
        resp = comfy.queue_prompt(wf)
        prompt_id = resp['prompt_id']
        comfy.track_progress(prompt_id)
        
        history = comfy.get_history(prompt_id)[prompt_id]
        outputs = history['outputs']
        vid_node = next((v for k,v in outputs.items() if 'gifs' in v), None)
        
        if vid_node:
            fname = vid_node['gifs'][0]['filename']
            sub = vid_node['gifs'][0]['subfolder']
            
            save_name = f"{shot_id}_video.mp4"
            local_path = os.path.join(pm.get_path("videos"), save_name)
            comfy.download_file(fname, sub, "video", local_path)
            
            pm.df.at[row_idx[0], 'Video_Path'] = local_path
            pm.save_data()
            return local_path
            
    except Exception as e:
        return f"Error: {e}"
    return "Video Generation Failed"

def assemble_video(full_song_path):
    df = pm.df
    clips = []
    
    if df.empty: return "No shots to assemble."

    df = df.sort_values(by="Start_Time")
    
    for index, row in df.iterrows():
        vid_path = row['Video_Path']
        if vid_path and os.path.exists(vid_path):
            try:
                clip = VideoFileClip(vid_path)
                target_dur = row['Duration']
                # If generated video is longer/shorter, handle it
                if clip.duration > target_dur:
                    clip = clip.subclip(0, target_dur)
                clips.append(clip)
            except Exception as e:
                print(f"Error loading clip {vid_path}: {e}")
                dur = row['Duration']
                # Updated ColorClip compatibility
                try:
                    clips.append(ColorClip(size=(1280, 720), color=(0,0,0), duration=dur))
                except:
                    # Fallback for old MoviePy
                    clips.append(ColorClip(size=(1280, 720), col=(0,0,0), duration=dur))
        else:
            dur = row['Duration']
            try:
                clips.append(ColorClip(size=(1280, 720), color=(0,0,0), duration=dur))
            except:
                clips.append(ColorClip(size=(1280, 720), col=(0,0,0), duration=dur))
            
    if not clips: return "No valid clips found."

    final = concatenate_videoclips(clips, method="compose")
    
    # Use provided song if available, otherwise check assets, otherwise use vocals
    audio_path = None
    if full_song_path and os.path.exists(full_song_path):
        audio_path = full_song_path
    else:
        # Fallback to asset store
        audio_path = pm.get_asset_path_if_exists("full_song.mp3")
        if not audio_path:
             audio_path = pm.get_asset_path_if_exists("vocals.mp3")
    
    if audio_path and os.path.exists(audio_path):
        try:
            audio = AudioFileClip(audio_path)
            if audio.duration > final.duration:
                audio = audio.subclip(0, final.duration)
            final = final.set_audio(audio)
        except Exception as e:
            print(f"Audio attach failed: {e}")
        
    out_path = os.path.join(pm.get_path("renders"), "final_cut.mp4")
    final.write_videofile(out_path, fps=24, codec='libx264', audio_codec='aac')
    return out_path

# ==========================================
# GRADIO UI
# ==========================================

DEFAULT_CONCEPT_PROMPT = (
    "Context: The overarching plot is: {plot}\n"
    "Current Shot Info: Timestamp {start}s, Duration {duration}s, Type: {type}.\n"
    "Task: Describe the visual content of the first frame for this specific shot based on the plot. "
    "If it is a 'Vocal' shot, focus on the singer. If 'Action', focus on the narrative/atmosphere."
)

with gr.Blocks(title="Music Video AI Studio", theme=gr.themes.Default()) as app:
    gr.Markdown("# 汐 AI Music Video Director")
    
    # Critical State Variable to track active project name
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
                    project_dropdown = gr.Dropdown(
                        label="Select Existing Project", 
                        choices=get_existing_projects(),
                        interactive=True
                    )
                    refresh_proj_btn = gr.Button("売", size="sm")
                
                load_btn = gr.Button("Load Selected Project")
        
        proj_status = gr.Textbox(label="System Status", interactive=False)
        
        gr.Markdown("### Assets")
        with gr.Row():
            vocals_up = gr.File(label="Upload Vocals (MP3)", file_types=[".mp3"], interactive=True)
            song_up = gr.File(label="Upload Full Song (MP3)", file_types=[".mp3"], interactive=True)
            lyrics_in = gr.Textbox(label="Lyrics", lines=5, interactive=True)
            
        save_proj_btn = gr.Button("沈 Save Project Changes (Assets & Lyrics)", variant="secondary")

# --- TAB 2: STORYBOARD ---
    with gr.Tab("2. Storyboard (The Brain)"):
        
        with gr.Accordion("Step 1: Timeline Settings", open=True):
            with gr.Row():
                min_silence_sl = gr.Slider(500, 2000, value=700, label="Min Silence (ms)")
                silence_thresh_sl = gr.Slider(-60, -20, value=-45, label="Silence Threshold (dB)")
            
            with gr.Row():
                shot_mode_drp = gr.Dropdown(["Fixed", "Random"], value="Random", label="Action Shot Mode")
                min_shot_dur = gr.Slider(1, 5, value=2, label="Min Duration (s)")
                max_shot_dur = gr.Slider(1, 5, value=4, label="Max Duration (s) [Hard Limit 5s]")
            
            scan_btn = gr.Button("1. Scan Vocals & Build Timeline", variant="primary")
        
        with gr.Accordion("Step 2: Plot & Concept Generation", open=True):
            with gr.Row():
                llm_dropdown = gr.Dropdown(choices=["qwen3-vl-8b-instruct-abliterated-v2.0"], label="Select LLM Model", interactive=True)
                refresh_llm_btn = gr.Button("売", size="sm")
                
            rough_concept_in = gr.Textbox(label="Rough User Concept / Vibe", placeholder="e.g. A cyberpunk rainstorm, emotional, neon lights...")
            gen_plot_btn = gr.Button("2. Generate Overarching Plot")
            
            plot_out = gr.Textbox(label="Overarching Plot (Editable)", lines=6, interactive=True)
            
            with gr.Accordion("Advanced: Concept Prompt Template", open=False):
                prompt_template_in = gr.Textbox(
                    value=DEFAULT_CONCEPT_PROMPT, 
                    label="Prompt sent to LLM for each shot", 
                    lines=4,
                    info="Variables: {plot}, {type}, {start}, {duration}"
                )

            with gr.Row():
                gen_concepts_btn = gr.Button("3. Generate/Resume Concepts", variant="primary")
                stop_concepts_btn = gr.Button("Stop Generation", variant="stop")
            
            gr.Markdown("---")
            save_tab2_btn = gr.Button("Save Tab 2 Settings (Timeline & Prompts)", variant="secondary")
        
        with gr.Row():
            regen_shot_id = gr.Textbox(label="Shot ID to Regenerate", placeholder="S005")
            regen_single_btn = gr.Button("Regenerate Single Shot")

        shot_table = gr.Dataframe(
            headers=["Shot_ID", "Type", "Start_Time", "End_Time", "Duration", "Concept", "Visual_Prompt"],
            interactive=True,
            wrap=True
        )

# --- TAB 3, 4, 5 (Unchanged) ---
    with gr.Tab("3. Image Generation"):
        with gr.Row():
            shot_sel_img = gr.Textbox(label="Shot ID (e.g., S001)")
            gen_img_btn = gr.Button("Generate Image", variant="primary")
        img_preview = gr.Image(label="Generated Storyboard Frame")
        gen_img_btn.click(generate_image_for_shot, inputs=[shot_sel_img], outputs=[img_preview])

    with gr.Tab("4. Video Generation"):
        with gr.Row():
            shot_sel_vid = gr.Textbox(label="Shot ID (e.g., S001)")
            gen_vid_btn = gr.Button("Generate Video", variant="primary")
        vid_preview = gr.Video(label="Generated Clip")
        gen_vid_btn.click(generate_video_for_shot, inputs=[shot_sel_vid], outputs=[vid_preview])

    with gr.Tab("5. Assembly"):
        with gr.Row():
            assemble_btn = gr.Button("Assemble Final Video", variant="primary")
        final_video_out = gr.Video(label="Final Cut")
        def run_assembly_wrapper(song_file):
            path = song_file.name if song_file else None
            return assemble_video(path)
        assemble_btn.click(run_assembly_wrapper, inputs=[song_up], outputs=[final_video_out])

# ==========================================
# GLOBAL LOGIC & WIRING
# ==========================================

    # --- HANDLERS ---
    
    def handle_create(name):
        msg = pm.create_project(name)
        # Return project name to state if successful
        proj_state = pm.sanitize_name(name) if "created" in msg else ""
        return msg, gr.Dropdown(choices=get_existing_projects()), proj_state

    def handle_load(name):
        # 1. Load Dataframe
        msg, df = pm.load_project(name)
        
        # 2. Load Assets
        lyrics = pm.get_lyrics()
        v_path = pm.get_asset_path_if_exists("vocals.mp3")
        s_path = pm.get_asset_path_if_exists("full_song.mp3")
        
        # 3. Load Tab 2 Settings
        settings = pm.load_project_settings()
        
        # Extract with defaults
        s_min_sil = settings.get("min_silence", 700)
        s_sil_thresh = settings.get("silence_thresh", -45)
        s_mode = settings.get("shot_mode", "Random")
        s_min_dur = settings.get("min_dur", 2)
        s_max_dur = settings.get("max_dur", 4)
        s_llm = settings.get("llm_model", "qwen3-vl-8b-instruct-abliterated-v2.0")
        s_concept = settings.get("rough_concept", "")
        s_plot = settings.get("plot", "")
        s_prompt = settings.get("prompt_template", DEFAULT_CONCEPT_PROMPT)
        
        # Return project name to state
        return (
            msg, df, lyrics, v_path, s_path, # Tab 1 outputs
            s_min_sil, s_sil_thresh, s_mode, s_min_dur, s_max_dur, # Tab 2 Timeline
            s_llm, s_concept, s_plot, s_prompt, # Tab 2 Prompts
            name # Update state
        )

    def handle_save_changes(project_state_name, lyrics_text, v_file, s_file):
        # Ensure PM is synced with state
        if not project_state_name:
             return "No project active. Load or create one first.", None, None
        
        pm.current_project = project_state_name
        pm.save_lyrics(lyrics_text)
        
        # Handle Vocals File
        v_path = None
        if v_file:
            if isinstance(v_file, str):
                v_path = v_file # Already a path
            elif hasattr(v_file, 'name'):
                v_path = pm.save_asset(v_file.name, "vocals.mp3")
        
        # Handle Song File
        s_path = None
        if s_file:
            if isinstance(s_file, str):
                s_path = s_file # Already a path
            elif hasattr(s_file, 'name'):
                s_path = pm.save_asset(s_file.name, "full_song.mp3")
                
        return f"Saved assets for '{project_state_name}'.", v_path, s_path

    def handle_save_tab2(project_state_name, min_sil, sil_thresh, mode, min_d, max_d, llm, concept, plot, template):
        if not project_state_name: return "No active project."
        pm.current_project = project_state_name
        
        settings = {
            "min_silence": min_sil,
            "silence_thresh": sil_thresh,
            "shot_mode": mode,
            "min_dur": min_d,
            "max_dur": max_d,
            "llm_model": llm,
            "rough_concept": concept,
            "plot": plot,
            "prompt_template": template
        }
        return pm.save_project_settings(settings)

    # --- EVENTS ---

    # 1. Project Management
    create_btn.click(
        handle_create, 
        inputs=proj_name, 
        outputs=[proj_status, project_dropdown, current_proj_var]
    )
    
    refresh_proj_btn.click(lambda: gr.Dropdown(choices=get_existing_projects()), outputs=project_dropdown)

    # 2. LOAD
    load_btn.click(
        handle_load, 
        inputs=project_dropdown, 
        outputs=[
            proj_status, shot_table, lyrics_in, vocals_up, song_up, # Tab 1
            min_silence_sl, silence_thresh_sl, shot_mode_drp, min_shot_dur, max_shot_dur, # Tab 2 Timeline
            llm_dropdown, rough_concept_in, plot_out, prompt_template_in, # Tab 2 Prompts
            current_proj_var # Update State
        ]
    )

    # 3. SAVE Handlers
    # Corrected: Now uses current_proj_var instead of implicitly relying on global pm
    save_proj_btn.click(
        handle_save_changes, 
        inputs=[current_proj_var, lyrics_in, vocals_up, song_up], 
        outputs=[proj_status, vocals_up, song_up]
    )
    
    save_tab2_btn.click(
        handle_save_tab2,
        inputs=[
            current_proj_var,
            min_silence_sl, silence_thresh_sl, shot_mode_drp, min_shot_dur, max_shot_dur,
            llm_dropdown, rough_concept_in, plot_out, prompt_template_in
        ],
        outputs=proj_status
    )

    # 4. Tab 2 Logic
    def refresh_models():
        llm = LLMBridge()
        models = llm.get_models()
        return gr.Dropdown(choices=models, value=models[0] if models else "")
    refresh_llm_btn.click(refresh_models, outputs=llm_dropdown)

    def run_scan(v_file, p_name, m_sil, s_thr, s_mode, min_d, max_d):
        if not p_name: 
            return pd.DataFrame()
        
        pm.current_project = p_name
        
        # Logic to handle both string paths (loaded project) and file objects (new upload)
        final_v_path = None
        if v_file is None:
            final_v_path = pm.get_asset_path_if_exists("vocals.mp3")
        elif isinstance(v_file, str):
            final_v_path = v_file
        elif hasattr(v_file, 'name'):
            final_v_path = pm.save_asset(v_file.name, "vocals.mp3")
            
        return scan_vocals_advanced(final_v_path, p_name, m_sil, s_thr, s_mode, min_d, max_d)

    # Corrected: Input includes current_proj_var instead of the 'proj_name' textbox
    scan_btn.click(
        run_scan, 
        inputs=[vocals_up, current_proj_var, min_silence_sl, silence_thresh_sl, shot_mode_drp, min_shot_dur, max_shot_dur], 
        outputs=shot_table
    )

    gen_plot_btn.click(generate_overarching_plot, inputs=[rough_concept_in, lyrics_in, llm_dropdown], outputs=plot_out)
    
    gen_concepts_btn.click(
        generate_concepts_logic, 
        inputs=[plot_out, prompt_template_in, llm_dropdown], 
        outputs=shot_table
    )
    
    stop_concepts_btn.click(stop_gen, outputs=[])
    
    regen_single_btn.click(
        generate_concepts_logic,
        inputs=[plot_out, prompt_template_in, llm_dropdown, regen_shot_id],
        outputs=shot_table
    )

if __name__ == "__main__":
    app.launch()