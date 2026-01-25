import os
import json
import requests
import websocket
import urllib.request
import math
from pydub import AudioSegment, silence
from moviepy.editor import VideoFileClip, AudioFileClip, ColorClip, CompositeVideoClip

# --- Configuration ---
COMFY_URL = "127.0.0.1:8188"
CLIENT_ID = "Ride_Or_Die_Video_Gen"
INPUT_DIR = os.getcwd()
OUTPUT_SUBDIR = os.path.join(INPUT_DIR, "video_segments")
FINAL_VIDEO_NAME = "final_synced_video.mp4"

# File Names
VOCALS_FILE = "vocals.mp3"
FULL_BAND_FILE = "Messing with my ride.mp3"
IMAGE_FILE = "Gemini_Generated_Image_s6wzbus6wzbus6wz.png"
WORKFLOW_FILE = "011426-LTX2-AudioSync-i2v-Ver2-Jakes Version API.json"

# Settings
FPS = 24
MAX_CHUNK_SECONDS = 19
# Silence Detection Settings (Tune these based on your specific audio levels)
MIN_SILENCE_LEN = 700   # (ms) Ignore gaps shorter than this (treat as breaths)
SILENCE_THRESH = -45    # (dBFS) Anything quieter than this is "silence"

os.makedirs(OUTPUT_SUBDIR, exist_ok=True)

# --- ComfyUI API Helpers ---
def queue_prompt(prompt_workflow):
    p = {"prompt": prompt_workflow, "client_id": CLIENT_ID}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request(f"http://{COMFY_URL}/prompt", data=data)
    return json.loads(urllib.request.urlopen(req).read())

def get_history(prompt_id):
    with urllib.request.urlopen(f"http://{COMFY_URL}/history/{prompt_id}") as response:
        return json.loads(response.read())

def upload_file(file_path):
    with open(file_path, 'rb') as f:
        response = requests.post(
            f"http://{COMFY_URL}/upload/image", 
            files={'image': f},
            data={'overwrite': 'true'}
        )
    return response.json()

def track_progress(prompt_id, ws):
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break
        else:
            continue

# --- Main Logic ---

def analyze_and_split_audio():
    print(f"Analyzing {VOCALS_FILE} for vocal phrases...")
    audio = AudioSegment.from_mp3(VOCALS_FILE)
    
    # 1. Detect non-silent chunks (phrases)
    # Returns list of [start_ms, end_ms]
    nonsilent_ranges = silence.detect_nonsilent(
        audio, 
        min_silence_len=MIN_SILENCE_LEN, 
        silence_thresh=SILENCE_THRESH
    )

    chunks_metadata = []
    
    print(f"Detected {len(nonsilent_ranges)} vocal phrases.")

    for i, (start_ms, end_ms) in enumerate(nonsilent_ranges):
        # Calculate duration
        duration_ms = end_ms - start_ms
        
        # 2. Check against MAX limit (19 seconds)
        max_ms = MAX_CHUNK_SECONDS * 1000
        
        # If phrase is too long, split it into sub-chunks
        num_splits = math.ceil(duration_ms / max_ms)
        
        for j in range(num_splits):
            chunk_start = start_ms + (j * max_ms)
            chunk_end = min(start_ms + ((j + 1) * max_ms), end_ms)
            
            # Extract audio
            chunk_audio = audio[chunk_start:chunk_end]
            
            # Calculate Frame timings
            # Start/End Frame relative to the WHOLE video
            global_start_frame = int((chunk_start / 1000.0) * FPS)
            global_end_frame = int((chunk_end / 1000.0) * FPS)
            
            # Naming convention: chunk_{StartFrame}_{EndFrame}.mp3
            filename = f"chunk_{global_start_frame:05d}_{global_end_frame:05d}.mp3"
            filepath = os.path.join(INPUT_DIR, filename)
            
            # Export
            chunk_audio.export(filepath, format="mp3")
            
            chunks_metadata.append({
                "path": filepath,
                "filename": filename,
                "duration_sec": len(chunk_audio) / 1000.0,
                "start_time_sec": chunk_start / 1000.0,
                "start_frame": global_start_frame,
                "end_frame": global_end_frame
            })

    return chunks_metadata, audio.duration_seconds

def main():
    # 1. Analyze Audio & Create Chunks
    chunks, total_duration = analyze_and_split_audio()
    
    if not chunks:
        print("No vocals detected! Check silence threshold settings.")
        return

    # 2. Setup Comfy Connection
    ws = websocket.WebSocket()
    ws.connect(f"ws://{COMFY_URL}/ws?clientId={CLIENT_ID}")

    # Load Workflow
    with open(WORKFLOW_FILE, 'r') as f:
        workflow_data = json.load(f)
        workflow = workflow_data.get("prompt", workflow_data)

    generated_clips_info = []

    # 3. Process each chunk
    for idx, chunk in enumerate(chunks):
        print(f"Processing chunk {idx+1}/{len(chunks)}: Frames {chunk['start_frame']} - {chunk['end_frame']} ({chunk['duration_sec']}s)")
        
        # Upload
        upload_resp = upload_file(chunk['path'])
        uploaded_name = upload_resp.get("name", chunk['filename'])

        # Update Workflow Nodes
        # Node 12: Load Audio
        workflow["12"]["inputs"]["audio"] = uploaded_name
        
        # Node 102: Audio Length (Float)
        workflow["102"]["inputs"]["value"] = chunk['duration_sec']
        
        # Node 62: Image
        workflow["62"]["inputs"]["image"] = IMAGE_FILE

        # Node 66: Filename Prefix (Include frame info for safety)
        prefix = f"render_{chunk['start_frame']:05d}_"
        workflow["66"]["inputs"]["filename_prefix"] = prefix

        # Execute
        try:
            resp = queue_prompt(workflow)
            prompt_id = resp['prompt_id']
            track_progress(prompt_id, ws)
            
            # Get Result
            history = get_history(prompt_id)[prompt_id]
            outputs = history['outputs']
            
            # Find video output
            node_66 = outputs.get("66", {}).get("gifs", [])
            if not node_66:
                 # Fallback search
                for k, v in outputs.items():
                    if 'gifs' in v:
                         node_66 = v['gifs']
                         break
            
            if node_66:
                vid_info = node_66[0]
                file_url = f"http://{COMFY_URL}/view?filename={vid_info['filename']}&subfolder={vid_info['subfolder']}&type={vid_info['type']}"
                
                # Save locally
                local_path = os.path.join(OUTPUT_SUBDIR, f"clip_{chunk['start_frame']}.mp4")
                urllib.request.urlretrieve(file_url, local_path)
                
                # Store info for assembly
                generated_clips_info.append({
                    "path": local_path,
                    "start_time": chunk['start_time_sec']
                })
            else:
                print(f"Error: No output for chunk {idx}")

        except Exception as e:
            print(f"Failed chunk {idx}: {e}")

    # 4. Assemble Final Video
    print("Assembling final timeline...")
    
    # Create background (Black frames)
    # ColorClip takes size=(W,H), color=(R,G,B), duration=sec
    bg_clip = ColorClip(size=(1920, 1080), color=(0,0,0), duration=total_duration)
    
    # Load all generated clips and set their start times
    video_clips = [bg_clip]
    for info in generated_clips_info:
        clip = VideoFileClip(info['path'])
        # Position this clip on the timeline
        clip = clip.set_start(info['start_time'])
        # Ensure it sits on top of the black background
        clip = clip.set_position("center") 
        video_clips.append(clip)

    # Composite them (layers clips on top of each other)
    final_video = CompositeVideoClip(video_clips)

    # 5. Add Full Audio
    print(f"Syncing full audio: {FULL_BAND_FILE}")
    full_audio = AudioFileClip(FULL_BAND_FILE)
    
    # Trim audio if it's longer than video, or vice versa
    if full_audio.duration > final_video.duration:
        full_audio = full_audio.subclip(0, final_video.duration)
    
    final_video = final_video.set_audio(full_audio)
    
    # 6. Render
    final_video.write_videofile(
        FINAL_VIDEO_NAME, 
        fps=FPS, 
        codec='libx264', 
        audio_codec='aac'
    )
    
    print("Done.")

if __name__ == "__main__":
    main()