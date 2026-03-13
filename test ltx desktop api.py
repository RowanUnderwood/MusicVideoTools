import requests
import time

# Base URL for the local LTX desktop server
BASE_URL = "http://127.0.0.1:8000/api"

def generate_video():
    """Triggers a video generation and waits for completion."""
    
    # 1. Define the payload based on GenerateVideoRequest in api_types.py
    payload = {
        "prompt": "A cinematic, slow-motion shot of a cyberpunk city at night, neon lights reflecting in rain puddles.",
        "resolution": "1080p",
        "aspectRatio": "16:9",
        "model": "fast",
        "duration": "2",
        "fps": "24",
        "cameraMotion": "dolly_in"
    }
    
    print(f"Sending generation request: {payload['prompt']}")
    
    # 2. Send the POST request to start generation
    try:
        response = requests.post(f"{BASE_URL}/generate", json=payload)
        response.raise_for_status()
        print("Generation started successfully.")
    except requests.exceptions.RequestException as e:
        print(f"Failed to start generation: {e}")
        if e.response is not None:
            print(f"Server replied: {e.response.text}")
        return

    # 3. Poll the progress endpoint until complete
        print("Polling for progress...")
    while True:
        try:
            prog_resp = requests.get(f"{BASE_URL}/generation/progress")
            prog_resp.raise_for_status()
            data = prog_resp.json()
            
            status = data.get("status")
            phase = data.get("phase")
            progress = data.get("progress", 0)
            
            # Print a clean progress update
            print(f"Status: {status} | Phase: {phase} | Progress: {progress}%", end="\r")
            
            if status == "complete":
                print("\nGeneration complete!")
                break
            elif status in ["error", "cancelled"]:
                print(f"\nGeneration stopped with status: {status}")
                break
                
        except requests.exceptions.RequestException as e:
            print(f"\nError checking progress: {e}")
            break
            
        time.sleep(2) # Wait 2 seconds before checking again

if __name__ == "__main__":
    generate_video()