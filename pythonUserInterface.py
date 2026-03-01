"""
Camera Recording Framework - Wispr-style
-----------------------------------------
Hold RIGHT_SHIFT (or your chosen hotkey) → camera records
Release key → saves MP4 locally → video is sent to lip-visemes API → decoded text is typed into active field

Dependencies:
    pip install pynput opencv-python pyautogui requests

On macOS you'll need to grant:
  - Accessibility permissions (for pynput global hotkeys + pyautogui typing)
  - Camera permissions (for opencv)
"""

import cv2
import threading
import time
import os
from datetime import datetime
from pynput import keyboard
import pyautogui
import requests

# ─── Config ───────────────────────────────────────────────────────────────────

TRIGGER_KEY = keyboard.Key.shift_r
OUTPUT_DIR = os.path.expanduser("~/Desktop/recordings")
CAMERA_INDEX = 1               # 0 = often iPhone/Continuity; 1 = usually Mac built-in
FPS = 30.0
RESOLUTION = (1280, 720)

# Video → text API (Modal lip-visemes decode)
VIDEO_TO_TEXT_URL = "https://akshathnag06--videototext-decode-video.modal.run"

# ─── State ────────────────────────────────────────────────────────────────────

is_recording = False
recording_thread = None
stop_event = threading.Event()
current_output_path = None

# ─── Recording ────────────────────────────────────────────────────────────────

def record_video(output_path: str, stop_event: threading.Event):
    """Captures webcam frames until stop_event is set, then saves to MP4."""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[!] Could not open camera. Check index (try 0 or 1) and permissions.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
    time.sleep(0.3)  # let camera warm up

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, FPS, RESOLUTION)

    print(f"[●] Recording started → {output_path}")

    fail_count = 0
    max_fail = 30  # allow a few bad reads (e.g. camera busy)
    frame_interval = 1.0 / FPS  # real-time so timestamps are strictly monotonic for decode API
    last_write_time = time.monotonic()
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            fail_count += 1
            if fail_count >= max_fail:
                print("[!] Camera read failed repeatedly, stopping.")
                break
            time.sleep(0.02)
            continue
        fail_count = 0
        now = time.monotonic()
        if now - last_write_time >= frame_interval:
            writer.write(frame)
            last_write_time = now

    cap.release()
    writer.release()
    print(f"[✓] Recording saved → {output_path}")


def start_recording():
    global is_recording, recording_thread, stop_event, current_output_path

    if is_recording:
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_output_path = os.path.join(OUTPUT_DIR, f"recording_{timestamp}.mp4")

    stop_event = threading.Event()
    recording_thread = threading.Thread(
        target=record_video,
        args=(current_output_path, stop_event),
        daemon=True
    )
    recording_thread.start()
    is_recording = True


def stop_recording():
    global is_recording

    if not is_recording:
        return

    stop_event.set()
    recording_thread.join(timeout=5)
    is_recording = False

    # Small delay so the previously focused field stays active
    time.sleep(0.1)

    decoded = decode_video_to_text(current_output_path)
    if decoded:
        inject_text(decoded)
    else:
        inject_text("(decode failed)")


# ─── Video → Text (Modal API) ─────────────────────────────────────────────────

def decode_video_to_text(video_path: str) -> str:
    """POST video to lip-visemes API; returns the 'decoded' text or empty string on failure."""
    if not os.path.isfile(video_path):
        print(f"[!] Video file not found: {video_path}")
        return ""
    try:
        with open(video_path, "rb") as f:
            r = requests.post(
                VIDEO_TO_TEXT_URL,
                files={"file": (os.path.basename(video_path), f, "video/mp4")},
                timeout=60,
            )
        if r.status_code != 200:
            print(f"[!] Decode API HTTP {r.status_code}: {r.text[:500]}")
            return ""
        data = r.json()
        decoded = data.get("decoded", "").strip() or ""
        if not decoded:
            print(f"[!] Decode API returned no 'decoded' field: {data}")
        return decoded
    except requests.exceptions.RequestException as e:
        print(f"[!] Decode API request error: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"    Response: {e.response.status_code} {e.response.text[:300]}")
        return ""
    except Exception as e:
        print(f"[!] Decode API error: {e}")
        return ""


# ─── Text Injection ───────────────────────────────────────────────────────────

def inject_text(text: str):
    """Types text into whatever field currently has focus."""
    print(f"[→] Injecting: '{text}'")
    pyautogui.typewrite(text, interval=0.03)
    # For unicode / non-ASCII text, use pyautogui.write or pyperclip + paste:
    # import pyperclip
    # pyperclip.copy(text)
    # pyautogui.hotkey("cmd", "v")   # macOS paste


# ─── Hotkey Listener ─────────────────────────────────────────────────────────

def on_press(key):
    if key == TRIGGER_KEY:
        start_recording()


def on_release(key):
    if key == TRIGGER_KEY:
        stop_recording()


# ─── Main ─────────────────────────────────────────────────────────────────────

if _name_ == "_main_":
    print("=" * 50)
    print("  Camera Recording Framework")
    print(f"  Hold [{TRIGGER_KEY}] to record, release to save + inject text")
    print(f"  Recordings saved to: {OUTPUT_DIR}")
    print("  Press Ctrl+C to quit")
    print("=" * 50)

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        try:
            listener.join()
        except KeyboardInterrupt:
            print("\n[✓] Exiting")