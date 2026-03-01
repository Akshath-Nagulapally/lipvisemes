# Key MediaPipe lip indices
UPPER_LIP_TOP    = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
LOWER_LIP_BOTTOM = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

# Inner lip (most useful for openness)
UPPER_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
LOWER_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]

# Corners
LEFT_CORNER  = 61
RIGHT_CORNER = 291

# Midpoints
UPPER_MID = 13   # inner upper lip center
LOWER_MID = 14   # inner lower lip center


import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

from features import get_lip_ratio_vector
from reference_target_labels import (
    REFERENCE_FEATURES,
    UNIVERSAL_MATCH_THRESHOLD,
    get_label_by_viseme,
)

# Load .env from the same directory as this script so GROQ_API_KEY is set
load_dotenv(Path(__file__).resolve().parent / ".env")


# MediaPipe 0.10.30+ uses Tasks API; model must be present
MODEL_PATH = Path(__file__).resolve().parent / "face_landmarker.task"
if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Face landmarker model not found at {MODEL_PATH}. "
        "Download it from: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    )

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def _create_face_landmarker():
    """Create a new FaceLandmarker. Use one per video so timestamps are monotonic per stream."""
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
    )
    return FaceLandmarker.create_from_options(options)

def get_lip_features(landmarks, img_w, img_h):
    def pt(idx):
        l = landmarks[idx]
        return np.array([l.x * img_w, l.y * img_h])
    
    # Mouth width
    left  = pt(61)
    right = pt(291)
    width = np.linalg.norm(right - left)
    
    # Mouth height (inner lips)
    upper = pt(13)
    lower = pt(14)
    height = np.linalg.norm(lower - upper)
    
    # Mouth Aspect Ratio
    MAR = height / (width + 1e-6)
    
    # Lip roundness: compare width to inter-lip distance
    # Small width + small height = pursed (OOH)
    
    # Normalized width relative to face
    # Use face width as reference (landmarks 234, 454 = cheeks)
    face_left  = pt(234)
    face_right = pt(454)
    face_width = np.linalg.norm(face_right - face_left)
    norm_width = width / (face_width + 1e-6)
    
    return {
        "MAR": MAR,
        "norm_width": norm_width,
        "height": height,
        "width": width
    }


def classify_viseme(features):
    MAR        = features["MAR"]
    norm_width = features["norm_width"]
    
    if MAR < 0.03:
        return "MBP"          # closed: m, b, p
    elif MAR > 0.35:
        return "AH"           # wide open: ah, aa
    elif MAR > 0.2:
        if norm_width < 0.35:
            return "OH"       # round open: oh
        else:
            return "AH"
    elif norm_width > 0.55 and MAR < 0.15:
        return "EEE"          # wide/spread: ee, ih
    elif norm_width < 0.35 and MAR < 0.15:
        return "OOH"          # pursed: oo, w, uw
    elif MAR < 0.1:
        return "REST"         # nearly closed
    else:
        return "REST"


def classify_viseme_by_reference(
    vec: np.ndarray,
    reference_features: dict,
    threshold: float,
) -> str:
    """Classify by nearest reference vector (Euclidean distance). If min distance > threshold, return S."""
    if not reference_features:
        return "S"
    vec = np.asarray(vec, dtype=np.float64)
    best_viseme = "S"
    best_dist = float("inf")
    for viseme, ref_vec in reference_features.items():
        if not ref_vec or len(ref_vec) != len(vec):
            continue
        ref = np.asarray(ref_vec, dtype=np.float64)
        dist = np.linalg.norm(vec - ref)
        if dist < best_dist:
            best_dist = dist
            best_viseme = viseme
    if best_dist > threshold:
        return "S"
    return best_viseme

# Minimum consecutive frames the same viseme must win before we commit/output it (1 = record every change).
STABLE_FRAMES = 1

# Single container for all decodes when Supermemory is used (no per-user IDs).
SUPERMEMORY_CONTAINER_TAG = "lipvisemes"


def _get_supermemory_context(container_tag: str, query: str) -> str:
    """Fetch user profile + relevant memories from Supermemory. Returns a context string or empty if unavailable."""
    import os
    api_key = os.environ.get("SUPERMEMORY_API_KEY")
    if not api_key:
        return ""
    try:
        from supermemory import Supermemory
        client = Supermemory(api_key=api_key)
        profile_resp = client.profile(container_tag=container_tag, q=query)
        parts = []
        if getattr(profile_resp, "profile", None):
            p = profile_resp.profile
            if getattr(p, "static", None) and p.static:
                parts.append("Static profile:\n" + "\n".join(p.static))
            if getattr(p, "dynamic", None) and p.dynamic:
                parts.append("Dynamic profile:\n" + "\n".join(p.dynamic))
        if getattr(profile_resp, "search_results", None) and profile_resp.search_results:
            results = getattr(profile_resp.search_results, "results", None) or []
            memories = []
            for r in results:
                mem = getattr(r, "memory", None) or (r.get("memory", "") if isinstance(r, dict) else "")
                if mem:
                    memories.append(mem)
            if memories:
                parts.append("Relevant memories:\n" + "\n".join(memories))
        if not parts:
            return ""
        return "User context (use to bias decoding toward this user's vocabulary and phrasing):\n" + "\n\n".join(parts)
    except Exception:
        return ""


def decode(sequence: list[str], *, silent: bool = False) -> str:
    """
    Decode the viseme/phoneme sequence into an actual sentence using an LLM.
    Requires GROQ_API_KEY in the environment. Returns the decoded string only.
    When SUPERMEMORY_API_KEY is set, uses Supermemory (single shared container) to
    inject profile + memories and to store each decoding.
    """
    if not sequence:
        return ""
    import os
    container_tag = SUPERMEMORY_CONTAINER_TAG if os.environ.get("SUPERMEMORY_API_KEY") else None
    raw = ", ".join(sequence)
    prompt = """You are a speech decoding assistant. You are given a sequence of phoneme-like tokens produced by a lip-reading system (viseme-to-phoneme). The format is:
- Comma-separated list of tokens.
- A token that is a single space " " indicates silence or a pause between words.
- A token like "f or v" means the lip shape could be either sound; choose the one that fits the intended word.
- A single token like "ah", "aw", "sil" is one phoneme/sound.

The speaker is always trying to say a normal, everyday English sentence—never adversarial or obscure. Decode the sequence into the single most likely sentence. If the sequence is ambiguous or unclear, output one plausible, typical sentence that fits the tokens. Your response must be exactly one sentence maximum: no lists, no multiple sentences, no preamble or explanation. Output only that one sentence.

Sequence:
"""
    prompt += raw.strip()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return ""

    messages = []
    if container_tag:
        supermemory_ctx = _get_supermemory_context(container_tag, raw)
        if supermemory_ctx:
            messages.append({"role": "system", "content": supermemory_ctx})
    messages.append({"role": "user", "content": prompt})

    from groq import Groq
    client = Groq(api_key=api_key)
    decoded: list[str] = []
    last_error: Exception | None = None
    for attempt in range(2):  # retry once on transient failure
        try:
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=1,
                max_completion_tokens=1024,
                top_p=1,
                stream=True,
                stop=None,
            )
            for chunk in completion:
                if chunk.choices:
                    part = chunk.choices[0].delta.content or ""
                    decoded.append(part)
                    if not silent:
                        print(part, end="", flush=True)
            if not silent:
                print(flush=True)
            break
        except Exception as e:
            last_error = e
            if attempt == 0:
                continue
            return ""
    result = "".join(decoded)
    if container_tag and result:
        try:
            from supermemory import Supermemory
            sm = Supermemory(api_key=os.environ["SUPERMEMORY_API_KEY"])
            sm.add(
                content=f"Viseme sequence: {raw}\nDecoded sentence: {result.strip()}",
                container_tag=container_tag,
            )
        except Exception:
            pass
    return result


def _process_frames(cap, face_landmarker) -> list[str]:
    """Run viseme pipeline on video capture; return phoneme token sequence. Uses a dedicated landmarker so timestamps are monotonic per video."""
    history = []
    SMOOTH_N = 3
    frame_timestamp_ms = 0
    last_committed: str | None = None
    stable_count = 0
    text_sequence: list[str] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = face_landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        frame_timestamp_ms += 33

        if result.face_landmarks:
            lms = result.face_landmarks[0]
            vec = get_lip_ratio_vector(lms, w, h)
            if REFERENCE_FEATURES:
                viseme = classify_viseme_by_reference(
                    vec, REFERENCE_FEATURES, UNIVERSAL_MATCH_THRESHOLD
                )
            else:
                features = get_lip_features(lms, w, h)
                viseme = classify_viseme(features)

            history.append(viseme)
            if len(history) > SMOOTH_N:
                history.pop(0)
            smoothed = max(set(history), key=history.count)

            if smoothed == last_committed:
                stable_count += 1
            else:
                stable_count = 1
            if stable_count >= STABLE_FRAMES and smoothed != last_committed:
                last_committed = smoothed
                label = get_label_by_viseme(smoothed)
                phonemes = label["phonemes"] if label else []  # type: ignore
                if not isinstance(phonemes, list):
                    phonemes = [str(phonemes)]
                if smoothed == "S":
                    token = " "
                elif len(phonemes) > 1:
                    token = " or ".join(phonemes)
                else:
                    token = phonemes[0] if phonemes else "?"
                text_sequence.append(token)
    return text_sequence


def decode_mp4(mp4_path: str) -> str:
    """
    Take an MP4 file path, run the viseme pipeline on its frames, decode with the LLM,
    and return only the decoded string. No recording, no display.
    Uses a fresh FaceLandmarker per video so MediaPipe timestamps are monotonic.
    Supermemory is used automatically when SUPERMEMORY_API_KEY is set (single shared container).
    """
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        return ""
    face_landmarker = _create_face_landmarker()
    try:
        text_sequence = _process_frames(cap, face_landmarker)
        return decode(text_sequence, silent=True)
    finally:
        cap.release()
        face_landmarker.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python lipvisemes.py <path_to.mp4>", file=sys.stderr)
        sys.exit(1)
    result = decode_mp4(sys.argv[1])
    print(result)