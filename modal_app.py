"""
Modal FastAPI endpoint: POST an MP4 file, get back the decoded string only.
Requires Modal Secrets:
  - "openai-api-key" with key OPENAI_API_KEY
  - "supermemory-api-key" with key SUPERMEMORY_API_KEY
"""
import os
import sys
import tempfile
from pathlib import Path

import modal
from fastapi import File, UploadFile

app = modal.App("lipvisemes")

# All packages needed for lipvisemes + FastAPI
# Install opencv-python-headless first so mediapipe doesn't pull in full opencv-python (which needs libGL).
# If cv2 still loads libGL, provide the lib so the container can load it.
lipvisemes_image = (
    modal.Image.debian_slim()
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .uv_pip_install(
        "opencv-python-headless",
        "mediapipe",
        "openai",
        "supermemory",
        "python-dotenv",
        "numpy",
        "fastapi[standard]",
    )
    .add_local_dir(Path(__file__).parent, "/app")
)

OPENAI_SECRET = modal.Secret.from_name("openai-api-key", required_keys=["OPENAI_API_KEY"])
SUPERMEMORY_SECRET = modal.Secret.from_name("supermemory-api-key", required_keys=["SUPERMEMORY_API_KEY"])


@app.function(
    image=lipvisemes_image,
    secrets=[OPENAI_SECRET, SUPERMEMORY_SECRET],
    timeout=300,
)
@modal.fastapi_endpoint(method="POST")
async def decode_video(file: UploadFile = File(...)):
    """Accept an MP4 file upload. Returns 200 with JSON: {"decoded": "<sentence>", "error": null} on success, or {"decoded": "", "error": "<message>"} on failure. Supermemory is used automatically when the secret is set. Use a long timeout (e.g. 90s) and retry once on failure."""
    try:
        sys.path.insert(0, "/app")
        os.chdir("/app")
        from lipvisemes import decode_mp4

        suffix = ".mp4"
        if file.filename and file.filename.lower().endswith(".mp4"):
            suffix = ".mp4"
        contents = await file.read()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(contents)
            path = f.name
        try:
            decoded = decode_mp4(path)
            return {"decoded": decoded or "", "error": None}
        finally:
            os.unlink(path)
    except Exception as e:
        return {"decoded": "", "error": str(e)}
