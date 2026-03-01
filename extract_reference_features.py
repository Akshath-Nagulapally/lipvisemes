"""
Build the viseme reference library: run MediaPipe on each reference image,
compute lip ratio vectors, average per viseme, and write reference_features.json.
Run from lipvisemes dir: uv run python extract_reference_features.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# Ensure we can import from same package
sys.path.insert(0, str(Path(__file__).resolve().parent))

from reference_target_labels import REFERENCE_IMAGES
from features import get_lip_ratio_vector

MODEL_PATH = Path(__file__).resolve().parent / "face_landmarker.task"
OUTPUT_PATH = Path(__file__).resolve().parent / "reference_features.json"


def main() -> None:
    if not MODEL_PATH.is_file():
        print(f"Model not found: {MODEL_PATH}", file=sys.stderr)
        sys.exit(1)

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
    )
    landmarker = FaceLandmarker.create_from_options(options)

    # Collect ratio vectors per viseme (list of vectors per viseme)
    vectors_by_viseme: dict[str, list[list[float]]] = {
        viseme: [] for viseme in REFERENCE_IMAGES
    }

    for viseme, paths in REFERENCE_IMAGES.items():
        for path in paths:
            if not path.is_file():
                continue
            img = cv2.imread(str(path))
            if img is None:
                print(f"Warning: could not read {path}", file=sys.stderr)
                continue
            h, w = img.shape[:2]
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)
            if not result.face_landmarks:
                print(f"Warning: no face in {path}", file=sys.stderr)
                continue
            lms = result.face_landmarks[0]
            vec = get_lip_ratio_vector(lms, w, h)
            vectors_by_viseme[viseme].append(vec.tolist())

    landmarker.close()

    # Average per viseme -> one vector per viseme (list of floats)
    reference_features: dict[str, list[float]] = {}
    for viseme, vecs in vectors_by_viseme.items():
        if not vecs:
            print(f"Warning: no valid vectors for viseme {viseme}", file=sys.stderr)
            reference_features[viseme] = []
            continue
        arr = np.array(vecs)
        mean_vec = np.mean(arr, axis=0)
        reference_features[viseme] = mean_vec.tolist()

    OUTPUT_PATH.write_text(json.dumps(reference_features, indent=2))
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
