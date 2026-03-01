from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional


# Tuning knob for nearest-reference classification.
# Example use: if best_distance > UNIVERSAL_MATCH_THRESHOLD -> return silence/unknown.
UNIVERSAL_MATCH_THRESHOLD = 0.22

REFERENCE_ROOT = Path(__file__).resolve().parent / "images_with_reference"
REFERENCE_FEATURES_PATH = Path(__file__).resolve().parent / "reference_features.json"


def _load_reference_features() -> Dict[str, List[float]]:
    if not REFERENCE_FEATURES_PATH.is_file():
        return {}
    try:
        data = json.loads(REFERENCE_FEATURES_PATH.read_text())
        return {k: list(v) for k, v in data.items() if isinstance(v, list)}
    except (json.JSONDecodeError, OSError):
        return {}


# One averaged lip ratio vector per viseme (loaded from reference_features.json).
REFERENCE_FEATURES: Dict[str, List[float]] = _load_reference_features()


# Canonical viseme metadata keyed by your numeric IDs.
TARGET_LABELS: Dict[int, Dict[str, object]] = {
    0: {
        "id": 0,
        "viseme": "A",
        "description": "Labiodental",
        "phonemes": ["f", "v"],
    },
    1: {
        "id": 1,
        "viseme": "B",
        "description": "Rounded vowels / semivowels",
        "phonemes": ["er", "ow", "r", "w", "uh", "uw", "axr", "ux"],
    },
    2: {
        "id": 2,
        "viseme": "C",
        "description": "Bilabial closure",
        "phonemes": ["b", "p", "m", "em"],
    },
    3: {
        "id": 3,
        "viseme": "D",
        "description": "Diphthong",
        "phonemes": ["aw"],
    },
    4: {
        "id": 4,
        "viseme": "E",
        "description": "Dental",
        "phonemes": ["dh", "th"],
    },
    5: {
        "id": 5,
        "viseme": "F",
        "description": "Palato-alveolar fricatives",
        "phonemes": ["ch", "jh", "sh", "zh"],
    },
    6: {
        "id": 6,
        "viseme": "G",
        "description": "Rounded vowel",
        "phonemes": ["oy", "ao"],
    },
    7: {
        "id": 7,
        "viseme": "H",
        "description": "Alveolar fricatives",
        "phonemes": ["s", "z"],
    },
    8: {
        "id": 8,
        "viseme": "I",
        "description": "Open vowels",
        "phonemes": ["aa", "ae", "ah", "ay", "eh", "ey", "ih", "iy", "y", "ao", "ax-h", "ax", "ix"],
    },
    9: {
        "id": 9,
        "viseme": "J",
        "description": "Alveolar stops / nasals / laterals",
        "phonemes": ["d", "l", "n", "t", "el", "nx", "en", "dx"],
    },
    10: {
        "id": 10,
        "viseme": "K",
        "description": "Velar",
        "phonemes": ["g", "k", "ng", "eng"],
    },
    11: {
        "id": 11,
        "viseme": "S",
        "description": "Silence",
        "phonemes": ["sil"],
    },
}


# Maps canonical viseme names to folder names in images_with_reference.
# B and G are intentionally separate.
REFERENCE_FOLDERS: Dict[str, List[str]] = {
    "A": ["labiodental"],
    "B": ["roundedvowels"],
    "C": ["bilabialClosure"],
    "D": ["diphthong"],
    "E": ["Dental"],
    "F": ["palatoAlveolar"],
    "G": ["RoundedVowel-oy"],
    "H": ["AlveolarFrictave"],
    "I": ["OpenVowels"],
    "J": ["AlveolarStop"],
    "K": ["Velar"],
    "S": ["silence"],
}


def _collect_images(folder_names: List[str]) -> List[Path]:
    paths: List[Path] = []
    valid_suffixes = {".jpg", ".jpeg", ".png", ".webp"}

    for folder_name in folder_names:
        folder = REFERENCE_ROOT / folder_name
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in valid_suffixes:
                paths.append(path)

    return paths


# Resolved absolute image paths grouped by canonical viseme.
REFERENCE_IMAGES: Dict[str, List[Path]] = {
    viseme: _collect_images(folder_names)
    for viseme, folder_names in REFERENCE_FOLDERS.items()
}


def get_label_by_id(label_id: int) -> Optional[Dict[str, object]]:
    return TARGET_LABELS.get(label_id)


def get_label_by_viseme(viseme: str) -> Optional[Dict[str, object]]:
    for label in TARGET_LABELS.values():
        if label["viseme"] == viseme:
            return label
    return None


def iter_reference_images(viseme: Optional[str] = None) -> List[Path]:
    if viseme is None:
        all_images: List[Path] = []
        for paths in REFERENCE_IMAGES.values():
            all_images.extend(paths)
        return all_images
    return REFERENCE_IMAGES.get(viseme, [])
