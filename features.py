"""
Lip ratio vector extraction from MediaPipe face landmarks.
All distances are normalized by face reference (face width 234–454) for scale invariance.
"""

from __future__ import annotations

import numpy as np
from typing import List, Union

# MediaPipe face mesh indices used for lip ratios
LEFT_CORNER = 61
RIGHT_CORNER = 291
UPPER_INNER = 13
LOWER_INNER = 14
# Outer lip vertical: upper center (nose tip area / philtrum) and lower lip center
UPPER_OUTER_MID = 0   # often used as upper lip reference
LOWER_OUTER_MID = 17
# Inner lip width
INNER_LEFT = 78
INNER_RIGHT = 308
# Face reference (cheeks)
FACE_LEFT = 234
FACE_RIGHT = 454
# Upper lip contour for curl (subset)
UPPER_LIP_CONTOUR = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]

EPS = 1e-6


def _pt(landmarks: List, idx: int, img_w: int, img_h: int) -> np.ndarray:
    l = landmarks[idx]
    return np.array([l.x * img_w, l.y * img_h], dtype=float)


def get_lip_ratio_vector(
    landmarks: List,
    img_w: int,
    img_h: int,
) -> np.ndarray:
    """
    Compute a fixed-length ratio vector from face landmarks.
    All distances are normalized by face width (234–454) for scale invariance.

    Returns vector of shape (4,) in order:
      [0] outer mouth width / outer mouth height
      [1] inner lip opening height / inner lip width
      [2] upper lip curl ratio (vertical extent of upper lip / mouth width)
      [3] corner asymmetry |left_height - right_height| / mouth_width
    """
    def pt(idx: int) -> np.ndarray:
        return _pt(landmarks, idx, img_w, img_h)

    face_left = pt(FACE_LEFT)
    face_right = pt(FACE_RIGHT)
    face_ref = np.linalg.norm(face_right - face_left) + EPS

    # Mouth width (corners) normalized
    left = pt(LEFT_CORNER)
    right = pt(RIGHT_CORNER)
    mouth_width_px = np.linalg.norm(right - left)
    mouth_width_norm = mouth_width_px / face_ref

    # Outer mouth height (upper outer mid to lower outer mid) normalized
    upper_outer = pt(UPPER_OUTER_MID)
    lower_outer = pt(LOWER_OUTER_MID)
    outer_height_px = np.linalg.norm(lower_outer - upper_outer)
    outer_height_norm = outer_height_px / face_ref
    outer_w_h_ratio = mouth_width_norm / (outer_height_norm + EPS)

    # Inner lip opening: height and width normalized
    upper_inner = pt(UPPER_INNER)
    lower_inner = pt(LOWER_INNER)
    inner_height_px = np.linalg.norm(lower_inner - upper_inner)
    inner_height_norm = inner_height_px / face_ref
    inner_left_pt = pt(INNER_LEFT)
    inner_right_pt = pt(INNER_RIGHT)
    inner_width_px = np.linalg.norm(inner_right_pt - inner_left_pt)
    inner_width_norm = inner_width_px / face_ref
    inner_open_ratio = inner_height_norm / (inner_width_norm + EPS)

    # Upper lip curl: vertical extent of upper lip contour / mouth width
    upper_ys = [pt(i)[1] for i in UPPER_LIP_CONTOUR]
    upper_vertical_extent_px = max(upper_ys) - min(upper_ys)
    upper_curl_ratio = (upper_vertical_extent_px / face_ref) / (mouth_width_norm + EPS)

    # Corner asymmetry: |left_side_height - right_side_height| / mouth_width
    left_to_upper = np.linalg.norm(upper_inner - left)
    right_to_upper = np.linalg.norm(upper_inner - right)
    corner_asym = abs(left_to_upper - right_to_upper) / face_ref / (mouth_width_norm + EPS)

    return np.array(
        [
            float(outer_w_h_ratio),
            float(inner_open_ratio),
            float(upper_curl_ratio),
            float(corner_asym),
        ],
        dtype=np.float64,
    )
