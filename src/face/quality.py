from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


def blur_laplacian_var(bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness_mean(bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def face_area_ratio(bbox: np.ndarray, img_shape: Tuple[int, int, int]) -> float:
    h, w = img_shape[:2]
    x1, y1, x2, y2 = bbox.astype(float)
    box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return float(box_area / (w * h + 1e-9))


def bbox_width(bbox: np.ndarray) -> float:
    x1, _, x2, _ = bbox.astype(float)
    return float(max(0.0, x2 - x1))


def quality_checks(bgr: np.ndarray, bbox: np.ndarray) -> Dict[str, Any]:
    q: Dict[str, Any] = {}
    q["blur_laplacian_var"] = blur_laplacian_var(bgr)
    q["brightness_mean"] = brightness_mean(bgr)
    q["face_area_ratio"] = face_area_ratio(bbox, bgr.shape)
    q["bbox_width_px"] = bbox_width(bbox)
    return q


def passes_quality(
    q: Dict[str, Any],
    *,
    blur_min_laplacian_var: float,
    brightness_min: float,
    brightness_max: float,
    min_face_area_ratio: float,
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if q["blur_laplacian_var"] < blur_min_laplacian_var:
        reasons.append("too_blurry")
    if q["brightness_mean"] < brightness_min:
        reasons.append("too_dark")
    if q["brightness_mean"] > brightness_max:
        reasons.append("too_bright")
    if q["face_area_ratio"] < min_face_area_ratio:
        reasons.append("face_too_small_ratio")
    return (len(reasons) == 0), reasons
