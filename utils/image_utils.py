"""
utils/image_utils.py — Shared OpenCV / NumPy helpers.

All functions are pure (no side-effects) so they are trivially
testable and safe to call from any thread.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


# ══════════════════════════════════════════════════════════════════
#  TYPE ALIASES
# ══════════════════════════════════════════════════════════════════
BGR   = np.ndarray   # shape (H, W, 3)  dtype uint8
GRAY  = np.ndarray   # shape (H, W)     dtype uint8
BINARY = np.ndarray  # shape (H, W)     dtype uint8  — values 0 / 255
Rect  = Tuple[int, int, int, int]   # x, y, w, h


# ══════════════════════════════════════════════════════════════════
#  REGION / BOUNDING-BOX HELPERS
# ══════════════════════════════════════════════════════════════════
def ink_bounding_box(
    gray: GRAY,
    threshold: int = 10,
    padding: int = 20,
) -> Optional[Rect]:
    """
    Return (x, y, w, h) of the tightest bounding box around all
    non-zero (ink) pixels, expanded by *padding* on every side.
    Returns None when the image is blank.
    """
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(binary)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    H, W = gray.shape
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(W, x + w + padding)
    y2 = min(H, y + h + padding)
    return x1, y1, x2 - x1, y2 - y1


def crop_to_ink(bgr: BGR, padding: int = 20) -> Optional[BGR]:
    """
    Return a tight crop of the canvas containing only the drawn ink.
    Returns None when the canvas is blank.
    """
    gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bbox  = ink_bounding_box(gray, padding=padding)
    if bbox is None:
        return None
    x, y, w, h = bbox
    return bgr[y : y + h, x : x + w]


# ══════════════════════════════════════════════════════════════════
#  RESIZE / SCALE
# ══════════════════════════════════════════════════════════════════
def scale_image(
    img: np.ndarray,
    factor: float,
    interpolation: int = cv2.INTER_CUBIC,
) -> np.ndarray:
    """Uniform scale by *factor* using the given interpolation mode."""
    if factor == 1.0:
        return img
    return cv2.resize(img, None, fx=factor, fy=factor,
                      interpolation=interpolation)


def fit_to_width(img: np.ndarray, max_w: int) -> np.ndarray:
    """Downscale *img* so it is at most *max_w* pixels wide."""
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    ratio = max_w / w
    return cv2.resize(img, (max_w, int(h * ratio)),
                      interpolation=cv2.INTER_AREA)


# ══════════════════════════════════════════════════════════════════
#  COLOUR CONVERSION
# ══════════════════════════════════════════════════════════════════
def bgr_to_rgb(img: BGR) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def bgr_to_pil(img: BGR):          # → PIL.Image  (import deferred)
    from PIL import Image
    return Image.fromarray(bgr_to_rgb(img))


def pil_to_bgr(pil_img) -> BGR:
    import numpy as np
    rgb = np.array(pil_img)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# ══════════════════════════════════════════════════════════════════
#  BLEND / COMPOSITE
# ══════════════════════════════════════════════════════════════════
def overlay_with_alpha(
    base: BGR,
    overlay: BGR,
    alpha: float,
    mask: Optional[GRAY] = None,
) -> BGR:
    """
    Blend *overlay* onto *base* with *alpha* weight.
    If *mask* is provided (binary, same H×W), only blend where mask > 0.
    Mutates *base* in-place and returns it.
    """
    if mask is not None:
        region = mask > 0
        base[region] = cv2.addWeighted(
            base, 1.0 - alpha, overlay, alpha, 0
        )[region]
    else:
        cv2.addWeighted(overlay, alpha, base, 1.0 - alpha, 0, dst=base)
    return base


# ══════════════════════════════════════════════════════════════════
#  DEBUG VISUALISATION GRID
# ══════════════════════════════════════════════════════════════════
def make_debug_grid(
    images: list[Tuple[str, np.ndarray]],
    target_w: int = 300,
) -> BGR:
    """
    Stack labelled images side-by-side for a debug visualisation
    window.  Each image is resized to *target_w* wide; aspect ratio
    is preserved.  Labels are drawn at the top of each panel.

    Parameters
    ----------
    images : list of (label, image) pairs
    target_w : width of each panel in pixels

    Returns
    -------
    Single BGR image — all panels in one row.
    """
    panels = []
    for label, img in images:
        # Normalise to BGR
        if img.ndim == 2:
            panel = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            panel = img.copy()

        # Resize
        h, w = panel.shape[:2]
        th   = max(1, int(h * target_w / max(w, 1)))
        panel = cv2.resize(panel, (target_w, th), interpolation=cv2.INTER_AREA)

        # Label background
        cv2.rectangle(panel, (0, 0), (target_w, 22), (30, 30, 30), -1)
        cv2.putText(
            panel, label, (4, 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 220, 255), 1, cv2.LINE_AA,
        )
        panels.append(panel)

    if not panels:
        return np.zeros((100, 300, 3), dtype=np.uint8)

    # Equalise heights by padding bottom
    max_h = max(p.shape[0] for p in panels)
    padded = []
    for p in panels:
        dh = max_h - p.shape[0]
        padded.append(np.pad(p, ((0, dh), (0, 0), (0, 0))))

    return np.hstack(padded)
