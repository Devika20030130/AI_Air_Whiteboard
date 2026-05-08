"""
ocr/preprocess.py — Multi-stage OpenCV image preprocessing pipeline
optimised for handwritten mathematical equations on a dark canvas.

Pipeline stages
---------------
1. Crop to ink bounding box       → removes dead space
2. Upscale (cubic)                → ~300 DPI equivalent for Tesseract/TrOCR
3. Adaptive Gaussian threshold    → handles variable brush opacity
4. Morphological close            → bridges micro-gaps in strokes
5. Contour noise removal          → drops isolated dust specks
6. fastNlMeans denoise            → final speckle suppression
7. Optional border pad            → models prefer context around glyphs

Each stage can be independently toggled via Config for debugging.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import CFG
from utils.image_utils import (
    BGR,
    BINARY,
    GRAY,
    bgr_to_pil,
    crop_to_ink,
    ink_bounding_box,
    make_debug_grid,
    scale_image,
)
from utils.logger import get_logger

log = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════
#  PREPROCESSOR
# ══════════════════════════════════════════════════════════════════
class EquationPreprocessor:
    """
    Converts a raw BGR whiteboard canvas into a clean binary image
    ready for OCR.

    The preprocessor is stateless — every call to `process()` is
    independent.  Intermediate stage images are available via
    `debug_stages` after each call (useful for visualisation).
    """

    def __init__(self) -> None:
        self._cfg = CFG.preprocess
        # Pre-allocate morphological kernel (avoid per-call allocation)
        self._morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (3, 3)
        )
        self.debug_stages: List[Tuple[str, np.ndarray]] = []

    # ──────────────────────────────────────────────────────────
    #  PUBLIC API
    # ──────────────────────────────────────────────────────────
    def process(
        self,
        canvas_bgr: BGR,
        collect_debug: bool = False,
    ) -> Optional[BINARY]:
        """
        Run the full preprocessing pipeline.

        Parameters
        ----------
        canvas_bgr    : Raw BGR canvas from CanvasManager.
        collect_debug : If True, save each stage into `self.debug_stages`.

        Returns
        -------
        Binary uint8 image (ink = 255, background = 0) ready for OCR,
        or None if the canvas is blank.
        """
        self.debug_stages = []
        c = canvas_bgr.copy()

        def _save(label: str, img: np.ndarray) -> None:
            if collect_debug:
                self.debug_stages.append((label, img.copy()))

        _save("0_raw", c)

        # ── Stage 1: Crop to ink ───────────────────────────
        cropped = crop_to_ink(c, padding=self._cfg.content_padding)
        if cropped is None:
            log.debug("Preprocess: canvas is blank — skipping OCR.")
            return None
        _save("1_crop", cropped)

        # ── Stage 2: Grayscale ─────────────────────────────
        gray: GRAY = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        _save("2_gray", gray)

        # ── Stage 3: Upscale ───────────────────────────────
        scaled: GRAY = scale_image(
            gray,
            self._cfg.ocr_scale,
            interpolation=cv2.INTER_CUBIC,
        )
        _save("3_upscale", scaled)

        # ── Stage 4: Adaptive threshold ────────────────────
        thresh: BINARY = cv2.adaptiveThreshold(
            scaled, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=self._cfg.adaptive_block,
            C=self._cfg.adaptive_c,
        )
        _save("4_threshold", thresh)

        # ── Stage 5: Morphological closing ────────────────
        closed: BINARY = cv2.morphologyEx(
            thresh, cv2.MORPH_CLOSE,
            self._morph_kernel,
            iterations=self._cfg.morph_iterations,
        )
        _save("5_morphclose", closed)

        # ── Stage 6: Contour noise removal ────────────────
        cleaned = self._remove_noise_contours(closed)
        _save("6_contour_clean", cleaned)

        # ── Stage 7: fastNlMeans denoising ────────────────
        denoised: BINARY = cv2.fastNlMeansDenoising(
            cleaned, h=self._cfg.denoise_h
        )
        _save("7_denoise", denoised)

        # ── Stage 8: Border padding (helps model context) ─
        pad = self._cfg.content_padding
        final: BINARY = cv2.copyMakeBorder(
            denoised, pad, pad, pad, pad,
            cv2.BORDER_CONSTANT, value=0,
        )
        _save("8_final", final)

        log.debug(
            "Preprocess complete: %dx%d → %dx%d",
            canvas_bgr.shape[1], canvas_bgr.shape[0],
            final.shape[1], final.shape[0],
        )
        return final

    # ──────────────────────────────────────────────────────────
    #  PROCESS FOR TRANSFORMER  (RGB PIL image)
    # ──────────────────────────────────────────────────────────
    def process_for_transformer(self, canvas_bgr: BGR):
        """
        Return a PIL RGB image optimised for vision transformers
        (TrOCR).  Uses a lighter pipeline:
        - crop to ink
        - mild upscale
        - white background with black ink
        """
        cropped = crop_to_ink(canvas_bgr, padding=self._cfg.content_padding)
        if cropped is None:
            return None

        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        # Upscale less aggressively — transformer has fixed-size patches
        scaled = scale_image(gray, min(self._cfg.ocr_scale, 2.0),
                             interpolation=cv2.INTER_CUBIC)

        # Convert to white-background, black-ink (standard handwriting)
        _, binary = cv2.threshold(scaled, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        inverted  = cv2.bitwise_not(binary)   # white bg, black ink

        # Convert to 3-channel RGB PIL
        rgb = cv2.cvtColor(inverted, cv2.COLOR_GRAY2RGB)
        from PIL import Image
        return Image.fromarray(rgb)

    # ──────────────────────────────────────────────────────────
    #  HELPERS
    # ──────────────────────────────────────────────────────────
    def _remove_noise_contours(self, binary: BINARY) -> BINARY:
        """
        Erase contours whose area is below `min_contour_area`.
        This removes isolated pixel clusters (dust, sweat drops on
        the camera lens) without touching real strokes.
        """
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        result = binary.copy()
        removed = 0
        for cnt in contours:
            if cv2.contourArea(cnt) < self._cfg.min_contour_area:
                cv2.drawContours(result, [cnt], -1, 0, -1)
                removed += 1
        if removed:
            log.debug("Contour noise removal: dropped %d micro-contours", removed)
        return result

    # ──────────────────────────────────────────────────────────
    #  DEBUG GRID
    # ──────────────────────────────────────────────────────────
    def show_debug_window(self, win_name: str = "Preprocess Debug") -> None:
        """
        Display a side-by-side grid of all pipeline stages.
        Must be called after `process(collect_debug=True)`.
        Call `cv2.waitKey(0)` in the caller to block.
        """
        if not self.debug_stages:
            log.warning("No debug stages available — call process(collect_debug=True) first.")
            return
        grid = make_debug_grid(self.debug_stages, target_w=260)
        cv2.imshow(win_name, grid)
