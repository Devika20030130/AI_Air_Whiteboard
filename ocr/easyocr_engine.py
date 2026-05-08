"""
ocr/easyocr_engine.py — EasyOCR handwritten-text recognition engine.

EasyOCR uses CRAFT (text detection) + CRNN (recognition) and
handles cluttered / multi-line handwriting better than Tesseract
out of the box.  It runs on GPU when CUDA is available.

Role in the pipeline
--------------------
Middle-tier fallback: better than Tesseract on messy handwriting,
faster than TrOCR, lower accuracy on pure mathematical notation.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import CFG
from utils.logger import get_logger, timer

log = get_logger(__name__)


class EasyOCREngine:
    """
    Lazy-loading EasyOCR inference engine.

    EasyOCR's Reader object is heavy (~1-2 s to initialise), so we
    defer construction until the first call to `recognize()`.
    """

    def __init__(self) -> None:
        self._cfg    = CFG.easyocr
        self._reader = None
        self._loaded = False
        self._load_error: Optional[str] = None

    # ──────────────────────────────────────────────────────────
    #  MODEL LOADING
    # ──────────────────────────────────────────────────────────
    def load(self) -> bool:
        if self._loaded:
            return True
        if self._load_error:
            return False

        try:
            log.info(
                "Loading EasyOCR  (gpu=%s, langs=%s) …",
                self._cfg.gpu, self._cfg.languages,
            )
            import easyocr

            with timer("EasyOCR Reader init", log):
                self._reader = easyocr.Reader(
                    self._cfg.languages,
                    gpu=self._cfg.gpu,
                    verbose=False,
                )
            self._loaded = True
            log.info("EasyOCR ready.")
            return True

        except ImportError:
            msg = "easyocr not installed.  Run: pip install easyocr"
            log.error(msg)
            self._load_error = msg
            return False
        except Exception as exc:
            msg = f"EasyOCR load failed: {exc}"
            log.error(msg)
            self._load_error = msg
            return False

    # ──────────────────────────────────────────────────────────
    #  INFERENCE
    # ──────────────────────────────────────────────────────────
    def recognize(
        self,
        canvas_bgr: np.ndarray,
        preprocessor=None,
    ) -> Tuple[str, float]:
        """
        Run EasyOCR on *canvas_bgr*.

        Returns
        -------
        (merged_text, mean_confidence)
        Returns ("", 0.0) when nothing is detected or on error.
        """
        if not self._loaded:
            if not self.load():
                return "", 0.0

        # ── Prepare image ──────────────────────────────────
        img = self._prepare(canvas_bgr, preprocessor)
        if img is None:
            return "", 0.0

        try:
            with timer("EasyOCR inference", log):
                raw_results = self._reader.readtext(
                    img,
                    allowlist=self._cfg.allowlist,
                    detail=1,        # returns (bbox, text, conf)
                    paragraph=True,  # merge nearby detections
                )

            return self._merge_results(raw_results)

        except Exception as exc:
            log.error("EasyOCR inference error: %s", exc)
            return "", 0.0

    # ──────────────────────────────────────────────────────────
    #  IMAGE PREPARATION
    # ──────────────────────────────────────────────────────────
    def _prepare(
        self,
        canvas_bgr: np.ndarray,
        preprocessor,
    ) -> Optional[np.ndarray]:
        """
        Return a uint8 numpy array (BGR or GRAY) for EasyOCR.

        EasyOCR accepts numpy arrays directly (no PIL needed).
        We apply a lighter preprocessing than for Tesseract:
        crop → upscale → gentle denoise only.
        """
        from utils.image_utils import crop_to_ink, scale_image
        cfg = CFG.preprocess

        cropped = crop_to_ink(canvas_bgr, padding=cfg.content_padding)
        if cropped is None:
            return None

        gray    = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        scaled  = scale_image(gray, cfg.ocr_scale, cv2.INTER_CUBIC)

        # Light denoise — EasyOCR handles thresholding internally
        denoised = cv2.fastNlMeansDenoising(scaled, h=8)

        # Invert: EasyOCR expects dark-ink on light background
        _, binary  = cv2.threshold(denoised, 0, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        white_bg   = cv2.bitwise_not(binary)

        # Convert to 3-channel so EasyOCR's detector works correctly
        return cv2.cvtColor(white_bg, cv2.COLOR_GRAY2BGR)

    # ──────────────────────────────────────────────────────────
    #  RESULT MERGING
    # ──────────────────────────────────────────────────────────
    def _merge_results(
        self,
        raw: List,
    ) -> Tuple[str, float]:
        """
        EasyOCR returns a list of (bbox, text, confidence).
        We filter by min_confidence, sort top-to-bottom then
        left-to-right (natural reading order), and join tokens.
        """
        min_conf = self._cfg.min_confidence
        accepted = []

        for item in raw:
            if len(item) == 3:
                bbox, text, conf = item
            elif len(item) == 2:
                bbox, text = item
                conf = 0.5   # paragraph mode sometimes omits conf
            else:
                continue

            if conf < min_conf:
                log.debug("EasyOCR: dropped %r (conf %.2f < %.2f)", text, conf, min_conf)
                continue

            # Use top-left corner of bbox for sort key
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 1:
                top_y  = bbox[0][1]
                left_x = bbox[0][0]
            else:
                top_y, left_x = 0, 0

            accepted.append((top_y, left_x, text, conf))

        if not accepted:
            return "", 0.0

        # Sort: primary top-to-bottom, secondary left-to-right
        accepted.sort(key=lambda t: (t[0], t[1]))

        texts = [t[2] for t in accepted]
        confs = [t[3] for t in accepted]

        merged = " ".join(texts).strip()
        mean_conf = float(np.mean(confs))

        log.debug("EasyOCR  → %r  (conf %.3f)", merged, mean_conf)
        return merged, mean_conf

    # ──────────────────────────────────────────────────────────
    #  PROPERTIES
    # ──────────────────────────────────────────────────────────
    @property
    def is_loaded(self) -> bool:
        return self._loaded
