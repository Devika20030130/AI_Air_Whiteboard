"""
ocr/tesseract_engine.py — Tesseract OCR baseline engine.

Role in the pipeline
--------------------
Last-resort fallback — fastest, lowest RAM, worst on cursive.
Always available as long as the Tesseract binary is installed.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pytesseract

from config import CFG
from utils.logger import get_logger, timer

log = get_logger(__name__)


class TesseractEngine:
    """Thin, stateless wrapper around pytesseract."""

    def __init__(self) -> None:
        self._cfg = CFG.tesseract
        pytesseract.pytesseract.tesseract_cmd = self._cfg.binary_path

    def recognize(
        self,
        canvas_bgr: np.ndarray,
        preprocessor=None,
    ) -> Tuple[str, float]:
        """
        Run Tesseract on *canvas_bgr* with the 5-stage preprocessor.

        Returns
        -------
        (text, confidence) — confidence estimated from Tesseract's
        per-word confidence scores (mean of all words ≥ 30 %).
        Returns ("", 0.0) on blank canvas or error.
        """
        if preprocessor is None:
            from ocr.preprocess import EquationPreprocessor
            preprocessor = EquationPreprocessor()

        binary = preprocessor.process(canvas_bgr)
        if binary is None:
            return "", 0.0

        try:
            with timer("Tesseract inference", log):
                data = pytesseract.image_to_data(
                    binary,
                    config=self._cfg.config_string,
                    output_type=pytesseract.Output.DICT,
                )

            text, confidence = self._extract(data)
            log.debug("Tesseract  → %r  (conf %.3f)", text, confidence)
            return text, confidence

        except pytesseract.TesseractNotFoundError:
            log.error(
                "Tesseract binary not found at '%s'.  "
                "Set TESSERACT_PATH env var.",
                self._cfg.binary_path,
            )
            return "", 0.0
        except Exception as exc:
            log.error("Tesseract error: %s", exc)
            return "", 0.0

    # ──────────────────────────────────────────────────────────
    @staticmethod
    def _extract(data: dict) -> Tuple[str, float]:
        """Extract text and mean word-level confidence from image_to_data output."""
        words, confs = [], []
        for text, conf in zip(data["text"], data["conf"]):
            try:
                c = int(conf)
            except (ValueError, TypeError):
                continue
            if c >= 30 and text.strip():
                words.append(text.strip())
                confs.append(c / 100.0)

        merged = " ".join(words).strip()
        mean_conf = float(np.mean(confs)) if confs else 0.0
        return merged, mean_conf
