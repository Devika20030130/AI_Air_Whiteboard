"""
ocr/hybrid_ocr.py — Waterfall OCR pipeline orchestrator.

Strategy
--------
Each engine is tried in priority order (TrOCR → EasyOCR → Tesseract).
A result is accepted and the chain stops as soon as confidence
exceeds `CFG.hybrid.accept_threshold`.
If all engines fall below the threshold, the highest-confidence
result is returned along with a quality warning.

Features
--------
* Confidence-gated waterfall with early exit
* Per-stage latency logging  (CFG.hybrid.benchmark_each)
* Optional debug visualisation window  (CFG.hybrid.debug_visualize)
* Thread-safe for single-process use (one pipeline instance is enough)
* Graceful degradation — works with only Tesseract installed
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import CFG
from ocr.equation_parser import EquationParser, ParseResult
from ocr.easyocr_engine import EasyOCREngine
from ocr.preprocess import EquationPreprocessor
from ocr.tesseract_engine import TesseractEngine
from ocr.trocr_engine import TrOCREngine
from utils.image_utils import make_debug_grid
from utils.logger import get_logger

log = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════
#  RESULT DATACLASS
# ══════════════════════════════════════════════════════════════════
@dataclass
class OCRResult:
    """Complete result from the hybrid pipeline."""
    text:        str            # Raw OCR text (best engine)
    confidence:  float          # Confidence score  [0, 1]
    engine_used: str            # Which engine produced the result
    parse:       Optional[ParseResult]  # SymPy parse result
    latency_ms:  float          # Total pipeline time
    all_results: List[Tuple[str, str, float]]  # [(engine, text, conf)]

    @property
    def solution(self) -> str:
        if self.parse and self.parse.success:
            return self.parse.result
        return ""

    @property
    def display_text(self) -> str:
        """Short string suitable for HUD notification."""
        if not self.text:
            return "⚠ No text detected"
        sol = self.solution
        short = (self.text[:24] + "…") if len(self.text) > 25 else self.text
        if sol:
            return f"'{short}'  →  {sol}"
        return f"'{short}'  (unresolved)"


# ══════════════════════════════════════════════════════════════════
#  HYBRID PIPELINE
# ══════════════════════════════════════════════════════════════════
class HybridOCRPipeline:
    """
    Orchestrates TrOCR → EasyOCR → Tesseract with confidence-gated
    waterfall fallback and optional debug visualisation.
    """

    def __init__(self) -> None:
        self._cfg         = CFG.hybrid
        self.preprocessor = EquationPreprocessor()
        self.tesseract    = TesseractEngine()
        self.easyocr      = EasyOCREngine()
        self.trocr        = TrOCREngine()
        self.parser       = EquationParser()

        # Map config engine names → callable(canvas_bgr) → (text, conf)
        self._engine_map = {
            "trocr":     self._run_trocr,
            "easyocr":   self._run_easyocr,
            "tesseract": self._run_tesseract,
        }

    # ──────────────────────────────────────────────────────────
    #  MAIN ENTRY
    # ──────────────────────────────────────────────────────────
    def run(self, canvas_bgr: np.ndarray) -> OCRResult:
        """
        Run the full hybrid pipeline on *canvas_bgr*.

        Returns an OCRResult regardless of success — callers should
        check `.text` and `.confidence` to decide what to display.
        """
        t_start = time.perf_counter()
        all_results: List[Tuple[str, str, float]] = []
        best_text, best_conf, best_engine = "", 0.0, "none"

        # ── Waterfall ─────────────────────────────────────
        for engine_name in self._cfg.engine_priority:
            fn = self._engine_map.get(engine_name)
            if fn is None:
                log.warning("Unknown engine '%s' in priority list.", engine_name)
                continue

            t0 = time.perf_counter()
            text, conf = fn(canvas_bgr)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            all_results.append((engine_name, text, conf))

            if self._cfg.benchmark_each:
                log.info(
                    "OCR  %-10s  conf=%.3f  %7.1f ms  %r",
                    engine_name, conf, elapsed_ms,
                    (text[:35] + "…") if len(text) > 36 else text,
                )

            if conf > best_conf:
                best_text, best_conf, best_engine = text, conf, engine_name

            # Early exit if this result is good enough
            if conf >= self._cfg.accept_threshold and text:
                log.info(
                    "OCR accepted from %s (conf %.3f ≥ %.3f)",
                    engine_name, conf, self._cfg.accept_threshold,
                )
                break
        else:
            # Exhausted all engines without hitting threshold
            if best_text:
                log.warning(
                    "All OCR engines below threshold %.3f — "
                    "using best: %s (%.3f)",
                    self._cfg.accept_threshold, best_engine, best_conf,
                )
            else:
                log.warning("OCR: no text detected by any engine.")

        # ── Parse & Solve ─────────────────────────────────
        parse_result: Optional[ParseResult] = None
        if best_text:
            parse_result = self.parser.parse_and_solve(best_text)
            if parse_result.success:
                log.info("Solved: %s", parse_result.result)
            else:
                log.debug("Parser: %s", parse_result.result)

        # ── Debug visualisation ───────────────────────────
        if self._cfg.debug_visualize and self.preprocessor.debug_stages:
            self._show_debug(canvas_bgr, all_results)

        latency_ms = (time.perf_counter() - t_start) * 1000
        log.debug("Hybrid pipeline total: %.1f ms", latency_ms)

        return OCRResult(
            text        = best_text,
            confidence  = best_conf,
            engine_used = best_engine,
            parse       = parse_result,
            latency_ms  = latency_ms,
            all_results = all_results,
        )

    # ──────────────────────────────────────────────────────────
    #  PER-ENGINE WRAPPERS
    #  Each wrapper returns (text, confidence) and is responsible
    #  for passing the shared preprocessor to its engine.
    # ──────────────────────────────────────────────────────────
    def _run_trocr(self, canvas_bgr: np.ndarray) -> Tuple[str, float]:
        return self.trocr.recognize(canvas_bgr, preprocessor=self.preprocessor)

    def _run_easyocr(self, canvas_bgr: np.ndarray) -> Tuple[str, float]:
        return self.easyocr.recognize(canvas_bgr, preprocessor=self.preprocessor)

    def _run_tesseract(self, canvas_bgr: np.ndarray) -> Tuple[str, float]:
        # Run the full preprocessing pipeline and collect debug stages
        collect = self._cfg.debug_visualize
        binary = self.preprocessor.process(canvas_bgr, collect_debug=collect)
        if binary is None:
            return "", 0.0
        return self.tesseract.recognize(canvas_bgr, preprocessor=self.preprocessor)

    # ──────────────────────────────────────────────────────────
    #  DEBUG WINDOW
    # ──────────────────────────────────────────────────────────
    def _show_debug(
        self,
        original: np.ndarray,
        results: List[Tuple[str, str, float]],
    ) -> None:
        """
        Display a composite debug window:
        Left  — preprocessing stage grid
        Right — per-engine OCR results panel
        """
        # Preprocessing stage grid
        stage_grid = make_debug_grid(
            self.preprocessor.debug_stages, target_w=220
        )

        # Results panel
        panel_h = max(stage_grid.shape[0], 150)
        panel_w = 420
        panel   = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        cv2.putText(panel, "OCR Results", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 220, 255), 1, cv2.LINE_AA)
        for i, (eng, txt, conf) in enumerate(results):
            y = 50 + i * 55
            col = (80, 220, 80) if conf >= CFG.hybrid.accept_threshold else (80, 80, 200)
            cv2.putText(panel, f"[{eng}]  conf={conf:.3f}", (8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
            short = (txt[:48] + "…") if len(txt) > 49 else txt
            cv2.putText(panel, short or "<empty>", (8, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1, cv2.LINE_AA)

        # Equalise heights
        gh, gw = stage_grid.shape[:2]
        ph, pw = panel.shape[:2]
        max_h  = max(gh, ph)
        stage_grid = np.pad(stage_grid, ((0, max_h - gh), (0, 0), (0, 0)))
        panel      = np.pad(panel,      ((0, max_h - ph), (0, 0), (0, 0)))

        composite = np.hstack([stage_grid, panel])
        cv2.imshow("Hybrid OCR — Debug", composite)
        cv2.waitKey(1)
