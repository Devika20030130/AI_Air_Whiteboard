"""
ocr/trocr_engine.py — Microsoft TrOCR handwritten-text recognition engine.

TrOCR (Transformer OCR) is a vision-encoder + language-decoder model
fine-tuned on handwritten documents.  It substantially outperforms
Tesseract on cursive / mixed-case / mathematical handwriting.

Model used : microsoft/trocr-base-handwritten  (~340 MB first download)
Fallback   : Returns ("", 0.0) on any error — HybridOCR handles retry.

GPU support
-----------
If CUDA is available the model is automatically moved to the GPU.
Set CFG.device = "cpu" in config.py to force CPU mode.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

import numpy as np
import torch

from config import CFG
from utils.logger import get_logger, timer

log = get_logger(__name__)


class TrOCREngine:
    """
    Lazy-loading TrOCR inference engine.

    The model is NOT downloaded or loaded until the first call to
    `recognize()` (or an explicit call to `load()`).  This keeps
    startup time fast and avoids unnecessary VRAM usage if TrOCR
    is never triggered.
    """

    def __init__(self) -> None:
        self._cfg        = CFG.trocr
        self._device     = torch.device(self._cfg.device)
        self._processor  = None   # TrOCRProcessor
        self._model      = None   # VisionEncoderDecoderModel
        self._loaded     = False
        self._load_error: Optional[str] = None

    # ──────────────────────────────────────────────────────────
    #  MODEL LOADING
    # ──────────────────────────────────────────────────────────
    def load(self) -> bool:
        """
        Download (first run) and load the TrOCR model.
        Returns True on success, False on failure.
        """
        if self._loaded:
            return True
        if self._load_error:
            return False     # Don't retry a known-broken state

        try:
            log.info(
                "Loading TrOCR model '%s' on %s …  "
                "(first run downloads ~340 MB)",
                self._cfg.model_name, self._cfg.device,
            )
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel

            with timer("TrOCR model load", log):
                self._processor = TrOCRProcessor.from_pretrained(
                    self._cfg.model_name
                )
                self._model = VisionEncoderDecoderModel.from_pretrained(
                    self._cfg.model_name
                ).to(self._device)

            self._model.eval()   # inference mode — disables dropout
            self._loaded = True
            log.info("TrOCR model ready on %s.", self._cfg.device)
            return True

        except ImportError:
            msg = (
                "transformers package not found.  "
                "Run: pip install transformers sentencepiece"
            )
            log.error(msg)
            self._load_error = msg
            return False

        except Exception as exc:
            msg = f"TrOCR load failed: {exc}"
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
        Run TrOCR on *canvas_bgr*.

        Parameters
        ----------
        canvas_bgr    : Raw BGR canvas from CanvasManager.
        preprocessor  : Optional EquationPreprocessor instance.
                        If provided, a transformer-friendly PIL image
                        is prepared from it; otherwise a direct crop is used.

        Returns
        -------
        (text, confidence) where confidence ∈ [0, 1].
        Returns ("", 0.0) if the model is unavailable or the canvas
        is blank.
        """
        # Lazy-load on first use
        if not self._loaded:
            if not self.load():
                return "", 0.0

        # ── Build PIL input image ──────────────────────────
        if preprocessor is not None:
            pil_img = preprocessor.process_for_transformer(canvas_bgr)
        else:
            from utils.image_utils import crop_to_ink, bgr_to_pil
            cropped = crop_to_ink(canvas_bgr)
            pil_img = bgr_to_pil(cropped) if cropped is not None else None

        if pil_img is None:
            log.debug("TrOCR: canvas is blank.")
            return "", 0.0

        # Ensure RGB (TrOCR processor expects RGB PIL)
        pil_img = pil_img.convert("RGB")

        try:
            with timer("TrOCR inference", log):
                pixel_values = self._processor(
                    images=pil_img,
                    return_tensors="pt",
                ).pixel_values.to(self._device)

                with torch.no_grad():
                    # Generate with beam search for better accuracy
                    outputs = self._model.generate(
                        pixel_values,
                        max_new_tokens=self._cfg.max_new_tokens,
                        num_beams=4,
                        early_stopping=True,
                        output_scores=True,
                        return_dict_in_generate=True,
                    )

                generated_ids = outputs.sequences
                text = self._processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0].strip()

                # ── Confidence from sequence scores ───────
                confidence = self._compute_confidence(outputs)

            log.debug("TrOCR  → %r  (conf %.3f)", text, confidence)
            return text, confidence

        except Exception as exc:
            log.error("TrOCR inference error: %s", exc)
            return "", 0.0

    # ──────────────────────────────────────────────────────────
    #  CONFIDENCE ESTIMATION
    # ──────────────────────────────────────────────────────────
    @staticmethod
    def _compute_confidence(outputs) -> float:
        """
        Derive a scalar confidence score from the beam-search
        sequence scores (log-probabilities per token).

        We take the mean token log-prob and map it to [0, 1] with
        a sigmoid-like normalisation.  This is an approximation —
        exact calibration would require a separate temperature-scaling
        step.
        """
        try:
            if not hasattr(outputs, "scores") or not outputs.scores:
                return 0.5    # Unknown — return neutral confidence

            # Each element of scores is shape (batch, vocab_size)
            log_probs = []
            for score_tensor in outputs.scores:
                # Max log-prob across vocab = greedy token probability
                lp = torch.max(torch.log_softmax(score_tensor, dim=-1), dim=-1).values
                log_probs.append(lp.mean().item())

            mean_lp = float(np.mean(log_probs)) if log_probs else -5.0
            # Map [-10, 0] → [0, 1]  (clamp extremes)
            conf = float(np.clip((mean_lp + 10.0) / 10.0, 0.0, 1.0))
            return conf

        except Exception:
            return 0.5

    # ──────────────────────────────────────────────────────────
    #  PROPERTIES
    # ──────────────────────────────────────────────────────────
    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def device(self) -> str:
        return str(self._device)
