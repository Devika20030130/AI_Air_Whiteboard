"""
config.py — Single source of truth for all project settings.

Edit values here; nothing else in the codebase should contain
magic numbers or hard-coded paths.
"""

from __future__ import annotations

import os
import torch
from dataclasses import dataclass, field
from typing import List, Tuple


# ══════════════════════════════════════════════════════════════════
#  DEVICE AUTO-DETECTION
#  Picked up once at import time; used by OCR engines + benchmark.
# ══════════════════════════════════════════════════════════════════
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class CameraConfig:
    index:  int   = 0
    width:  int   = 1280
    height: int   = 720
    fps:    int   = 30
    # 1-frame internal buffer → minimum capture latency
    buffer_size: int = 1
    use_dshow: bool = True   # DirectShow backend on Windows


@dataclass(frozen=True)
class MediaPipeConfig:
    max_hands:             int   = 2
    model_complexity:      int   = 0   # 0 = fastest; 1 = accurate
    detection_confidence:  float = 0.75
    tracking_confidence:   float = 0.75


@dataclass(frozen=True)
class DrawingConfig:
    brush_thickness:   int   = 8
    brush_min:         int   = 2
    brush_max:         int   = 50
    brush_step:        int   = 2
    eraser_radius:     int   = 45
    smooth_alpha:      float = 0.45   # EMA weight
    smooth_window:     int   = 6
    min_draw_dist:     int   = 3      # px — ignore micro-movements
    canvas_ink_weight: float = 0.90   # ink opacity over camera feed

    # BGR colour palette  (White Red Green Blue Yellow Cyan Magenta Orange)
    palette: Tuple = (
        (255, 255, 255),
        (0,   0,   255),
        (0,   255, 0  ),
        (255, 0,   0  ),
        (0,   255, 255),
        (255, 255, 0  ),
        (255, 0,   255),
        (0,   165, 255),
    )


@dataclass(frozen=True)
class GestureConfig:
    confirm_frames:   int = 3    # consecutive frames to confirm gesture
    cooldown_frames:  int = 12   # frames of cooldown after transition
    fist_clear_frames: int = 22  # frames fist must be held to clear


@dataclass(frozen=True)
class CanvasConfig:
    max_undo_steps:        int = 40
    undo_snapshot_threshold: int = 500   # dirty-px before auto-snapshot
    save_dir:              str = "drawings"
    autosave_file:         str = "autosave.png"


@dataclass(frozen=True)
class UIConfig:
    notif_duration: int   = 90     # frames a notification toast is shown
    ui_alpha:       float = 0.55   # translucency of HUD panels
    palette_box:    int   = 26     # px side of each palette swatch
    palette_gap:    int   = 5
    fps_window:     int   = 30     # rolling average for FPS display


# ══════════════════════════════════════════════════════════════════
#  OCR PIPELINE CONFIG
# ══════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class PreprocessConfig:
    """OpenCV preprocessing parameters."""
    ocr_scale:          float = 2.5     # upscale before OCR (≈300 DPI)
    adaptive_block:     int   = 21      # must be odd
    adaptive_c:         int   = 8
    morph_iterations:   int   = 2
    denoise_h:          float = 12.0    # fastNlMeans filter strength
    # Minimum contour area to keep (removes dust specks)
    min_contour_area:   int   = 80
    # Padding added around the bounding box of all ink (px)
    content_padding:    int   = 20


@dataclass(frozen=True)
class TesseractConfig:
    binary_path: str = os.getenv("TESSERACT_PATH", "tesseract")
    psm:         int = 6     # Uniform block of text
    oem:         int = 3     # LSTM engine
    char_whitelist: str = (
        "0123456789"
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "+-*/^()=. "
    )

    @property
    def config_string(self) -> str:
        return (
            f"--psm {self.psm} --oem {self.oem} "
            f"-c tessedit_char_whitelist={self.char_whitelist}"
        )


@dataclass(frozen=True)
class EasyOCRConfig:
    languages:      List[str] = field(default_factory=lambda: ["en"])
    gpu:            bool      = field(default_factory=lambda: DEVICE == "cuda")
    # Confidence gate — results below this are discarded
    min_confidence: float     = 0.30
    # Allowlist passed to EasyOCR reader
    allowlist: str = (
        "0123456789"
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "+-*/^()=. "
    )


@dataclass(frozen=True)
class TrOCRConfig:
    # microsoft/trocr-base-handwritten is ~340 MB; large is ~1.4 GB
    model_name:     str   = "microsoft/trocr-base-handwritten"
    device:         str   = field(default_factory=lambda: DEVICE)
    min_confidence: float = 0.30
    max_new_tokens: int   = 128
    # Lazy-load: model is only fetched the first time OCR is triggered
    lazy_load:      bool  = True


@dataclass(frozen=True)
class HybridOCRConfig:
    """
    Controls the waterfall fallback chain:
      TrOCR  →  EasyOCR  →  Tesseract

    A stage is tried if the previous stage's confidence is below
    the stage's own min_confidence.
    """
    # Order: highest accuracy first
    engine_priority: Tuple[str, ...] = ("trocr", "easyocr", "tesseract")
    # Minimum combined confidence to accept a result without fallback
    accept_threshold: float = 0.55
    # Show a side-by-side debug window when OCR is triggered
    debug_visualize: bool   = False
    # Log timing for each stage
    benchmark_each:  bool   = True


@dataclass(frozen=True)
class EquationParserConfig:
    # Substitutions applied before SymPy parsing
    normalizations: Tuple[Tuple[str, str], ...] = (
        ("×",  "*"),
        ("÷",  "/"),
        ("^",  "**"),
        ("²",  "**2"),
        ("³",  "**3"),
        ("√",  "sqrt"),
        (" ",  ""),      # strip spaces
    )
    # Symbols that make an expression unsolvable — skip silently
    reject_symbols: Tuple[str, ...] = ("∞", "∅", "∇", "∂")


# ══════════════════════════════════════════════════════════════════
#  ASSEMBLED PROJECT CONFIG  (import this everywhere)
# ══════════════════════════════════════════════════════════════════
class AppConfig:
    camera     = CameraConfig()
    mediapipe  = MediaPipeConfig()
    drawing    = DrawingConfig()
    gesture    = GestureConfig()
    canvas     = CanvasConfig()
    ui         = UIConfig()
    preprocess = PreprocessConfig()
    tesseract  = TesseractConfig()
    easyocr    = EasyOCRConfig()
    trocr      = TrOCRConfig()
    hybrid     = HybridOCRConfig()
    parser     = EquationParserConfig()
    device     = DEVICE


CFG = AppConfig()
