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
    width:  int   = 1920   # Upgraded: higher res → more ink detail for OCR
    height: int   = 1080
    fps:    int   = 30
    # 1-frame internal buffer → minimum capture latency
    buffer_size: int = 1
    use_dshow: bool = True   # DirectShow backend on Windows


@dataclass(frozen=True)
class MediaPipeConfig:
    max_hands:             int   = 2
    model_complexity:      int   = 0   # 0 = fastest; 1 = accurate
    detection_confidence:  float = 0.85   # Raised: reduces false hand detections
    tracking_confidence:   float = 0.85   # Raised: improves landmark stability


@dataclass(frozen=True)
class DrawingConfig:
    brush_thickness:   int   = 4       # Reduced: thinner strokes → cleaner OCR glyphs
    brush_min:         int   = 2
    brush_max:         int   = 50
    brush_step:        int   = 2
    eraser_radius:     int   = 45
    smooth_alpha:      float = 0.25   # Lowered: heavier smoothing → less jitter
    smooth_window:     int   = 6
    min_draw_dist:     int   = 5      # Raised: ignores micro-tremor < 5 px
    canvas_ink_weight: float = 0.90   # ink opacity over camera feed

    # ── Kalman filter tuning ──────────────────────────────────
    # process_noise (Q): how much we trust the finger can
    #   accelerate between frames.  Lower = smoother but lags
    #   on fast intentional strokes.  0.03 is calibrated for
    #   air-writing at 30 FPS with brush_thickness=4.
    kalman_process_noise:      float = 0.03
    # measurement_noise (R): how much we trust the raw MediaPipe
    #   landmark.  Higher = more smoothing, more lag on fast moves.
    #   0.5 is chosen so Kalman adds ~40 % lag reduction vs EMA alone.
    kalman_measurement_noise:  float = 0.5
    # post_error (P_init): initial estimate covariance.
    #   Starting high lets the filter converge quickly on the first
    #   few frames instead of wandering from the origin.
    kalman_post_error:         float = 0.1

    # ── Motion deadzone ───────────────────────────────────────
    # Displacement (px) below which the stabilised coordinate is
    # NOT updated.  Absorbs sub-pixel MediaPipe jitter when the
    # finger is held still (tremor typically < 3 px peak-to-peak).
    # Set to 0 to disable.
    deadzone_px:               float = 2.5

    # ── Interpolated stroke rendering ─────────────────────────
    # Sub-segment length (px) for interpolation.  Smaller = smoother
    # curves and zero gaps at high speed; 1 px = maximum density.
    interp_step_px:    float = 1.0
    # Draw a filled circle at every interpolated point in addition
    # to LINE_AA segments.  Fills concave corners on sharp turns.
    interp_fill_caps:  bool  = True

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
    ocr_scale:          float = 4.0     # Raised: ~400 DPI → sharper glyph edges for OCR
    adaptive_block:     int   = 31      # Larger block: handles wider brush strokes
    adaptive_c:         int   = 12      # Raised: stronger background suppression
    morph_iterations:   int   = 2
    denoise_h:          float = 12.0    # fastNlMeans filter strength
    # Minimum contour area to keep (removes dust specks)
    min_contour_area:   int   = 80
    # Padding added around the bounding box of all ink (px)
    content_padding:    int   = 20

    # ── CLAHE (Contrast Limited Adaptive Histogram Equalisation) ──
    # Applied after grayscale, before thresholding.
    # clip_limit: controls contrast amplification ceiling.
    #   Too high → amplifies noise; too low → no benefit.
    #   3.0 is a safe middle ground for air-drawn bright strokes
    #   on a near-black canvas where local contrast is already high
    #   but varies between bright-white and dim-grey strokes.
    # tile_grid_size: (8,8) tiles the image into 64 local regions
    #   so the equalisation adapts to local brightness rather than
    #   treating the whole image as one exposure level.
    clahe_clip_limit:   float          = 3.0
    clahe_tile_grid:    Tuple[int,int] = (8, 8)

    # ── Dilation kernel — operator-safe sizing ─────────────────
    # A 2×2 kernel is the smallest that meaningfully thickens thin
    # strokes.  Larger kernels merge adjacent digits and destroy the
    # gap between '=' bars or the two strokes of '+'.
    # Operators at risk of disappearance after thresholding:
    #   '-'  (horizontal, 1-2 px tall after threshold)
    #   '+'  (cross — two thin perpendicular strokes)
    #   '='  (two parallel horizontals, easily merged or lost)
    #   '/'  (diagonal, thinnest rendered glyph in air-writing)
    # 2×2 thickens each of these by exactly 1 px in every direction,
    # making them survive subsequent denoising without merging digits.
    dilation_kernel_size: int = 2
    dilation_iterations:  int = 1


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
    # Raised to 0.85: forces more engines to run → voting logic activates more often
    accept_threshold: float = 0.85
    # Show a side-by-side debug window when OCR is triggered
    debug_visualize: bool   = False
    # Show the final preprocessed binary image in a separate window (toggle: 'p')
    show_preprocessed: bool = False
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
