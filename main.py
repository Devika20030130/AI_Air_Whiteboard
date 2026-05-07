#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║           AI Air Whiteboard — v4.0  (Production Grade)          ║
║──────────────────────────────────────────────────────────────────║
║  Real-time hand-gesture virtual whiteboard with:                 ║
║    • MediaPipe Hands tracking  • OpenCV rendering                ║
║    • Tesseract OCR             • SymPy equation solving          ║
║    • Undo / Redo               • Color palette                   ║
║    • Save / Load               • FPS counter + HUD               ║
╚══════════════════════════════════════════════════════════════════╝

Python  : 3.11+
OpenCV  : 4.x
MediaPipe: 0.10.x
Author  : github.com/yourname
License : MIT
"""

# ──────────────────────────────────────────────────────────────────
# STANDARD LIBRARY
# ──────────────────────────────────────────────────────────────────
import os
import sys
import time
import logging
import collections
from enum import Enum, auto
from typing import Deque, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────────
# THIRD-PARTY
# ──────────────────────────────────────────────────────────────────
import cv2
import numpy as np
import mediapipe as mp
import pytesseract
import sympy as sp

# ──────────────────────────────────────────────────────────────────
# LOGGING  (structured, timestamped)
# ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-8s]  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("AIWhiteboard")


# ══════════════════════════════════════════════════════════════════
#  SECTION 1 ─ CONFIGURATION
#  All tuneable constants live here.  Change these, never magic
#  numbers buried in logic.
# ══════════════════════════════════════════════════════════════════
class Config:
    # ── Camera ────────────────────────────────────────────────────
    CAMERA_INDEX: int = 0
    CAMERA_WIDTH: int = 1280
    CAMERA_HEIGHT: int = 720
    CAMERA_FPS: int = 30
    # Use DirectShow on Windows; set False on Linux / macOS
    USE_DSHOW: bool = sys.platform == "win32"

    # ── MediaPipe ─────────────────────────────────────────────────
    MAX_HANDS: int = 2
    MP_DETECTION_CONFIDENCE: float = 0.75
    MP_TRACKING_CONFIDENCE: float = 0.75
    # 0 = fastest (recommended for real-time), 1 = more accurate
    MP_MODEL_COMPLEXITY: int = 0

    # ── Drawing ───────────────────────────────────────────────────
    BRUSH_THICKNESS: int = 8          # Starting brush size (px)
    BRUSH_MIN: int = 2
    BRUSH_MAX: int = 50
    BRUSH_STEP: int = 2
    ERASER_RADIUS: int = 45           # Eraser circle radius (px)
    # Exponential-MA weight: higher = more responsive, less smooth
    SMOOTH_ALPHA: float = 0.45
    SMOOTH_WINDOW: int = 6            # History window for EMA
    MIN_DRAW_DIST: int = 3            # Min px movement before drawing

    # ── Colour Palette (BGR) ──────────────────────────────────────
    PALETTE: List[Tuple[int, int, int]] = [
        (255, 255, 255),   # 1 White
        (0,   0,   255),   # 2 Red
        (0,   255, 0  ),   # 3 Green
        (255, 0,   0  ),   # 4 Blue
        (0,   255, 255),   # 5 Yellow
        (255, 255, 0  ),   # 6 Cyan
        (255, 0,   255),   # 7 Magenta
        (0,   165, 255),   # 8 Orange
    ]

    # ── Gesture Debouncing ────────────────────────────────────────
    # Frames a gesture must be held before it becomes "active"
    GESTURE_CONFIRM_FRAMES: int = 3
    # Frames of cooldown after a gesture transition
    GESTURE_COOLDOWN_FRAMES: int = 12
    # Frames fist must be held before board clears
    FIST_CLEAR_FRAMES: int = 22

    # ── Undo / Redo ───────────────────────────────────────────────
    MAX_UNDO_STEPS: int = 40
    # Minimum px² canvas change to justify a new undo snapshot
    UNDO_SNAPSHOT_THRESHOLD: int = 500

    # ── OCR ───────────────────────────────────────────────────────
    TESSERACT_PATH: str = os.getenv("TESSERACT_PATH", "tesseract")
    # PSM 6 = uniform block of text; OEM 3 = LSTM engine
    TESSERACT_CONFIG: str = (
        "--psm 6 --oem 3 "
        "-c tessedit_char_whitelist="
        "0123456789abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ+-*/^()=. "
    )
    # Scale factor for upscaling canvas before OCR (higher = better)
    OCR_SCALE: float = 2.5

    # ── File I/O ──────────────────────────────────────────────────
    SAVE_DIR: str = "drawings"
    AUTOSAVE_FILE: str = "autosave.png"

    # ── UI / HUD ──────────────────────────────────────────────────
    FONT: int = cv2.FONT_HERSHEY_SIMPLEX
    UI_ALPHA: float = 0.55            # Background panel transparency
    PALETTE_BOX: int = 26             # px side of each palette swatch
    PALETTE_GAP: int = 5
    NOTIF_DURATION: int = 90          # Frames a notification is shown

    # ── Canvas Blend ─────────────────────────────────────────────
    # How strongly the ink appears over the camera feed (0-1)
    CANVAS_INK_WEIGHT: float = 0.90


# ══════════════════════════════════════════════════════════════════
#  SECTION 2 ─ ENUMS
# ══════════════════════════════════════════════════════════════════
class GestureState(Enum):
    IDLE    = auto()
    DRAWING = auto()
    SPACE   = auto()   # Lift pen without drawing
    ERASING = auto()
    FIST    = auto()   # Board clear (held)


# ══════════════════════════════════════════════════════════════════
#  SECTION 3 ─ COORDINATE SMOOTHER
#
#  Uses an Exponential Moving Average (EMA) combined with a
#  fixed-window history to reduce fingertip jitter while keeping
#  latency low.  The recency-weighted scheme means the smoother
#  "wakes up" immediately when the finger moves fast.
# ══════════════════════════════════════════════════════════════════
class CoordinateSmoother:
    """EMA + history-window smoother for 2-D finger coordinates."""

    def __init__(
        self,
        window: int = Config.SMOOTH_WINDOW,
        alpha: float = Config.SMOOTH_ALPHA,
    ):
        self._hx: Deque[int] = collections.deque(maxlen=window)
        self._hy: Deque[int] = collections.deque(maxlen=window)
        self._alpha = alpha
        # Pre-compute weight arrays up to the max window length
        self._weight_cache: dict = {}

    def _weights(self, n: int) -> np.ndarray:
        """Return normalised EMA weights for length n (cached)."""
        if n not in self._weight_cache:
            w = np.array([self._alpha ** (n - 1 - i) for i in range(n)],
                         dtype=np.float32)
            w /= w.sum()
            self._weight_cache[n] = w
        return self._weight_cache[n]

    def update(self, x: int, y: int) -> Tuple[int, int]:
        self._hx.append(x)
        self._hy.append(y)
        n = len(self._hx)
        w = self._weights(n)
        sx = int(np.dot(w, list(self._hx)))
        sy = int(np.dot(w, list(self._hy)))
        return sx, sy

    def reset(self) -> None:
        self._hx.clear()
        self._hy.clear()

    @property
    def ready(self) -> bool:
        return len(self._hx) >= 2


# ══════════════════════════════════════════════════════════════════
#  SECTION 4 ─ GESTURE DEBOUNCER
#
#  State machine that:
#    1. Requires N consecutive identical frames before transitioning
#    2. Enforces a cooldown period after every transition
#    3. Uses a separate, longer hold requirement for FIST (clear)
#
#  This eliminates accidental flickers and mis-triggers.
# ══════════════════════════════════════════════════════════════════
class GestureDebouncer:
    """
    State-machine gesture stabiliser with independent cooldown,
    confirmation window, and FIST hold timer.
    """

    def __init__(
        self,
        confirm: int  = Config.GESTURE_CONFIRM_FRAMES,
        cooldown: int = Config.GESTURE_COOLDOWN_FRAMES,
        fist_hold: int = Config.FIST_CLEAR_FRAMES,
    ):
        self._confirm     = confirm
        self._cooldown    = cooldown
        self._fist_hold   = fist_hold

        self._stable      = GestureState.IDLE   # Last confirmed state
        self._candidate   = GestureState.IDLE   # Current candidate
        self._cand_count  = 0                   # Consecutive frames of candidate
        self._cd_remain   = 0                   # Cooldown frames remaining
        self._fist_count  = 0                   # Consecutive FIST frames

        # FIST fires once then requires reset — prevent repeat clears
        self._fist_fired  = False

    def update(self, raw: GestureState) -> Tuple[GestureState, bool]:
        """
        Feed raw gesture per frame.
        Returns (stable_gesture, fist_triggered_this_frame).
        """
        # ── Tick cooldown ──────────────────────────────────────
        if self._cd_remain > 0:
            self._cd_remain -= 1

        # ── FIST special path ──────────────────────────────────
        if raw == GestureState.FIST:
            self._fist_count += 1
        else:
            self._fist_count = 0
            self._fist_fired = False   # Reset so next fist can fire

        fist_triggered = (
            self._fist_count >= self._fist_hold and not self._fist_fired
        )
        if fist_triggered:
            self._fist_fired = True
            self._fist_count = 0
            self._stable = GestureState.FIST
            return self._stable, True

        # ── Normal candidate tracking ──────────────────────────
        if raw == self._candidate:
            self._cand_count += 1
        else:
            self._candidate  = raw
            self._cand_count = 1

        # Transition only when confirmed and not in cooldown
        if (
            self._cand_count >= self._confirm
            and self._cd_remain == 0
            and raw != GestureState.FIST       # FIST handled above
        ):
            if raw != self._stable:
                self._cd_remain = self._cooldown
                self._stable = raw

        return self._stable, False


# ══════════════════════════════════════════════════════════════════
#  SECTION 5 ─ CANVAS MANAGER
#
#  Owns the drawing surface and provides:
#    • draw_line / erase primitives
#    • undo / redo via copy-on-write snapshot stack
#    • save / load from disk
#    • efficient dirty-flag tracking to reduce snapshot frequency
# ══════════════════════════════════════════════════════════════════
class CanvasManager:
    """BGR drawing canvas with undo/redo and file I/O."""

    def __init__(self, h: int, w: int, max_undo: int = Config.MAX_UNDO_STEPS):
        self._h   = h
        self._w   = w
        self._img = np.zeros((h, w, 3), dtype=np.uint8)

        self._undo: Deque[np.ndarray] = collections.deque(maxlen=max_undo)
        self._redo: Deque[np.ndarray] = collections.deque(maxlen=max_undo)

        # Dirty tracking: how many pixels changed since last snapshot
        self._dirty_px: int = 0

    # ── Private ────────────────────────────────────────────────
    def _push_undo(self) -> None:
        self._undo.append(self._img.copy())
        self._redo.clear()
        self._dirty_px = 0

    # ── Public primitives ──────────────────────────────────────
    def draw_line(
        self,
        p1: Tuple[int, int],
        p2: Tuple[int, int],
        color: Tuple[int, int, int],
        thickness: int,
    ) -> None:
        cv2.line(self._img, p1, p2, color, thickness, lineType=cv2.LINE_AA)
        # Approximate pixels touched by this stroke
        dist = int(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        self._dirty_px += dist * thickness
        if self._dirty_px >= Config.UNDO_SNAPSHOT_THRESHOLD:
            self._push_undo()

    def erase(self, center: Tuple[int, int], radius: int) -> None:
        cv2.circle(self._img, center, radius, (0, 0, 0), -1)

    def clear(self) -> None:
        self._push_undo()
        self._img = np.zeros((self._h, self._w, 3), dtype=np.uint8)

    # ── Undo / Redo ────────────────────────────────────────────
    def undo(self) -> bool:
        if not self._undo:
            return False
        self._redo.append(self._img.copy())
        self._img = self._undo.pop()
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append(self._img.copy())
        self._img = self._redo.pop()
        return True

    # ── Snapshot for undo when lifting pen ────────────────────
    def commit_stroke(self) -> None:
        """Call once after a drawing stroke ends to lock in undo state."""
        if self._dirty_px > 0:
            self._push_undo()

    # ── File I/O ───────────────────────────────────────────────
    def save(self, path: str) -> bool:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            ok = cv2.imwrite(path, self._img)
            if ok:
                log.info("Canvas saved → %s", path)
            return ok
        except Exception as exc:
            log.error("Save failed: %s", exc)
            return False

    def load(self, path: str) -> bool:
        try:
            img = cv2.imread(path)
            if img is None:
                raise FileNotFoundError(path)
            self._push_undo()
            self._img = cv2.resize(img, (self._w, self._h))
            log.info("Canvas loaded ← %s", path)
            return True
        except Exception as exc:
            log.error("Load failed: %s", exc)
            return False

    # ── Accessors ──────────────────────────────────────────────
    @property
    def image(self) -> np.ndarray:
        return self._img

    @property
    def shape(self) -> Tuple[int, int]:
        return self._h, self._w


# ══════════════════════════════════════════════════════════════════
#  SECTION 6 ─ HAND GESTURE CLASSIFIER
#
#  Stateless utility — takes raw MediaPipe landmarks and returns
#  a GestureState.  Separated from the debouncer so the logic is
#  unit-testable.
# ══════════════════════════════════════════════════════════════════
class GestureClassifier:
    """Pure-function gesture classification from MediaPipe landmarks."""

    # Finger landmark indices: (TIP, PIP)
    _FINGERS: Tuple[Tuple[int, int], ...] = (
        (8,  6),   # Index
        (12, 10),  # Middle
        (16, 14),  # Ring
        (20, 18),  # Pinky
    )

    @classmethod
    def fingers_up(cls, lm) -> Tuple[bool, bool, bool, bool]:
        """
        Returns (index, middle, ring, pinky) extension state.
        A finger is "up" when its tip y < its PIP y (image coords, Y↓).
        """
        return tuple(lm[tip].y < lm[pip].y for tip, pip in cls._FINGERS)

    @classmethod
    def classify(cls, lm) -> GestureState:
        """
        Priority order: FIST > ERASING > DRAWING > SPACE > IDLE
        Resolves the eraser/space conflict by requiring ALL 4 fingers
        for erase, exactly index-only for draw, index+middle for space.
        """
        idx, mid, rng, pnk = cls.fingers_up(lm)

        # ✊ All fingers curled
        if not idx and not mid and not rng and not pnk:
            return GestureState.FIST

        # ✋ All 4 fingers extended — ERASE
        if idx and mid and rng and pnk:
            return GestureState.ERASING

        # ☝  Index only — DRAW
        if idx and not mid and not rng and not pnk:
            return GestureState.DRAWING

        # ✌  Index + Middle — SPACE (lift pen)
        if idx and mid and not rng and not pnk:
            return GestureState.SPACE

        return GestureState.IDLE


# ══════════════════════════════════════════════════════════════════
#  SECTION 7 ─ OCR PROCESSOR
#
#  Multi-stage preprocessing pipeline optimised for handwritten
#  mathematical notation on a black canvas:
#    1. Upscale (Tesseract accuracy degrades below ~150 DPI)
#    2. Adaptive threshold  (handles brush pressure variation)
#    3. Morphological close (bridges small gaps in strokes)
#    4. Fast non-local means denoising
# ══════════════════════════════════════════════════════════════════
class OCRProcessor:
    """Optimised Tesseract OCR pipeline with SymPy equation solving."""

    def __init__(self) -> None:
        pytesseract.pytesseract.tesseract_cmd = Config.TESSERACT_PATH
        # Morphological kernel re-used across calls (avoid repeated alloc)
        self._morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (3, 3)
        )

    # ── Preprocessing ──────────────────────────────────────────
    def preprocess(self, canvas_bgr: np.ndarray) -> np.ndarray:
        """Return a binary image optimised for OCR."""
        # 1. Grayscale
        gray = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2GRAY)

        # 2. Upscale — Tesseract works best at ~300 DPI equivalent
        scale = Config.OCR_SCALE
        gray = cv2.resize(
            gray, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )

        # 3. Adaptive threshold — handles uneven stroke brightness
        #    blockSize must be odd; C controls background suppression
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=21,
            C=8,
        )

        # 4. Morphological closing — fills micro-gaps in brush strokes
        closed = cv2.morphologyEx(
            thresh, cv2.MORPH_CLOSE,
            self._morph_kernel, iterations=2,
        )

        # 5. Denoising — remove isolated speckles
        denoised = cv2.fastNlMeansDenoising(closed, h=12)

        return denoised

    # ── OCR ────────────────────────────────────────────────────
    def recognize(self, canvas_bgr: np.ndarray) -> str:
        """Run OCR and return stripped text string."""
        processed = self.preprocess(canvas_bgr)
        try:
            text = pytesseract.image_to_string(
                processed, config=Config.TESSERACT_CONFIG
            )
            return text.strip()
        except pytesseract.TesseractNotFoundError:
            log.error(
                "Tesseract binary not found.  "
                "Set the TESSERACT_PATH environment variable."
            )
            return ""
        except Exception as exc:
            log.error("OCR error: %s", exc)
            return ""

    # ── Solver ─────────────────────────────────────────────────
    @staticmethod
    def solve(text: str) -> str:
        """
        Attempt to evaluate/solve with SymPy.
        Handles expressions (3+4*2) and equations (x^2+2x=8).
        """
        if not text:
            return ""
        # Normalise common notation variants
        cleaned = (
            text
            .replace("^", "**")
            .replace("×", "*")
            .replace("÷", "/")
            .replace("x", "x")   # keep as symbol
        )
        try:
            expr = sp.sympify(cleaned, evaluate=True)
            return str(expr)
        except Exception:
            pass
        try:
            x = sp.Symbol("x")
            solutions = sp.solve(sp.sympify(cleaned), x)
            return f"x = {solutions}"
        except Exception:
            return "⚠ Cannot parse"


# ══════════════════════════════════════════════════════════════════
#  SECTION 8 ─ UI RENDERER
#
#  All HUD elements drawn here so the main loop stays clean:
#    • FPS counter (rolling 30-frame average)
#    • Gesture state indicator
#    • Colour palette
#    • Notification toast
#    • Keyboard shortcuts help panel
# ══════════════════════════════════════════════════════════════════
class UIRenderer:
    """Renders all HUD overlays onto the display frame."""

    _GESTURE_META = {
        GestureState.IDLE:    ("IDLE",       (140, 140, 140)),
        GestureState.DRAWING: ("DRAW  ☝",    (80,  220, 80 )),
        GestureState.SPACE:   ("SPACE ✌",    (220, 220, 80 )),
        GestureState.ERASING: ("ERASE ✋",   (80,  80,  220)),
        GestureState.FIST:    ("CLEAR ✊",   (80,  160, 255)),
    }

    _HELP = [
        ("☝",  "Draw"),
        ("✌",  "Lift pen"),
        ("✋",  "Erase"),
        ("✊",  "Clear (hold)"),
        ("r",   "OCR + Solve"),
        ("s",   "Save"),
        ("l",   "Load last"),
        ("z",   "Undo"),
        ("y",   "Redo"),
        ("+/-", "Brush size"),
        ("1-8", "Pick colour"),
        ("h",   "Toggle help"),
        ("q",   "Quit"),
    ]

    def __init__(self) -> None:
        self._show_help   = False
        self._notif_text  = ""
        self._notif_timer = 0

        # FPS tracking
        self._fps_buf: Deque[float] = collections.deque(maxlen=30)
        self._t_prev   = time.perf_counter()

    # ── FPS ────────────────────────────────────────────────────
    def tick(self) -> float:
        now = time.perf_counter()
        dt  = now - self._t_prev
        self._t_prev = now
        if dt > 1e-6:
            self._fps_buf.append(1.0 / dt)
        return float(np.mean(self._fps_buf)) if self._fps_buf else 0.0

    # ── Notifications ──────────────────────────────────────────
    def notify(self, msg: str, dur: int = Config.NOTIF_DURATION) -> None:
        self._notif_text  = msg
        self._notif_timer = dur

    def toggle_help(self) -> None:
        self._show_help = not self._show_help

    # ── Drawing helpers ────────────────────────────────────────
    @staticmethod
    def _label(
        img: np.ndarray,
        text: str,
        pos: Tuple[int, int],
        color: Tuple[int, int, int] = (230, 230, 230),
        scale: float = 0.62,
        thickness: int = 1,
        bg: bool = True,
    ) -> None:
        """Draw text with an optional dark background rectangle."""
        (tw, th), bl = cv2.getTextSize(text, Config.FONT, scale, thickness)
        x, y = pos
        if bg:
            cv2.rectangle(
                img,
                (x - 4, y - th - 4),
                (x + tw + 4, y + bl + 2),
                (15, 15, 15), -1,
            )
        cv2.putText(img, text, pos, Config.FONT, scale, color, thickness, cv2.LINE_AA)

    # ── Palette ────────────────────────────────────────────────
    @staticmethod
    def _draw_palette(img: np.ndarray, active: int) -> None:
        W   = img.shape[1]
        box = Config.PALETTE_BOX
        gap = Config.PALETTE_GAP
        total_w = len(Config.PALETTE) * (box + gap) - gap
        sx  = W - total_w - 12
        sy  = 10
        for i, color in enumerate(Config.PALETTE):
            x = sx + i * (box + gap)
            cv2.rectangle(img, (x, sy), (x + box, sy + box), color, -1)
            # White active border
            border_col = (255, 255, 255) if i == active else (60, 60, 60)
            cv2.rectangle(img, (x - 2, sy - 2), (x + box + 2, sy + box + 2),
                          border_col, 2)
            # Keyboard hint
            cv2.putText(img, str(i + 1), (x + 4, sy + box - 4),
                        Config.FONT, 0.35, (0, 0, 0), 1, cv2.LINE_AA)

    # ── Main render ────────────────────────────────────────────
    def render(
        self,
        frame: np.ndarray,
        fps: float,
        gesture: GestureState,
        color_idx: int,
        brush: int,
        undo_depth: int,
    ) -> np.ndarray:
        out = frame  # mutate in-place (caller passes a copy)
        H, W = out.shape[:2]

        # FPS
        self._label(out, f"FPS  {fps:5.1f}", (10, 28), (80, 230, 80))

        # Gesture
        glabel, gcol = self._GESTURE_META.get(
            gesture, ("?", (200, 200, 200))
        )
        self._label(out, f"Gesture  {glabel}", (10, 60), gcol)

        # Brush / undo info
        self._label(out, f"Brush {brush}px   Undo {undo_depth}", (10, 92))

        # Palette
        self._draw_palette(out, color_idx)

        # Notification toast
        if self._notif_timer > 0:
            alpha = min(1.0, self._notif_timer / 30.0)
            col   = (int(0 * alpha), int(220 * alpha), int(220 * alpha))
            tw, _ = cv2.getTextSize(self._notif_text, Config.FONT, 0.75, 2)[:2]
            nx    = max(0, (W - tw[0]) // 2)
            self._label(out, self._notif_text, (nx, H - 36), col, scale=0.75, thickness=2)
            self._notif_timer -= 1

        # Help overlay
        if self._show_help:
            pw  = 230
            lh  = 22
            ph  = len(self._HELP) * lh + 30
            px  = W - pw - 10
            py  = 50
            overlay = out.copy()
            cv2.rectangle(overlay, (px, py), (px + pw, py + ph), (20, 20, 20), -1)
            cv2.addWeighted(overlay, Config.UI_ALPHA, out, 1 - Config.UI_ALPHA, 0, out)
            cv2.rectangle(out, (px, py), (px + pw, py + ph), (80, 80, 80), 1)
            cv2.putText(out, "Keyboard Shortcuts", (px + 8, py + 18),
                        Config.FONT, 0.5, (180, 220, 255), 1, cv2.LINE_AA)
            for j, (key, desc) in enumerate(self._HELP):
                ty = py + 30 + j * lh
                cv2.putText(out, f"[{key}]", (px + 8, ty),
                            Config.FONT, 0.48, (220, 200, 80), 1, cv2.LINE_AA)
                cv2.putText(out, desc, (px + 70, ty),
                            Config.FONT, 0.48, (210, 210, 210), 1, cv2.LINE_AA)

        return out


# ══════════════════════════════════════════════════════════════════
#  SECTION 9 ─ WEBCAM MANAGER
#
#  Wraps cv2.VideoCapture with:
#    • Retry logic on failure
#    • Minimal buffer size (CAP_PROP_BUFFERSIZE=1) to cut latency
#    • Configurable resolution + FPS
# ══════════════════════════════════════════════════════════════════
class WebcamManager:
    """Robust webcam wrapper with retry and low-latency settings."""

    def __init__(self) -> None:
        self._cap: Optional[cv2.VideoCapture] = None
        self._h: int = Config.CAMERA_HEIGHT
        self._w: int = Config.CAMERA_WIDTH

    def open(self, retries: int = 3) -> bool:
        backend = cv2.CAP_DSHOW if Config.USE_DSHOW else cv2.CAP_ANY
        for attempt in range(1, retries + 1):
            cap = cv2.VideoCapture(Config.CAMERA_INDEX, backend)
            if not cap.isOpened():
                log.warning("Camera open attempt %d/%d failed.", attempt, retries)
                time.sleep(0.5)
                continue
            # ── Apply settings ──────────────────────────────
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  Config.CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS,          Config.CAMERA_FPS)
            cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # 1-frame buffer = min latency
            # Report actual negotiated resolution
            self._w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            log.info(
                "Camera opened  [%dx%d @ %.0f FPS]",
                self._w, self._h, actual_fps,
            )
            self._cap = cap
            return True
        log.critical("Could not open camera after %d attempts.", retries)
        return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cap and self._cap.isOpened():
            return self._cap.read()
        return False, None

    def release(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None
            log.info("Camera released.")

    @property
    def frame_size(self) -> Tuple[int, int]:
        """Returns (height, width)."""
        return self._h, self._w


# ══════════════════════════════════════════════════════════════════
#  SECTION 10 ─ MAIN APPLICATION
#
#  Orchestrates every subsystem in a clean, readable event loop.
#  The loop follows this fixed pipeline every frame:
#
#    read → flip → MediaPipe → classify → debounce → draw/erase
#    → composite → HUD → display → keyboard
# ══════════════════════════════════════════════════════════════════
class AIWhiteboardApp:
    """
    Top-level controller.  Owns subsystem instances and runs
    the main capture/render loop.
    """

    def __init__(self) -> None:
        log.info("AI Air Whiteboard v4.0  —  initialising…")

        # ── Subsystems ────────────────────────────────────────
        self._cam    = WebcamManager()
        self._ocr    = OCRProcessor()
        self._ui     = UIRenderer()
        self._canvas: Optional[CanvasManager] = None   # created after cam opens

        # ── MediaPipe ────────────────────────────────────────
        mp_h = mp.solutions.hands
        self._mp_hands    = mp_h
        self._mp_draw     = mp.solutions.drawing_utils
        self._draw_spec_lm = self._mp_draw.DrawingSpec(
            color=(80, 22, 10), thickness=2, circle_radius=3
        )
        self._draw_spec_cn = self._mp_draw.DrawingSpec(
            color=(80, 44, 121), thickness=2, circle_radius=2
        )
        self._hands = mp_h.Hands(
            static_image_mode=False,
            max_num_hands=Config.MAX_HANDS,
            model_complexity=Config.MP_MODEL_COMPLEXITY,
            min_detection_confidence=Config.MP_DETECTION_CONFIDENCE,
            min_tracking_confidence=Config.MP_TRACKING_CONFIDENCE,
        )

        # ── Per-hand state (indexed 0..MAX_HANDS-1) ──────────
        self._smoothers:  List[CoordinateSmoother] = [
            CoordinateSmoother() for _ in range(Config.MAX_HANDS)
        ]
        self._debouncers: List[GestureDebouncer] = [
            GestureDebouncer() for _ in range(Config.MAX_HANDS)
        ]
        self._prev_pts: List[Optional[Tuple[int, int]]] = [
            None
        ] * Config.MAX_HANDS
        # Track whether a hand was in DRAWING last frame (to commit undo)
        self._was_drawing: List[bool] = [False] * Config.MAX_HANDS

        # ── App state ────────────────────────────────────────
        self._color_idx: int = 0
        self._brush:     int = Config.BRUSH_THICKNESS
        self._running:   bool = False

        # Pre-allocated mask for canvas compositing
        self._comp_mask: Optional[np.ndarray] = None

    # ──────────────────────────────────────────────────────────
    # INITIALISATION
    # ──────────────────────────────────────────────────────────
    def _init(self) -> bool:
        if not self._cam.open():
            return False
        h, w = self._cam.frame_size
        self._canvas    = CanvasManager(h, w)
        self._comp_mask = np.zeros((h, w), dtype=np.uint8)
        # Attempt to restore last autosave
        autosave = os.path.join(Config.SAVE_DIR, Config.AUTOSAVE_FILE)
        if os.path.isfile(autosave):
            self._canvas.load(autosave)
            log.info("Autosave restored from %s", autosave)
        return True

    # ──────────────────────────────────────────────────────────
    # PER-HAND GESTURE PROCESSING
    # ──────────────────────────────────────────────────────────
    def _process_hand(
        self,
        hand_idx: int,
        lm,
        frame: np.ndarray,
        H: int,
        W: int,
    ) -> GestureState:
        """
        Classify, debounce, and act on a single hand's landmarks.
        Returns the stable gesture for this hand.
        """
        raw     = GestureClassifier.classify(lm)
        stable, fist_now = self._debouncers[hand_idx].update(raw)

        # ── Pixel coordinates ──────────────────────────────
        ix = int(lm[8].x * W)    # Index tip
        iy = int(lm[8].y * H)
        px = int(lm[9].x * W)    # Palm centre
        py = int(lm[9].y * H)

        smoother = self._smoothers[hand_idx]
        canvas   = self._canvas

        # ── FIST: immediate clear ──────────────────────────
        if fist_now:
            smoother.reset()
            self._prev_pts[hand_idx]  = None
            self._was_drawing[hand_idx] = False
            canvas.clear()
            self._ui.notify("🧹  Board cleared")
            return GestureState.FIST

        # ── DRAWING ────────────────────────────────────────
        if stable == GestureState.DRAWING:
            sx, sy = smoother.update(ix, iy)
            prev   = self._prev_pts[hand_idx]

            if prev is not None:
                dist = np.hypot(sx - prev[0], sy - prev[1])
                if dist >= Config.MIN_DRAW_DIST:
                    canvas.draw_line(
                        prev, (sx, sy),
                        Config.PALETTE[self._color_idx],
                        self._brush,
                    )

            self._prev_pts[hand_idx]    = (sx, sy)
            self._was_drawing[hand_idx] = True

            # Visual cursor
            cv2.circle(frame, (sx, sy),
                       max(4, self._brush // 2),
                       Config.PALETTE[self._color_idx], -1)

        # ── ERASING ────────────────────────────────────────
        elif stable == GestureState.ERASING:
            # Commit any prior stroke before erasing
            if self._was_drawing[hand_idx]:
                canvas.commit_stroke()
                self._was_drawing[hand_idx] = False

            self._prev_pts[hand_idx] = None
            smoother.reset()

            canvas.erase((px, py), Config.ERASER_RADIUS)
            cv2.circle(frame, (px, py), Config.ERASER_RADIUS, (60, 60, 220), 2)
            cv2.circle(frame, (px, py), 4, (60, 60, 220), -1)
            UIRenderer._label(
                frame, "ERASE",
                (px - 20, py - Config.ERASER_RADIUS - 10),
                (60, 60, 220), scale=0.55,
            )

        # ── SPACE / IDLE: lift pen ─────────────────────────
        else:
            if self._was_drawing[hand_idx]:
                canvas.commit_stroke()
                self._was_drawing[hand_idx] = False
            self._prev_pts[hand_idx] = None
            smoother.reset()

        return stable

    # ──────────────────────────────────────────────────────────
    # CANVAS → FRAME COMPOSITING
    #
    # Strategy: only overwrite pixels where ink exists.
    # This preserves the camera feed everywhere else and avoids
    # a full-frame blend each tick.
    # ──────────────────────────────────────────────────────────
    def _composite(self, frame: np.ndarray) -> None:
        """Blend canvas ink onto frame in-place."""
        canvas_bgr = self._canvas.image
        # Build binary mask of painted pixels
        cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2GRAY, dst=self._comp_mask)
        cv2.threshold(
            self._comp_mask, 1, 255, cv2.THRESH_BINARY, dst=self._comp_mask
        )
        # Where mask is set: blend strongly toward canvas colour
        ink_region = self._comp_mask > 0
        frame[ink_region] = cv2.addWeighted(
            frame, 1.0 - Config.CANVAS_INK_WEIGHT,
            canvas_bgr, Config.CANVAS_INK_WEIGHT, 0,
        )[ink_region]

    # ──────────────────────────────────────────────────────────
    # KEYBOARD HANDLER
    # ──────────────────────────────────────────────────────────
    def _handle_key(self, key: int) -> bool:
        """Process a key press.  Returns False to request quit."""

        if key == ord("q"):
            return False

        elif key == ord("r"):
            self._do_ocr()

        elif key == ord("s"):
            ts   = int(time.time())
            path = os.path.join(Config.SAVE_DIR, f"drawing_{ts}.png")
            if self._canvas.save(path):
                self._ui.notify(f"💾  Saved → {os.path.basename(path)}")
            else:
                self._ui.notify("⚠  Save failed")

        elif key == ord("l"):
            path = os.path.join(Config.SAVE_DIR, Config.AUTOSAVE_FILE)
            if self._canvas.load(path):
                self._ui.notify("📂  Drawing loaded")
            else:
                self._ui.notify(f"⚠  No file at {path}")

        elif key == ord("z"):
            if self._canvas.undo():
                self._ui.notify("↩  Undo")
            else:
                self._ui.notify("⚠  Nothing to undo")

        elif key == ord("y"):
            if self._canvas.redo():
                self._ui.notify("↪  Redo")
            else:
                self._ui.notify("⚠  Nothing to redo")

        elif key == ord("h"):
            self._ui.toggle_help()

        elif key in (ord("+"), ord("=")):
            self._brush = min(self._brush + Config.BRUSH_STEP, Config.BRUSH_MAX)
            self._ui.notify(f"Brush  {self._brush} px")

        elif key == ord("-"):
            self._brush = max(self._brush - Config.BRUSH_STEP, Config.BRUSH_MIN)
            self._ui.notify(f"Brush  {self._brush} px")

        elif ord("1") <= key <= ord("8"):
            idx = key - ord("1")
            if idx < len(Config.PALETTE):
                self._color_idx = idx
                self._ui.notify(f"Colour  {idx + 1}")

        return True

    # ──────────────────────────────────────────────────────────
    # OCR + SOLVE
    # ──────────────────────────────────────────────────────────
    def _do_ocr(self) -> None:
        log.info("Running OCR …")
        text = self._ocr.recognize(self._canvas.image)
        log.info("Recognised  : %s", text or "<empty>")
        if text:
            answer = self._ocr.solve(text)
            log.info("Solved      : %s", answer)
            short = text[:28] + ("…" if len(text) > 28 else "")
            self._ui.notify(f"'{short}'  →  {answer}", dur=150)
        else:
            self._ui.notify("⚠  OCR: no text detected")

    # ──────────────────────────────────────────────────────────
    # MAIN LOOP
    # ──────────────────────────────────────────────────────────
    def run(self) -> None:
        if not self._init():
            log.critical("Initialisation failed.  Exiting.")
            return

        H, W = self._canvas.shape
        self._running = True
        log.info("Running — press [h] for shortcuts, [q] to quit.")

        try:
            while self._running:
                # ── 1. Capture ─────────────────────────────
                ok, frame = self._cam.read()
                if not ok or frame is None:
                    log.warning("Frame drop — skipping.")
                    continue

                frame = cv2.flip(frame, 1)

                # ── 2. MediaPipe hand detection ────────────
                #    Setting writeable=False avoids an internal copy
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                mp_result = self._hands.process(rgb)
                rgb.flags.writeable = True

                # ── 3. Hand processing ─────────────────────
                dominant_gesture = GestureState.IDLE

                if mp_result.multi_hand_landmarks:
                    for hi, hand_lm in enumerate(mp_result.multi_hand_landmarks):
                        if hi >= Config.MAX_HANDS:
                            break
                        # Draw skeleton
                        self._mp_draw.draw_landmarks(
                            frame, hand_lm,
                            self._mp_hands.HAND_CONNECTIONS,
                            self._draw_spec_lm,
                            self._draw_spec_cn,
                        )
                        g = self._process_hand(hi, hand_lm.landmark, frame, H, W)
                        if g not in (GestureState.IDLE, GestureState.SPACE):
                            dominant_gesture = g

                else:
                    # No hands in frame — commit any open stroke
                    for hi in range(Config.MAX_HANDS):
                        if self._was_drawing[hi]:
                            self._canvas.commit_stroke()
                            self._was_drawing[hi] = False
                        self._smoothers[hi].reset()
                        self._prev_pts[hi] = None

                # ── 4. Composite canvas onto frame ─────────
                self._composite(frame)

                # ── 5. HUD ─────────────────────────────────
                fps = self._ui.tick()
                output = self._ui.render(
                    frame, fps, dominant_gesture,
                    self._color_idx, self._brush,
                    undo_depth=len(self._canvas._undo),
                )

                # ── 6. Display ─────────────────────────────
                cv2.imshow("AI Air Whiteboard  v4.0", output)

                # ── 7. Keyboard ────────────────────────────
                key = cv2.waitKey(1) & 0xFF
                if key != 0xFF and not self._handle_key(key):
                    break

        except KeyboardInterrupt:
            log.info("Interrupted by user (Ctrl-C).")
        finally:
            self._shutdown()

    # ──────────────────────────────────────────────────────────
    # SHUTDOWN
    # ──────────────────────────────────────────────────────────
    def _shutdown(self) -> None:
        log.info("Shutting down …")
        # Always autosave so user never loses work
        if self._canvas is not None:
            path = os.path.join(Config.SAVE_DIR, Config.AUTOSAVE_FILE)
            self._canvas.save(path)
        self._hands.close()
        self._cam.release()
        cv2.destroyAllWindows()
        log.info("✅  Shutdown complete.")


# ══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    AIWhiteboardApp().run()