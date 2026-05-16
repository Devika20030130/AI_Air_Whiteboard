#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║        AI Air Whiteboard — v5.0  (Hybrid OCR Edition)           ║
║──────────────────────────────────────────────────────────────────║
║  Real-time gesture whiteboard + production-grade hybrid OCR      ║
║                                                                  ║
║  OCR Stack : TrOCR (transformer) → EasyOCR → Tesseract          ║
║  Gesture   : MediaPipe Hands  +  EMA smoother + state machine    ║
║  Solver    : SymPy (arithmetic · algebra · exponents · fractions)║
╚══════════════════════════════════════════════════════════════════╝

Python   : 3.11+
Author   : github.com/yourname
License  : MIT
"""

from __future__ import annotations

# ── stdlib ────────────────────────────────────────────────────────
import collections
import os
import sys
import time
import threading                        # NEW: async OCR runs off the main thread
from enum import Enum, auto
from typing import Deque, List, Optional, Tuple

# ── third-party ───────────────────────────────────────────────────
import cv2
import mediapipe as mp
import numpy as np

# ── project ───────────────────────────────────────────────────────
from config import CFG
from ocr.hybrid_ocr import HybridOCRPipeline, OCRResult
from utils.logger import get_logger

log = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════
#  ENUMS
# ══════════════════════════════════════════════════════════════════
class GestureState(Enum):
    IDLE    = auto()
    DRAWING = auto()
    SPACE   = auto()
    ERASING = auto()
    FIST    = auto()


# ══════════════════════════════════════════════════════════════════
#  FINGERTIP TRACKER  —  Kalman filter  +  EMA  +  motion deadzone
#
#  Why three layers instead of one?
#  ─────────────────────────────────
#  MediaPipe landmark noise has two distinct components:
#
#  1. HIGH-FREQUENCY JITTER  (< 3 px peak-to-peak, random, every frame)
#     Caused by: sub-pixel detection uncertainty, JPEG compression
#     artifacts in the webcam stream, minor hand tremor.
#     Fix: Kalman filter — models the finger as a point mass with
#     constant velocity and uses the prediction-correction cycle to
#     suppress measurement noise without introducing systematic lag.
#
#  2. LOW-FREQUENCY DRIFT  (3–8 px, semi-correlated, multi-frame)
#     Caused by: MediaPipe's temporal smoothing producing oscillations
#     around the true position when the hand moves slowly.
#     Fix: EMA window — weighted average of recent Kalman outputs.
#     The EMA acts as a second-order low-pass after the Kalman so
#     drift is damped without removing intentional slow strokes.
#
#  3. STATIONARY NOISE  (< deadzone_px when finger held still)
#     Caused by: all of the above, combined.  When drawing a period
#     or a comma the finger barely moves but the raw coordinate
#     wanders ± 3 px, producing a fuzzy blob instead of a dot.
#     Fix: deadzone — if the post-EMA displacement from the last
#     committed point is < deadzone_px, return the last committed
#     point unchanged.  This freezes the drawing cursor when still.
#
#  Signal flow per frame
#  ──────────────────────
#  raw (x,y)  →  Kalman.correct()  →  EMA window  →  deadzone gate
#                     ↑ Kalman.predict() at start of each frame
#
#  Performance
#  ───────────
#  cv2.KalmanFilter uses internally optimised C++ BLAS routines.
#  The EMA is a single np.dot over a deque of length 6.
#  Total cost: < 0.05 ms per frame per hand on a mid-range CPU.
# ══════════════════════════════════════════════════════════════════
class FingertipTracker:
    """
    Per-hand fingertip coordinate stabiliser combining:
      • 2-D constant-velocity Kalman filter
      • Exponential Moving Average (EMA) post-filter
      • Motion deadzone gate

    One instance is created per tracked hand; all state is encapsulated
    so multi-hand usage requires no shared mutable globals.

    Usage
    -----
    tracker = FingertipTracker()
    for each frame:
        sx, sy = tracker.update(raw_x, raw_y)
        # sx, sy are stabilised pixel coordinates
    tracker.reset()   # call when hand disappears from frame
    """

    def __init__(self) -> None:
        d = CFG.drawing

        # ── Kalman filter setup ────────────────────────────────
        # State vector  x = [px, py, vx, vy]  (position + velocity)
        # Measurement   z = [px, py]            (position only)
        #
        # cv2.KalmanFilter(dynamParams, measureParams)
        #   dynamParams  = 4  (state dimension)
        #   measureParams= 2  (measurement dimension)
        self._kf = cv2.KalmanFilter(4, 2)

        # Transition matrix  F  (constant-velocity model):
        #   [1 0 dt 0 ]        dt = 1 frame
        #   [0 1 0  dt]
        #   [0 0 1  0 ]
        #   [0 0 0  1 ]
        # At 30 FPS dt=1 means velocity is in px/frame units.
        self._kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]],
            dtype=np.float32,
        )

        # Measurement matrix  H:  z = H * x  →  observe px,py only
        self._kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]],
            dtype=np.float32,
        )

        # Process noise covariance  Q:
        #   Diagonal — each state variable is independent.
        #   Position noise < velocity noise (position is well-constrained
        #   by measurement; velocity is a latent variable).
        q = d.kalman_process_noise
        self._kf.processNoiseCov = np.diag(
            [q, q, q * 4, q * 4]         # vx,vy allowed 4× more uncertainty
        ).astype(np.float32)

        # Measurement noise covariance  R:
        #   Diagonal — x and y measurement errors are independent.
        r = d.kalman_measurement_noise
        self._kf.measurementNoiseCov = np.array(
            [[r, 0],
             [0, r]],
            dtype=np.float32,
        )

        # Initial posterior error covariance  P:
        #   Large initial value → filter converges quickly from any
        #   starting position instead of drifting from the origin.
        p = d.kalman_post_error
        self._kf.errorCovPost = np.eye(4, dtype=np.float32) * p

        # State post (initial position estimate — will be set on first
        # measurement so we mark it as uninitialised).
        self._kf.statePost = np.zeros((4, 1), dtype=np.float32)
        self._initialised = False

        # ── EMA post-filter ────────────────────────────────────
        self._alpha  = d.smooth_alpha
        self._hx: collections.deque = collections.deque(maxlen=d.smooth_window)
        self._hy: collections.deque = collections.deque(maxlen=d.smooth_window)
        # Weight cache: key = window length, value = normalised weight array
        self._weight_cache: dict = {}

        # ── Deadzone gate ──────────────────────────────────────
        self._deadzone    = d.deadzone_px
        # Last coordinate that passed the deadzone gate
        self._last_stable: Optional[Tuple[int, int]] = None

    # ──────────────────────────────────────────────────────────
    #  PUBLIC API
    # ──────────────────────────────────────────────────────────
    def update(self, raw_x: int, raw_y: int) -> Tuple[int, int]:
        """
        Feed one raw MediaPipe landmark position and return the
        stabilised (x, y) coordinate for this frame.

        Pipeline
        --------
        1. Kalman predict  — project state forward by one time step
        2. Kalman correct  — fuse prediction with new measurement
        3. EMA             — weighted average of recent Kalman outputs
        4. Deadzone gate   — freeze coordinate when displacement < threshold

        Parameters
        ----------
        raw_x, raw_y : Raw landmark pixel position from MediaPipe.

        Returns
        -------
        Stabilised (x, y) as integer pixel coordinates.
        """
        # ── Step 1 & 2: Kalman predict + correct ──────────────
        if not self._initialised:
            # Seed the state with the first measurement so the filter
            # does not waste frames converging from (0, 0).
            self._kf.statePost = np.array(
                [[raw_x], [raw_y], [0.0], [0.0]], dtype=np.float32
            )
            self._initialised = True

        # predict() advances the state by one time step using F.
        # Must be called every frame even if we do not use the
        # prediction directly, so the covariance matrix P evolves.
        self._kf.predict()

        # correct() fuses the new measurement with the prediction.
        # Returns the posterior state estimate.
        measurement = np.array([[raw_x], [raw_y]], dtype=np.float32)
        corrected   = self._kf.correct(measurement)

        kx = int(corrected[0, 0])
        ky = int(corrected[1, 0])

        # ── Step 3: EMA post-filter ────────────────────────────
        # Feed the Kalman output into the EMA window.
        # The EMA acts as a second-order smoother: the Kalman already
        # removed high-frequency noise; the EMA damps the residual
        # low-frequency oscillations that appear when the finger
        # moves slowly.
        self._hx.append(kx)
        self._hy.append(ky)

        n = len(self._hx)
        w = self._ema_weights(n)

        ex = int(np.dot(w, list(self._hx)))
        ey = int(np.dot(w, list(self._hy)))

        # ── Step 4: Deadzone gate ──────────────────────────────
        # If no stable coordinate exists yet, accept unconditionally.
        if self._last_stable is None:
            self._last_stable = (ex, ey)
            return ex, ey

        lx, ly = self._last_stable
        displacement = float(np.hypot(ex - lx, ey - ly))

        if displacement < self._deadzone:
            # Finger has not moved enough to justify updating the
            # drawing cursor — return the last accepted position.
            # This prevents the cursor from drifting when the hand
            # is stationary, which would smear dots and commas.
            return lx, ly

        # Displacement exceeds deadzone → accept and commit
        self._last_stable = (ex, ey)
        return ex, ey

    def reset(self) -> None:
        """
        Reset all state.

        Must be called when the hand leaves the frame so the Kalman
        filter does not carry over velocity from the previous stroke
        and produce a phantom streak at the start of the next one.
        """
        self._initialised = False
        self._kf.statePost = np.zeros((4, 1), dtype=np.float32)
        self._hx.clear()
        self._hy.clear()
        self._last_stable = None

    # ──────────────────────────────────────────────────────────
    #  PRIVATE HELPERS
    # ──────────────────────────────────────────────────────────
    def _ema_weights(self, n: int) -> np.ndarray:
        """
        Return a normalised EMA weight vector of length n (cached).

        w[i] = alpha^(n-1-i)  for i in 0..n-1
        Normalised so sum(w) = 1.

        Cached by length so repeated calls for the same n (the steady-
        state case after warm-up) cost a single dict lookup.
        """
        if n not in self._weight_cache:
            w = np.array(
                [self._alpha ** (n - 1 - i) for i in range(n)],
                dtype=np.float32,
            )
            w /= w.sum()
            self._weight_cache[n] = w
        return self._weight_cache[n]


# ══════════════════════════════════════════════════════════════════
#  GESTURE DEBOUNCER
# ══════════════════════════════════════════════════════════════════
class GestureDebouncer:
    """N-frame confirmation + cooldown + one-shot FIST trigger."""

    def __init__(self) -> None:
        g = CFG.gesture
        self._confirm   = g.confirm_frames
        self._cooldown  = g.cooldown_frames
        self._fist_hold = g.fist_clear_frames

        self._stable     = GestureState.IDLE
        self._candidate  = GestureState.IDLE
        self._cand_count = 0
        self._cd_remain  = 0
        self._fist_count = 0
        self._fist_fired = False

    def update(self, raw: GestureState) -> Tuple[GestureState, bool]:
        """Returns (stable_state, fist_triggered_this_frame)."""
        if self._cd_remain > 0:
            self._cd_remain -= 1

        # FIST special path
        if raw == GestureState.FIST:
            self._fist_count += 1
        else:
            self._fist_count = 0
            self._fist_fired = False

        fist_now = (
            self._fist_count >= self._fist_hold and not self._fist_fired
        )
        if fist_now:
            self._fist_fired = True
            self._fist_count = 0
            self._stable = GestureState.FIST
            return self._stable, True

        # Normal confirmation
        if raw == self._candidate:
            self._cand_count += 1
        else:
            self._candidate  = raw
            self._cand_count = 1

        if (self._cand_count >= self._confirm
                and self._cd_remain == 0
                and raw != GestureState.FIST):
            if raw != self._stable:
                self._cd_remain = self._cooldown
                self._stable    = raw

        return self._stable, False


# ══════════════════════════════════════════════════════════════════
#  GESTURE CLASSIFIER  (stateless)
# ══════════════════════════════════════════════════════════════════
class GestureClassifier:
    _FINGERS = ((8, 6), (12, 10), (16, 14), (20, 18))  # (TIP, PIP)

    @classmethod
    def classify(cls, lm) -> GestureState:
        idx, mid, rng, pnk = (lm[t].y < lm[p].y for t, p in cls._FINGERS)
        if not idx and not mid and not rng and not pnk:
            return GestureState.FIST
        if idx and mid and rng and pnk:
            return GestureState.ERASING
        if idx and not mid and not rng and not pnk:
            return GestureState.DRAWING
        if idx and mid and not rng and not pnk:
            return GestureState.SPACE
        return GestureState.IDLE


# ══════════════════════════════════════════════════════════════════
#  STROKE INTERPOLATOR
#
#  Problem being solved
#  ────────────────────
#  At 30 FPS with a fast-moving finger the fingertip can travel
#  40–80 px between consecutive frames.  A single cv2.line() call
#  covers that distance, but at the end-points the stroke width
#  creates a "sausage-link" appearance — flattened caps on diagonal
#  segments and visible gaps on sharp direction changes.
#
#  Solution
#  ────────
#  We linearly interpolate N sub-points between p1 and p2 spaced
#  ≤ interp_step_px apart, then:
#    1. Draw a LINE_AA segment between every consecutive sub-point
#       pair  →  fully continuous, anti-aliased coverage.
#    2. Draw a filled circle (radius = thickness/2) at every sub-point
#       →  round caps that fill concave corners on sharp turns.
#
#  The result is visually identical to a vector-path "round join +
#  round cap" stroke, which is exactly what OCR engines expect from
#  handwritten glyphs.
#
#  Performance
#  ───────────
#  np.linspace generates all sub-points in one vectorised call.
#  The subsequent loop is over integers (not floats) and calls
#  cv2.line only N-1 times — typically 1–6 iterations at 30 FPS.
#  On a mid-range CPU this adds < 0.1 ms per frame.
# ══════════════════════════════════════════════════════════════════
class StrokeInterpolator:
    """
    Produces a gapless, anti-aliased stroke between two 2-D points
    by dense linear interpolation.

    Used exclusively by CanvasManager.draw_stroke().
    Stateless — every call is independent.
    """

    @staticmethod
    def interpolated_points(
        p1: Tuple[int, int],
        p2: Tuple[int, int],
        step_px: float,
    ) -> List[Tuple[int, int]]:
        """
        Return a list of (x, y) integer points evenly spaced along
        the segment p1 → p2 with spacing ≤ step_px.

        Always includes p1 and p2 as the first and last element.

        Parameters
        ----------
        p1, p2   : Start and end pixel coordinates (integers).
        step_px  : Maximum distance between consecutive returned points.
                   Use 1.0 for maximum density (no gaps at any speed).

        Returns
        -------
        List of (x, y) tuples.  Length ≥ 2 even when p1 == p2.

        Implementation note
        -------------------
        np.linspace is used instead of np.arange so that p2 is
        always the exact final element regardless of floating-point
        rounding in the step calculation.
        """
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dist = float(np.hypot(dx, dy))

        if dist < 1e-6:
            # Points are identical — return a single point pair so
            # the caller can still draw a cap circle there.
            return [p1, p2]

        # Number of sub-segments: at least 1, ceiling of dist/step_px
        n_steps = max(1, int(np.ceil(dist / step_px)))

        # Vectorised interpolation — one call for both axes
        xs = np.linspace(p1[0], p2[0], n_steps + 1, dtype=np.float32)
        ys = np.linspace(p1[1], p2[1], n_steps + 1, dtype=np.float32)

        # Round to integer pixel coordinates
        return list(zip(xs.round().astype(int).tolist(),
                        ys.round().astype(int).tolist()))

    @staticmethod
    def draw_stroke(
        img:       np.ndarray,
        p1:        Tuple[int, int],
        p2:        Tuple[int, int],
        color:     Tuple[int, int, int],
        thickness: int,
        step_px:   float = 1.0,
        fill_caps: bool  = True,
    ) -> int:
        """
        Draw a gapless, anti-aliased stroke from p1 to p2 onto *img*
        (mutates in-place).

        Parameters
        ----------
        img        : BGR canvas ndarray (mutated in-place).
        p1, p2     : Start / end pixel coordinates.
        color      : BGR tuple.
        thickness  : Stroke width in pixels.
        step_px    : Interpolation step (px).  Default 1.0 = no gaps.
        fill_caps  : If True, draw a filled circle at every sub-point
                     to produce round joins and caps.

        Returns
        -------
        Pixel-distance between p1 and p2 as int (for dirty tracking).

        Why two rendering passes?
        ─────────────────────────
        Pass 1 — LINE_AA segments  : anti-aliased coverage along the
            stroke body; handles subpixel endpoints correctly.
        Pass 2 — filled circles    : round caps at each sub-point.
            cv2.circle with LINE_AA gives a soft edge that blends
            into adjacent segments, eliminating the "seam" visible
            with LINE_8 joins.
        """
        pts = StrokeInterpolator.interpolated_points(p1, p2, step_px)
        cap_r = max(1, thickness // 2)   # radius for cap circles

        # ── Pass 1: anti-aliased line segments ────────────────
        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i + 1], color, thickness,
                     lineType=cv2.LINE_AA)

        # ── Pass 2: round caps at every sub-point ─────────────
        if fill_caps:
            for pt in pts:
                cv2.circle(img, pt, cap_r, color, -1,
                           lineType=cv2.LINE_AA)

        return int(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))


# ══════════════════════════════════════════════════════════════════
#  CANVAS MANAGER
# ══════════════════════════════════════════════════════════════════
class CanvasManager:
    """BGR drawing surface with undo/redo and file I/O."""

    def __init__(self, h: int, w: int) -> None:
        self._h, self._w = h, w
        self._img  = np.zeros((h, w, 3), dtype=np.uint8)
        cfg = CFG.canvas
        self._undo: Deque[np.ndarray] = collections.deque(maxlen=cfg.max_undo_steps)
        self._redo: Deque[np.ndarray] = collections.deque(maxlen=cfg.max_undo_steps)
        self._dirty = 0

    def _push_undo(self) -> None:
        self._undo.append(self._img.copy())
        self._redo.clear()
        self._dirty = 0

    def draw_stroke(
        self,
        p1:        Tuple[int, int],
        p2:        Tuple[int, int],
        color:     Tuple[int, int, int],
        thickness: int,
    ) -> None:
        """
        Render an interpolated, anti-aliased stroke from p1 → p2.

        Delegates to StrokeInterpolator.draw_stroke() which:
          • Breaks the segment into sub-steps ≤ interp_step_px apart
          • Draws LINE_AA segments between every consecutive sub-point
          • Draws filled circle caps at each sub-point (round joins)

        This replaces the previous single cv2.line() call and
        eliminates all three gap/quality problems:
          1. Fast-movement gaps   — step_px=1 means sub-points are
             never more than 1 px apart regardless of finger speed.
          2. Diagonal jaggedness  — LINE_AA + sub-pixel cap circles
             produce smooth, continuous coverage.
          3. Sharp-turn voids     — round caps fill the concave region
             between two segments meeting at an angle.

        The dirty-pixel counter uses Euclidean distance (unchanged)
        so undo snapshot frequency is unaffected.
        """
        dist = StrokeInterpolator.draw_stroke(
            self._img, p1, p2, color, thickness,
            step_px   = CFG.drawing.interp_step_px,
            fill_caps = CFG.drawing.interp_fill_caps,
        )
        # Dirty tracking: accumulate stroke length × thickness
        self._dirty += dist * thickness
        if self._dirty >= CFG.canvas.undo_snapshot_threshold:
            self._push_undo()

    def erase(self, center, radius: int) -> None:
        cv2.circle(self._img, center, radius, (0, 0, 0), -1)

    def clear(self) -> None:
        self._push_undo()
        self._img = np.zeros((self._h, self._w, 3), dtype=np.uint8)

    def commit_stroke(self) -> None:
        if self._dirty > 0:
            self._push_undo()

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

    def save(self, path: str) -> bool:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            ok = cv2.imwrite(path, self._img)
            if ok:
                log.info("Canvas saved → %s", path)
            return ok
        except Exception as e:
            log.error("Save failed: %s", e)
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
        except Exception as e:
            log.error("Load failed: %s", e)
            return False

    @property
    def image(self) -> np.ndarray:
        return self._img

    @property
    def undo_depth(self) -> int:
        return len(self._undo)


# ══════════════════════════════════════════════════════════════════
#  UI RENDERER
# ══════════════════════════════════════════════════════════════════
class UIRenderer:
    _GESTURE_META = {
        GestureState.IDLE:    ("IDLE",       (130, 130, 130)),
        GestureState.DRAWING: ("DRAW  ☝",    (80,  220, 80 )),
        GestureState.SPACE:   ("SPACE ✌",    (220, 220, 80 )),
        GestureState.ERASING: ("ERASE ✋",   (80,  80,  220)),
        GestureState.FIST:    ("CLEAR ✊",   (80,  160, 255)),
    }

    _HELP = [
        ("☝",    "Draw"),
        ("✌",    "Lift pen"),
        ("✋",    "Erase"),
        ("✊",    "Clear (hold)"),
        ("r",    "Hybrid OCR + Solve"),
        ("d",    "Toggle OCR debug"),
        ("p",    "Toggle preprocess preview"),   # NEW
        ("b",    "Benchmark all engines"),
        ("s",    "Save drawing"),
        ("l",    "Load autosave"),
        ("z",    "Undo"),
        ("y",    "Redo"),
        ("+/-",  "Brush size"),
        ("1-8",  "Colour"),
        ("h",    "Help toggle"),
        ("q",    "Quit"),
    ]

    def __init__(self) -> None:
        self._show_help   = False
        self._notif_text  = ""
        self._notif_timer = 0
        self._fps_buf: Deque[float] = collections.deque(
            maxlen=CFG.ui.fps_window
        )
        self._t_prev = time.perf_counter()

        # Last OCR result — shown in a persistent HUD band
        self._last_ocr: Optional[OCRResult] = None

        # NEW: OCR-busy flag — set True while async thread is running
        # so the render loop can draw a "Processing OCR…" overlay
        # without the main loop blocking on the inference call.
        self.ocr_running: bool = False

    def tick(self) -> float:
        now = time.perf_counter()
        dt  = now - self._t_prev
        self._t_prev = now
        if dt > 1e-6:
            self._fps_buf.append(1.0 / dt)
        return float(np.mean(self._fps_buf)) if self._fps_buf else 0.0

    def notify(self, msg: str, dur: int = None) -> None:
        self._notif_text  = msg
        self._notif_timer = dur or CFG.ui.notif_duration

    def set_last_ocr(self, result: OCRResult) -> None:
        self._last_ocr = result

    def toggle_help(self) -> None:
        self._show_help = not self._show_help

    # ── helpers ────────────────────────────────────────────────
    @staticmethod
    def _label(img, text, pos, color=(230, 230, 230),
               scale=0.6, thickness=1, bg=True):
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                       scale, thickness)
        x, y = pos
        if bg:
            cv2.rectangle(img, (x-4, y-th-4), (x+tw+4, y+bl+2),
                          (15, 15, 15), -1)
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thickness, cv2.LINE_AA)

    @staticmethod
    def _draw_palette(img, active_idx: int) -> None:
        W   = img.shape[1]
        box = CFG.ui.palette_box
        gap = CFG.ui.palette_gap
        n   = len(CFG.drawing.palette)
        sx  = W - n * (box + gap) + gap - 12
        sy  = 10
        for i, color in enumerate(CFG.drawing.palette):
            x = sx + i * (box + gap)
            cv2.rectangle(img, (x, sy), (x+box, sy+box), color, -1)
            border = (255, 255, 255) if i == active_idx else (55, 55, 55)
            cv2.rectangle(img, (x-2, sy-2), (x+box+2, sy+box+2), border, 2)
            cv2.putText(img, str(i+1), (x+4, sy+box-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 0, 0), 1)

    def _draw_ocr_band(self, img) -> None:
        """Persistent bottom band showing last OCR result."""
        if self._last_ocr is None:
            return
        H, W = img.shape[:2]
        by = H - 28
        cv2.rectangle(img, (0, by), (W, H), (18, 18, 18), -1)

        # Engine badge
        eng   = self._last_ocr.engine_used
        conf  = self._last_ocr.confidence
        badge_col = (80, 220, 80) if conf >= CFG.hybrid.accept_threshold \
                    else (80, 80, 200)
        badge = f"[{eng}  {conf:.2f}]"
        (bw, _), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.putText(img, badge, (6, H-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, badge_col, 1, cv2.LINE_AA)

        # Result text
        display = self._last_ocr.display_text
        max_chars = (W - bw - 20) // 9
        if len(display) > max_chars:
            display = display[:max_chars] + "…"
        cv2.putText(img, display, (bw + 14, H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1, cv2.LINE_AA)

    # ── main render ────────────────────────────────────────────
    def render(self, frame, fps, gesture, color_idx, brush, undo_depth):
        out = frame
        H, W = out.shape[:2]

        # FPS + gesture
        self._label(out, f"FPS  {fps:5.1f}", (10, 28), (80, 230, 80))
        glabel, gcol = self._GESTURE_META.get(gesture, ("?", (200,200,200)))
        self._label(out, f"{glabel}", (10, 58), gcol)
        self._label(out, f"Brush {brush}px  Undo {undo_depth}", (10, 88))

        # Palette
        self._draw_palette(out, color_idx)

        # Notification toast
        if self._notif_timer > 0:
            alpha = min(1.0, self._notif_timer / 30.0)
            c = (int(0*alpha), int(220*alpha), int(220*alpha))
            (tw, _), _ = cv2.getTextSize(
                self._notif_text, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2
            )
            nx = max(0, (W - tw) // 2)
            self._label(out, self._notif_text, (nx, H - 60),
                        c, scale=0.72, thickness=2)
            self._notif_timer -= 1

        # ── NEW: OCR loading overlay ───────────────────────────
        # Displayed every frame while the async OCR thread is alive.
        # A pulsing dot-counter (. / .. / ...) gives visual feedback
        # that inference is running without freezing the main loop.
        if self.ocr_running:
            dots = "." * (int(time.time() * 2) % 3 + 1)   # cycles 1-2-3 Hz
            msg  = f"Processing OCR{dots}"
            (tw2, th2), _ = cv2.getTextSize(
                msg, cv2.FONT_HERSHEY_SIMPLEX, 0.78, 2
            )
            bx = (W - tw2) // 2 - 10
            by = H // 2 - 26
            # Semi-transparent dark panel behind text
            overlay = out.copy()
            cv2.rectangle(overlay,
                          (bx - 8, by - 4), (bx + tw2 + 18, by + th2 + 10),
                          (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.72, out, 0.28, 0, out)
            cv2.putText(out, msg, (bx, by + th2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.78,
                        (80, 220, 255), 2, cv2.LINE_AA)

        # OCR result band
        self._draw_ocr_band(out)

        # Help overlay
        if self._show_help:
            lh, pw = 22, 245
            ph = len(self._HELP) * lh + 32
            px, py = W - pw - 10, 50
            ov = out.copy()
            cv2.rectangle(ov, (px, py), (px+pw, py+ph), (20, 20, 20), -1)
            cv2.addWeighted(ov, CFG.ui.ui_alpha, out, 1-CFG.ui.ui_alpha, 0, out)
            cv2.rectangle(out, (px, py), (px+pw, py+ph), (70, 70, 70), 1)
            cv2.putText(out, "Shortcuts", (px+8, py+18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                        (180, 220, 255), 1, cv2.LINE_AA)
            for j, (key, desc) in enumerate(self._HELP):
                ty = py + 32 + j * lh
                cv2.putText(out, f"[{key}]", (px+8, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                            (220, 200, 80), 1, cv2.LINE_AA)
                cv2.putText(out, desc, (px+78, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                            (210, 210, 210), 1, cv2.LINE_AA)
        return out


# ══════════════════════════════════════════════════════════════════
#  WEBCAM MANAGER
# ══════════════════════════════════════════════════════════════════
class WebcamManager:
    def __init__(self) -> None:
        self._cap: Optional[cv2.VideoCapture] = None
        self._h = CFG.camera.height
        self._w = CFG.camera.width

    def open(self, retries: int = 3) -> bool:
        backend = cv2.CAP_DSHOW if CFG.camera.use_dshow else cv2.CAP_ANY
        for attempt in range(1, retries + 1):
            cap = cv2.VideoCapture(CFG.camera.index, backend)
            if not cap.isOpened():
                log.warning("Camera attempt %d/%d failed.", attempt, retries)
                time.sleep(0.5)
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CFG.camera.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CFG.camera.height)
            cap.set(cv2.CAP_PROP_FPS,          CFG.camera.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE,   CFG.camera.buffer_size)
            self._w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps_actual = cap.get(cv2.CAP_PROP_FPS)
            log.info("Camera  [%dx%d @ %.0f FPS]", self._w, self._h, fps_actual)
            self._cap = cap
            return True
        log.critical("Cannot open camera after %d attempts.", retries)
        return False

    def read(self):
        if self._cap and self._cap.isOpened():
            return self._cap.read()
        return False, None

    def release(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None

    @property
    def frame_size(self) -> Tuple[int, int]:
        return self._h, self._w


# ══════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════
class AIWhiteboardApp:
    """
    Orchestrates all subsystems in a clean 8-step per-frame pipeline:
    read → flip → MediaPipe → classify → debounce → draw/erase
    → composite → HUD → display → keyboard
    """

    def __init__(self) -> None:
        log.info("AI Air Whiteboard v5.0  —  Hybrid OCR Edition")

        # Subsystems
        self._cam    = WebcamManager()
        self._ocr    = HybridOCRPipeline()          # NEW: hybrid pipeline
        self._ui     = UIRenderer()
        self._canvas: Optional[CanvasManager] = None

        # MediaPipe
        mp_h = mp.solutions.hands
        self._mp_hands = mp_h
        self._mp_draw  = mp.solutions.drawing_utils
        self._lm_spec  = self._mp_draw.DrawingSpec(
            color=(80, 22, 10), thickness=2, circle_radius=3)
        self._cn_spec  = self._mp_draw.DrawingSpec(
            color=(80, 44, 121), thickness=2, circle_radius=2)
        self._hands = mp_h.Hands(
            static_image_mode=False,
            max_num_hands=CFG.mediapipe.max_hands,
            model_complexity=CFG.mediapipe.model_complexity,
            min_detection_confidence=CFG.mediapipe.detection_confidence,
            min_tracking_confidence=CFG.mediapipe.tracking_confidence,
        )

        # Per-hand state
        n = CFG.mediapipe.max_hands
        # FingertipTracker replaces the old CoordinateSmoother:
        # each hand gets its own Kalman filter + EMA + deadzone instance.
        self._trackers:     List[FingertipTracker]   = [FingertipTracker()   for _ in range(n)]
        self._debouncers:   List[GestureDebouncer]   = [GestureDebouncer()   for _ in range(n)]
        self._prev_pts:     List[Optional[Tuple]]    = [None] * n
        self._was_drawing:  List[bool]               = [False] * n

        # App state
        self._color_idx = 0
        self._brush     = CFG.drawing.brush_thickness
        self._comp_mask: Optional[np.ndarray] = None

    # ──────────────────────────────────────────────────────────
    def _init(self) -> bool:
        if not self._cam.open():
            return False
        h, w = self._cam.frame_size
        self._canvas    = CanvasManager(h, w)
        self._comp_mask = np.zeros((h, w), dtype=np.uint8)

        # Restore autosave
        autosave = os.path.join(CFG.canvas.save_dir, CFG.canvas.autosave_file)
        if os.path.isfile(autosave):
            self._canvas.load(autosave)
            self._ui.notify("📂  Autosave restored")
        return True

    # ──────────────────────────────────────────────────────────
    def _process_hand(self, hi: int, lm, frame, H: int, W: int) -> GestureState:
        raw              = GestureClassifier.classify(lm)
        stable, fist_now = self._debouncers[hi].update(raw)

        ix = int(lm[8].x * W);  iy = int(lm[8].y * H)   # index tip
        px = int(lm[9].x * W);  py = int(lm[9].y * H)   # palm centre

        tracker = self._trackers[hi]    # FingertipTracker for this hand
        canvas  = self._canvas

        if fist_now:
            tracker.reset()
            self._prev_pts[hi]    = None
            self._was_drawing[hi] = False
            canvas.clear()
            self._ui.notify("🧹  Board cleared")
            return GestureState.FIST

        if stable == GestureState.DRAWING:
            # ── Stabilise raw landmark through Kalman → EMA → deadzone ──
            # tracker.update() returns the final stabilised (x, y):
            #   1. Kalman filter predicts position from previous velocity
            #      then corrects with the new MediaPipe measurement.
            #   2. EMA post-filter smooths residual low-freq oscillations.
            #   3. Deadzone gate freezes cursor when displacement < 2.5 px
            #      so stationary tremor does not produce spurious ink.
            sx, sy = tracker.update(ix, iy)
            prev   = self._prev_pts[hi]

            if prev is not None:
                dist = np.hypot(sx - prev[0], sy - prev[1])
                if dist >= CFG.drawing.min_draw_dist:
                    # Interpolated stroke: 1 px sub-steps + round caps
                    canvas.draw_stroke(
                        prev, (sx, sy),
                        CFG.drawing.palette[self._color_idx],
                        self._brush,
                    )
            self._prev_pts[hi]    = (sx, sy)
            self._was_drawing[hi] = True
            # Visual cursor at the stabilised position
            cv2.circle(frame, (sx, sy),
                       max(4, self._brush // 2),
                       CFG.drawing.palette[self._color_idx], -1)

        elif stable == GestureState.ERASING:
            if self._was_drawing[hi]:
                canvas.commit_stroke()
                self._was_drawing[hi] = False
            self._prev_pts[hi] = None
            tracker.reset()   # clear Kalman state — eraser uses palm, not tip
            canvas.erase((px, py), CFG.drawing.eraser_radius)
            r = CFG.drawing.eraser_radius
            cv2.circle(frame, (px, py), r, (60, 60, 220), 2)
            cv2.circle(frame, (px, py), 4, (60, 60, 220), -1)

        else:
            if self._was_drawing[hi]:
                canvas.commit_stroke()
                self._was_drawing[hi] = False
            self._prev_pts[hi] = None
            tracker.reset()   # clear velocity so next stroke starts clean

        return stable

    # ──────────────────────────────────────────────────────────
    def _composite(self, frame: np.ndarray) -> None:
        """Blend ink onto camera frame (ink-only pixels, not full frame)."""
        ink = self._canvas.image
        cv2.cvtColor(ink, cv2.COLOR_BGR2GRAY, dst=self._comp_mask)
        cv2.threshold(self._comp_mask, 1, 255, cv2.THRESH_BINARY,
                      dst=self._comp_mask)
        region = self._comp_mask > 0
        frame[region] = cv2.addWeighted(
            frame, 1.0 - CFG.drawing.canvas_ink_weight,
            ink,   CFG.drawing.canvas_ink_weight, 0,
        )[region]

    # ──────────────────────────────────────────────────────────
    def _do_ocr(self) -> None:
        """
        Launch the Hybrid OCR pipeline on a background thread so the
        main render loop keeps running at full FPS during inference.

        Thread safety
        -------------
        * `self._ui.ocr_running` is set/cleared from the worker thread.
          It is a plain bool — Python's GIL makes single-assignment
          reads/writes atomic, so no Lock is needed here.
        * `self._canvas.image` is read once at call time and passed as
          a snapshot (`canvas_snapshot`) so drawing can continue without
          the OCR thread seeing a partially-modified canvas.
        * `self._ui.notify` and `self._ui.set_last_ocr` each perform a
          single attribute assignment — also GIL-safe.
        """
        # Guard: don't stack multiple OCR threads
        if self._ui.ocr_running:
            self._ui.notify("⏳  OCR already running…")
            return

        # Take an immutable snapshot of the canvas at trigger time
        canvas_snapshot: np.ndarray = self._canvas.image.copy()
        log.info("OCR triggered — spawning background thread.")

        def _worker() -> None:
            self._ui.ocr_running = True
            try:
                result = self._ocr.run(canvas_snapshot)
                self._ui.set_last_ocr(result)
                if result.text:
                    self._ui.notify(result.display_text, dur=180)
                    log.info(
                        "OCR done  engine=%-10s  conf=%.3f  latency=%.0f ms",
                        result.engine_used, result.confidence, result.latency_ms,
                    )
                else:
                    self._ui.notify("⚠  No text detected")
            except Exception as exc:
                log.error("OCR worker exception: %s", exc)
                self._ui.notify("⚠  OCR error — see log")
            finally:
                # Always clear the flag so the UI stops showing the spinner
                self._ui.ocr_running = False

        # daemon=True: thread auto-dies if the app exits before it finishes
        t = threading.Thread(target=_worker, daemon=True, name="ocr-worker")
        t.start()

    # ──────────────────────────────────────────────────────────
    def _do_benchmark(self) -> None:
        """Quick benchmark of all three engines and print report."""
        from utils.benchmark import OCRBenchmark, BenchmarkReport
        log.info("Running OCR benchmark …")
        self._ui.notify("⏱  Benchmarking…", dur=40)

        bench = OCRBenchmark()
        bench.register("tesseract",
                       lambda i: self._ocr.tesseract.recognize(i, self._ocr.preprocessor))
        bench.register("easyocr",
                       lambda i: self._ocr.easyocr.recognize(i, self._ocr.preprocessor))
        bench.register("trocr",
                       lambda i: self._ocr.trocr.recognize(i, self._ocr.preprocessor))

        report = bench.run_all(self._canvas.image, runs=2)
        OCRBenchmark.print_report(report)
        best = report.best()
        if best:
            self._ui.notify(
                f"⚡ Best: {best.name}  conf={best.confidence:.2f}  "
                f"{best.latency_ms:.0f}ms",
                dur=180,
            )

    # ──────────────────────────────────────────────────────────
    def _handle_key(self, key: int) -> bool:
        if key == ord("q"):
            return False

        elif key == ord("r"):
            self._do_ocr()

        elif key == ord("d"):
            # Toggle debug visualisation in hybrid pipeline
            import config as _c
            # Frozen dataclass workaround — flip debug flag at runtime
            current = CFG.hybrid.debug_visualize
            object.__setattr__(CFG.hybrid, "debug_visualize", not current)
            state = "ON" if not current else "OFF"
            self._ui.notify(f"OCR Debug  {state}")
            log.info("OCR debug visualisation: %s", state)

        elif key == ord("p"):
            # Toggle the preprocessed binary image preview window.
            # Shows exactly what Tesseract / TrOCR see — invaluable for
            # diagnosing why OCR misreads a particular character.
            current_p = CFG.hybrid.show_preprocessed
            object.__setattr__(CFG.hybrid, "show_preprocessed", not current_p)
            state_p = "ON" if not current_p else "OFF"
            self._ui.notify(f"Preprocess preview  {state_p}")
            log.info("Preprocessed preview window: %s", state_p)
            # If turning off, close the window immediately
            if current_p:
                try:
                    cv2.destroyWindow("OCR Preprocessed  [p to close]")
                except Exception:
                    pass

        elif key == ord("b"):
            self._do_benchmark()

        elif key == ord("s"):
            ts   = int(time.time())
            path = os.path.join(CFG.canvas.save_dir, f"drawing_{ts}.png")
            if self._canvas.save(path):
                self._ui.notify(f"💾  Saved  {os.path.basename(path)}")
            else:
                self._ui.notify("⚠  Save failed")

        elif key == ord("l"):
            path = os.path.join(CFG.canvas.save_dir, CFG.canvas.autosave_file)
            if self._canvas.load(path):
                self._ui.notify("📂  Loaded autosave")
            else:
                self._ui.notify("⚠  No autosave found")

        elif key == ord("z"):
            self._ui.notify("↩  Undo" if self._canvas.undo() else "⚠  Nothing to undo")

        elif key == ord("y"):
            self._ui.notify("↪  Redo" if self._canvas.redo() else "⚠  Nothing to redo")

        elif key == ord("h"):
            self.toggle_help()

        elif key in (ord("+"), ord("=")):
            self._brush = min(self._brush + CFG.drawing.brush_step, CFG.drawing.brush_max)
            self._ui.notify(f"Brush  {self._brush} px")

        elif key == ord("-"):
            self._brush = max(self._brush - CFG.drawing.brush_step, CFG.drawing.brush_min)
            self._ui.notify(f"Brush  {self._brush} px")

        elif ord("1") <= key <= ord("8"):
            idx = key - ord("1")
            if idx < len(CFG.drawing.palette):
                self._color_idx = idx
                self._ui.notify(f"Colour  {idx + 1}")

        return True

    def toggle_help(self):
        self._ui.toggle_help()

    # ──────────────────────────────────────────────────────────
    def run(self) -> None:
        if not self._init():
            log.critical("Init failed — exiting.")
            return

        H, W = self._canvas.undo_depth, 0  # placeholder until frame arrives
        log.info("Running. Press [h] for help, [q] to quit.")

        try:
            while True:
                # 1. Capture
                ok, frame = self._cam.read()
                if not ok or frame is None:
                    log.warning("Frame drop.")
                    continue
                frame = cv2.flip(frame, 1)
                H, W  = frame.shape[:2]

                # 2. MediaPipe  (writeable=False skips internal copy)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                mp_result = self._hands.process(rgb)
                rgb.flags.writeable = True

                # 3. Per-hand processing
                dominant = GestureState.IDLE
                if mp_result.multi_hand_landmarks:
                    for hi, hand_lm in enumerate(mp_result.multi_hand_landmarks):
                        if hi >= CFG.mediapipe.max_hands:
                            break
                        self._mp_draw.draw_landmarks(
                            frame, hand_lm,
                            self._mp_hands.HAND_CONNECTIONS,
                            self._lm_spec, self._cn_spec,
                        )
                        g = self._process_hand(hi, hand_lm.landmark, frame, H, W)
                        if g not in (GestureState.IDLE, GestureState.SPACE):
                            dominant = g
                else:
                    # No hands detected — commit any open stroke and reset
                    # each tracker so residual Kalman velocity does not
                    # carry over into the next stroke.
                    for hi in range(CFG.mediapipe.max_hands):
                        if self._was_drawing[hi]:
                            self._canvas.commit_stroke()
                            self._was_drawing[hi] = False
                        self._trackers[hi].reset()
                        self._prev_pts[hi] = None

                # 4. Composite ink
                self._composite(frame)

                # 5. HUD
                fps    = self._ui.tick()
                output = self._ui.render(
                    frame, fps, dominant,
                    self._color_idx, self._brush,
                    self._canvas.undo_depth,
                )

                # 6. Display
                cv2.imshow("AI Air Whiteboard  v5.0", output)

                # 7. Keyboard
                key = cv2.waitKey(1) & 0xFF
                if key != 0xFF and not self._handle_key(key):
                    break

        except KeyboardInterrupt:
            log.info("Ctrl-C received.")
        finally:
            self._shutdown()

    # ──────────────────────────────────────────────────────────
    def _shutdown(self) -> None:
        log.info("Shutting down …")
        if self._canvas is not None:
            path = os.path.join(CFG.canvas.save_dir, CFG.canvas.autosave_file)
            self._canvas.save(path)
        self._hands.close()
        self._cam.release()
        cv2.destroyAllWindows()
        log.info("✅  Clean shutdown complete.")


# ══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    AIWhiteboardApp().run()
