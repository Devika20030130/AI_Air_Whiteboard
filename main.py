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
Author   : Devika Das
"""

from __future__ import annotations

# ── stdlib ────────────────────────────────────────────────────────
import collections
import os
import sys
import time
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
#  COORDINATE SMOOTHER  (EMA + fixed window)
# ══════════════════════════════════════════════════════════════════
class CoordinateSmoother:
    """Exponential Moving Average fingertip smoother."""

    def __init__(self) -> None:
        d = CFG.drawing
        self._hx: Deque[int] = collections.deque(maxlen=d.smooth_window)
        self._hy: Deque[int] = collections.deque(maxlen=d.smooth_window)
        self._alpha  = d.smooth_alpha
        self._cache: dict = {}

    def _weights(self, n: int) -> np.ndarray:
        if n not in self._cache:
            w = np.array([self._alpha ** (n - 1 - i) for i in range(n)],
                         dtype=np.float32)
            w /= w.sum()
            self._cache[n] = w
        return self._cache[n]

    def update(self, x: int, y: int) -> Tuple[int, int]:
        self._hx.append(x)
        self._hy.append(y)
        n = len(self._hx)
        w = self._weights(n)
        return int(np.dot(w, list(self._hx))), int(np.dot(w, list(self._hy)))

    def reset(self) -> None:
        self._hx.clear()
        self._hy.clear()


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

    def draw_line(self, p1, p2, color, thickness: int) -> None:
        cv2.line(self._img, p1, p2, color, thickness, lineType=cv2.LINE_AA)
        self._dirty += int(np.hypot(p2[0]-p1[0], p2[1]-p1[1])) * thickness
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
        self._smoothers:    List[CoordinateSmoother] = [CoordinateSmoother() for _ in range(n)]
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
        raw            = GestureClassifier.classify(lm)
        stable, fist_now = self._debouncers[hi].update(raw)

        ix = int(lm[8].x * W);  iy = int(lm[8].y * H)   # index tip
        px = int(lm[9].x * W);  py = int(lm[9].y * H)   # palm centre

        smoother = self._smoothers[hi]
        canvas   = self._canvas

        if fist_now:
            smoother.reset()
            self._prev_pts[hi]   = None
            self._was_drawing[hi] = False
            canvas.clear()
            self._ui.notify("🧹  Board cleared")
            return GestureState.FIST

        if stable == GestureState.DRAWING:
            sx, sy = smoother.update(ix, iy)
            prev   = self._prev_pts[hi]
            if prev is not None:
                dist = np.hypot(sx - prev[0], sy - prev[1])
                if dist >= CFG.drawing.min_draw_dist:
                    canvas.draw_line(
                        prev, (sx, sy),
                        CFG.drawing.palette[self._color_idx],
                        self._brush,
                    )
            self._prev_pts[hi]   = (sx, sy)
            self._was_drawing[hi] = True
            cv2.circle(frame, (sx, sy),
                       max(4, self._brush // 2),
                       CFG.drawing.palette[self._color_idx], -1)

        elif stable == GestureState.ERASING:
            if self._was_drawing[hi]:
                canvas.commit_stroke()
                self._was_drawing[hi] = False
            self._prev_pts[hi] = None
            smoother.reset()
            canvas.erase((px, py), CFG.drawing.eraser_radius)
            r = CFG.drawing.eraser_radius
            cv2.circle(frame, (px, py), r, (60, 60, 220), 2)
            cv2.circle(frame, (px, py), 4, (60, 60, 220), -1)

        else:
            if self._was_drawing[hi]:
                canvas.commit_stroke()
                self._was_drawing[hi] = False
            self._prev_pts[hi] = None
            smoother.reset()

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
        """Run hybrid OCR and update the UI with the result."""
        log.info("Running Hybrid OCR pipeline …")
        self._ui.notify("🔍  Running OCR…", dur=40)

        result = self._ocr.run(self._canvas.image)
        self._ui.set_last_ocr(result)

        if result.text:
            self._ui.notify(result.display_text, dur=180)
            log.info(
                "OCR done  engine=%-10s  conf=%.3f  latency=%.0f ms",
                result.engine_used, result.confidence, result.latency_ms,
            )
        else:
            self._ui.notify("⚠  No text detected")

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
                    for hi in range(CFG.mediapipe.max_hands):
                        if self._was_drawing[hi]:
                            self._canvas.commit_stroke()
                            self._was_drawing[hi] = False
                        self._smoothers[hi].reset()
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
