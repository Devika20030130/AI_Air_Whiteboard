# ✋ AI Air Whiteboard — v4.0

> A real-time, gesture-driven virtual whiteboard powered by **MediaPipe Hands**,
> **OpenCV**, **Tesseract OCR** and **SymPy** — refactored to production quality.

---

## ✨ Features

| Category | Details |
|---|---|
| **Gesture Drawing** | Index-finger-only stroke with EMA smoothing |
| **Lift Pen** | Index + middle (Space mode) — no accidental marks |
| **Eraser** | Full open palm — configurable radius |
| **Clear Board** | Closed fist held for ~22 frames — debounced to prevent accidents |
| **Multi-hand** | Up to 2 independent hands tracked simultaneously |
| **Colour Palette** | 8 colours, selected by keyboard `1`–`8` |
| **Brush Size** | Adjustable live with `+` / `-` |
| **Undo / Redo** | Copy-on-write stack — up to 40 steps |
| **Save / Load** | Timestamped PNG saves + autosave on quit |
| **OCR + Solve** | Adaptive-threshold pipeline → Tesseract → SymPy |
| **FPS Counter** | 30-frame rolling average displayed on HUD |
| **Help Overlay** | Press `h` to toggle keyboard shortcut panel |

---

## 🏗️ Architecture

```
AIWhiteboardApp          ← top-level orchestrator / main loop
├── WebcamManager        ← low-latency capture (1-frame buffer)
├── mp.solutions.Hands   ← MediaPipe hand landmark detector
├── GestureClassifier    ← stateless landmark → GestureState
├── GestureDebouncer     ← per-hand state machine (confirm + cooldown)
├── CoordinateSmoother   ← per-hand EMA fingertip smoother
├── CanvasManager        ← BGR drawing surface + undo/redo/save/load
├── OCRProcessor         ← adaptive threshold → Tesseract → SymPy
└── UIRenderer           ← FPS, HUD, palette, notifications, help panel
```

Every subsystem is a self-contained class with no shared mutable globals,
making each one independently testable and replaceable.

---

## ⚡ Optimisations (v3 → v4)

### Performance
| Optimisation | Impact |
|---|---|
| `CAP_PROP_BUFFERSIZE = 1` | Cuts camera latency by ~2 frames |
| `rgb.flags.writeable = False` before `hands.process()` | Skips MediaPipe's internal copy |
| `MP_MODEL_COMPLEXITY = 0` | Fastest model; accuracy still excellent for gestures |
| Ink-only compositing (mask-based) | Avoids full-frame `addWeighted` every tick |
| EMA weights pre-computed & cached | Removes repeated `np.array` allocation per frame |
| Morph kernel allocated once | Removes per-OCR-call `getStructuringElement` |

### Gesture Quality
| Optimisation | Impact |
|---|---|
| `GestureDebouncer` — N-frame confirmation window | Eliminates single-frame flickers |
| Independent cooldown after every transition | Prevents rapid state oscillation |
| FIST requires 22-frame hold + one-shot trigger | Prevents accidental board clears |
| Eraser requires **all 4** fingers up | Resolves eraser/space gesture conflict |
| Priority order: FIST > ERASE > DRAW > SPACE | Unambiguous classification |
| `GestureClassifier` is stateless | Easy to unit-test; no hidden dependencies |

### Drawing Quality
| Optimisation | Impact |
|---|---|
| Exponential Moving Average smoother | Removes high-frequency jitter |
| `MIN_DRAW_DIST` threshold | Skips `draw_line` for micro-movements |
| `cv2.LINE_AA` anti-aliased strokes | Smooth diagonal lines |
| `commit_stroke()` on pen-lift | Correct undo boundaries per stroke |
| Dirty-pixel counter triggers undo snapshot | Fewer snapshots; less memory |

### OCR Accuracy
| Optimisation | Impact |
|---|---|
| 2.5× upscale with `INTER_CUBIC` | Matches ~300 DPI Tesseract sweet-spot |
| `adaptiveThreshold` (Gaussian, 21-block) | Handles uneven brush opacity |
| Morphological close × 2 iterations | Bridges micro-gaps in strokes |
| `fastNlMeansDenoising` | Removes speckle noise before OCR |
| Character whitelist in Tesseract config | Discards non-equation glyphs |
| PSM 6 + OEM 3 (LSTM) | Best Tesseract mode for block handwriting |

---

## 🖥️ System Requirements

- Python **3.11+**
- **Tesseract OCR** binary installed (see below)
- Webcam (USB or built-in)
- OS: Windows 10/11 · macOS 12+ · Ubuntu 20.04+

---

## 🚀 Installation

### 1 — Clone

```bash
git clone https://github.com/yourname/ai-air-whiteboard.git
cd ai-air-whiteboard
```

### 2 — Python dependencies

```bash
pip install -r requirements.txt
```

### 3 — Tesseract binary

| Platform | Command |
|---|---|
| **Windows** | Download installer from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) |
| **macOS** | `brew install tesseract` |
| **Ubuntu / Debian** | `sudo apt install tesseract-ocr` |

If Tesseract is installed to a non-default path, set:

```bash
export TESSERACT_PATH="/custom/path/to/tesseract"   # Linux / macOS
set TESSERACT_PATH="C:\Program Files\Tesseract-OCR\tesseract.exe"  # Windows
```

### 4 — Run

```bash
python main.py
```

---

## 🎮 Controls

### Gestures

| Gesture | Action |
|---|---|
| ☝️ Index finger only | **Draw** |
| ✌️ Index + Middle | **Lift pen** (Space mode) |
| ✋ All 4 fingers extended | **Erase** |
| ✊ Closed fist (hold ~1 s) | **Clear entire board** |

### Keyboard

| Key | Action |
|---|---|
| `r` | Run OCR on canvas + solve equation |
| `s` | Save drawing (timestamped PNG) |
| `l` | Load last autosave |
| `z` | Undo |
| `y` | Redo |
| `+` / `-` | Increase / decrease brush size |
| `1` – `8` | Select colour from palette |
| `h` | Toggle keyboard shortcut overlay |
| `q` | Quit (autosaves canvas) |

---

## ⚙️ Configuration

All tuneable constants are at the top of `main.py` inside the `Config` class.
No need to touch any logic — just edit the values:

```python
class Config:
    CAMERA_INDEX      = 0          # 0 = default webcam
    CAMERA_WIDTH      = 1280
    CAMERA_HEIGHT     = 720
    BRUSH_THICKNESS   = 8          # px
    ERASER_RADIUS     = 45         # px
    SMOOTH_ALPHA      = 0.45       # EMA weight (0 = laggy, 1 = raw)
    FIST_CLEAR_FRAMES = 22         # frames fist must be held
    MAX_UNDO_STEPS    = 40
    OCR_SCALE         = 2.5        # upscale factor before OCR
    SAVE_DIR          = "drawings"
```

---

## 🗂️ Project Structure

```
ai-air-whiteboard/
├── main.py            ← single-file application (all 10 sections)
├── requirements.txt   ← pinned dependencies
├── README.md          ← this file
└── drawings/          ← auto-created; holds saves + autosave.png
```

---

## 🔭 Future Roadmap

- [ ] Multi-colour palette selection **by gesture** (pinch on swatch)
- [ ] `--source video.mp4` CLI flag for file-based demo
- [ ] FastAPI WebSocket stream for browser display
- [ ] ONNX / TFLite model swap for MediaPipe to reduce CPU
- [ ] GPU-accelerated canvas with CUDA OpenCV build
- [ ] Unit tests for `GestureClassifier` and `CoordinateSmoother`
- [ ] Export canvas as SVG (vectorise strokes)
- [ ] Speech-to-equation input as OCR fallback

---

*Built with ❤️ using MediaPipe · OpenCV · Tesseract · SymPy*