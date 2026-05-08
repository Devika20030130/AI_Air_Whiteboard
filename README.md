# ✋ AI Air Whiteboard — v5.0  (Hybrid OCR Edition)

> Production-grade gesture whiteboard with a **three-engine hybrid OCR pipeline**:
> **TrOCR** (transformer) → **EasyOCR** → **Tesseract** — with automatic
> confidence-gated waterfall fallback and **SymPy** equation solving.

---

## 🏗️ Project Structure

```
ai-air-whiteboard/
│
├── main.py                  ← Application entry point
├── config.py                ← All configuration constants (single source of truth)
├── requirements.txt         ← Pinned dependencies
├── README.md
│
├── ocr/
│   ├── __init__.py
│   ├── preprocess.py        ← 8-stage OpenCV preprocessing pipeline
│   ├── trocr_engine.py      ← Microsoft TrOCR transformer engine
│   ├── easyocr_engine.py    ← EasyOCR CRAFT+CRNN engine
│   ├── tesseract_engine.py  ← Tesseract baseline engine
│   ├── equation_parser.py   ← SymPy sanitiser + solver
│   └── hybrid_ocr.py        ← Waterfall orchestrator
│
├── utils/
│   ├── __init__.py
│   ├── logger.py            ← Coloured console + rotating file logger
│   ├── image_utils.py       ← Shared OpenCV / NumPy helpers
│   └── benchmark.py         ← Per-engine timing + accuracy report
│
├── drawings/                ← Auto-created; saves + autosave.png
└── logs/                    ← Auto-created; whiteboard.log
```

---

## ⚡ OCR Pipeline Architecture

```
Canvas BGR image
       │
       ▼
┌─────────────────────────────────────────┐
│         EquationPreprocessor            │
│  crop → upscale → adaptiveThreshold     │
│  → morphClose → contourClean → denoise  │
└──────────────┬──────────────────────────┘
               │ binary + PIL image
       ┌───────▼────────────────────────────────────┐
       │         HybridOCRPipeline  (waterfall)      │
       │                                             │
       │  1. TrOCREngine   (transformer, beam×4)     │
       │        ↓  if conf < 0.55                    │
       │  2. EasyOCREngine (CRAFT + CRNN, GPU)       │
       │        ↓  if conf < 0.55                    │
       │  3. TesseractEngine (LSTM, PSM-6)           │
       │                                             │
       │  → best result → EquationParser → SymPy    │
       └─────────────────────────────────────────────┘
```

---

## 🚀 Installation (PowerShell step-by-step)

### 1 — Navigate to your project

```powershell
cd D:\AI_Projects\Air_WritingBoard
```

### 2 — Activate your existing venv

```powershell
.\venv\Scripts\Activate.ps1
```

### 3 — Copy new files into place

```powershell
# Copy the files delivered above into the project root
# config.py, main.py, requirements.txt, README.md

# Create package folders
New-Item -ItemType Directory -Force -Path ocr, utils, drawings, logs

# Create __init__.py files
New-Item -ItemType File -Force -Path ocr\__init__.py
New-Item -ItemType File -Force -Path utils\__init__.py
```

### 4 — Install Tesseract binary (Windows)

Download and run the installer from:
https://github.com/UB-Mannheim/tesseract/wiki

Then set the environment variable (replace path if different):

```powershell
$env:TESSERACT_PATH = "C:\Program Files\Tesseract-OCR\tesseract.exe"
# To persist across sessions:
[System.Environment]::SetEnvironmentVariable(
    "TESSERACT_PATH",
    "C:\Program Files\Tesseract-OCR\tesseract.exe",
    "User"
)
```

### 5 — Install Python dependencies

**CPU only (works on any machine):**

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

**GPU (CUDA 12.1 — recommended for TrOCR speed):**

```powershell
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Verify GPU is detected:

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

### 6 — First run (TrOCR downloads ~340 MB on first OCR trigger)

```powershell
python main.py
```

---

## 🎮 Controls

### Gestures

| Gesture | Action |
|---|---|
| ☝️ Index finger only | **Draw** |
| ✌️ Index + Middle | **Lift pen** |
| ✋ All 4 fingers | **Erase** |
| ✊ Closed fist (hold ~1 s) | **Clear board** |

### Keyboard

| Key | Action |
|---|---|
| `r` | Run **Hybrid OCR** + solve equation |
| `d` | Toggle **OCR debug** visualisation window |
| `b` | Run **benchmark** — compare all 3 engines |
| `s` | Save drawing (timestamped PNG) |
| `l` | Load autosave |
| `z` | Undo |
| `y` | Redo |
| `+` / `-` | Brush size |
| `1` – `8` | Select colour |
| `h` | Toggle help overlay |
| `q` | Quit + autosave |

---

## ⚙️ Configuration

All settings live in `config.py`.  Key options:

```python
# Switch engine order (e.g. skip TrOCR for speed)
hybrid.engine_priority = ("easyocr", "tesseract")

# Lower threshold = accept lower-confidence results faster
hybrid.accept_threshold = 0.40

# Enable debug window (shows preprocessing stages + per-engine results)
hybrid.debug_visualize = True

# Force CPU even if CUDA is available
# Set before importing config:  DEVICE = "cpu"

# TrOCR model size: base (~340 MB) or large (~1.4 GB)
trocr.model_name = "microsoft/trocr-large-handwritten"
```

---

## 🔬 Benchmark Mode

Press **`b`** in the app, or run standalone:

```powershell
python -m utils.benchmark --image drawings\autosave.png --runs 3
```

Sample output:

```
────────────────────────────────────────────────────────────────
  OCR BENCHMARK REPORT   (runs per engine: 3)
  Image: drawings\autosave.png
────────────────────────────────────────────────────────────────
  Engine         Text                       Conf       ms  Status
────────────────────────────────────────────────────────────────
  trocr          '2x + 5 = 11'            0.821    842.3  ✓
  easyocr        '2x+5=11'                0.743    312.1  ✓
  tesseract      '2x + 5 = 11'            0.680    124.7  ✓
────────────────────────────────────────────────────────────────
  🏆  Best confidence : trocr  (0.821)
  ⚡  Fastest         : tesseract  (124.7 ms)
────────────────────────────────────────────────────────────────
```

---

## 🐛 Troubleshooting

| Problem | Fix |
|---|---|
| `TesseractNotFoundError` | Set `TESSERACT_PATH` env var to the `.exe` path |
| `CUDA out of memory` | Set `trocr.device = "cpu"` in `config.py` |
| TrOCR slow first run | Normal — model downloads ~340 MB once, then caches |
| EasyOCR import error | `pip install easyocr` |
| Camera not opening | Change `camera.index = 1` in `config.py` |
| Low FPS | Set `mediapipe.model_complexity = 0` (already default) |
| Black screen | Check that `CAP_DSHOW` is correct for your OS (set `use_dshow = False` on Linux/macOS) |

---

## 📊 Engine Comparison

| Engine | Best for | Speed | Accuracy |
|---|---|---|---|
| **TrOCR** | Cursive, mixed-case, complex math | ~800 ms | ⭐⭐⭐⭐⭐ |
| **EasyOCR** | Printed + semi-cursive, multi-line | ~300 ms | ⭐⭐⭐⭐ |
| **Tesseract** | Neat block handwriting | ~120 ms | ⭐⭐⭐ |

---

## 🔭 Roadmap

- [ ] LaTeX rendering of solved equations via MathJax overlay
- [ ] Speech-to-equation fallback (`whisper` integration)
- [ ] FastAPI `/ocr` endpoint for browser-based demo
- [ ] ONNX export of TrOCR for 3× faster CPU inference
- [ ] Unit tests for `GestureClassifier`, `EquationParser`, `CoordinateSmoother`
- [ ] Docker image with CUDA support

---

## 📄 Author

Devika Das

---

*Built with MediaPipe · OpenCV · TrOCR · EasyOCR · Tesseract · SymPy*