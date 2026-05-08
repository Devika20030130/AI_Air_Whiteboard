"""
utils/logger.py — Centralised, structured logging for AI Air Whiteboard.

Features
--------
* Coloured console output (Windows + ANSI terminals)
* Rotating file handler  → logs/whiteboard.log  (5 MB × 3 backups)
* One call to `get_logger(name)` from any module
* Performance timer context-manager
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import time
from contextlib import contextmanager
from typing import Generator


# ──────────────────────────────────────────────────────────────────
#  ANSI colour codes (gracefully degraded on Windows without ANSI)
# ──────────────────────────────────────────────────────────────────
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_COLOURS = {
    "DEBUG":    "\033[36m",    # Cyan
    "INFO":     "\033[32m",    # Green
    "WARNING":  "\033[33m",    # Yellow
    "ERROR":    "\033[31m",    # Red
    "CRITICAL": "\033[35m",    # Magenta
}


class _ColouredFormatter(logging.Formatter):
    """Console formatter with level-based ANSI colours."""

    FMT = "%(asctime)s  {colour}[%(levelname)-8s]{reset}  %(name)s  —  %(message)s"
    DATEFMT = "%H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        colour = _COLOURS.get(record.levelname, "")
        fmt = self.FMT.format(colour=colour, reset=_RESET)
        formatter = logging.Formatter(fmt, datefmt=self.DATEFMT)
        return formatter.format(record)


class _PlainFormatter(logging.Formatter):
    """Plain formatter for file output (no ANSI escapes)."""

    FMT = "%(asctime)s  [%(levelname)-8s]  %(name)s  —  %(message)s"
    DATEFMT = "%Y-%m-%d %H:%M:%S"

    def __init__(self) -> None:
        super().__init__(fmt=self.FMT, datefmt=self.DATEFMT)


# ──────────────────────────────────────────────────────────────────
#  GLOBAL SETUP  (runs once on first import)
# ──────────────────────────────────────────────────────────────────
_LOG_DIR  = "logs"
_LOG_FILE = os.path.join(_LOG_DIR, "whiteboard.log")
_ROOT     = "AIWhiteboard"
_initialised = False


def _setup_root() -> None:
    global _initialised
    if _initialised:
        return

    # Enable ANSI on Windows 10+ cmd / PowerShell
    if sys.platform == "win32":
        os.system("")   # triggers VT-100 mode

    os.makedirs(_LOG_DIR, exist_ok=True)

    root = logging.getLogger(_ROOT)
    root.setLevel(logging.DEBUG)

    # ── Console handler ──────────────────────────────────────
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(_ColouredFormatter())
    root.addHandler(ch)

    # ── Rotating file handler ────────────────────────────────
    fh = logging.handlers.RotatingFileHandler(
        _LOG_FILE,
        maxBytes=5 * 1024 * 1024,   # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(_PlainFormatter())
    root.addHandler(fh)

    root.propagate = False
    _initialised = True


def get_logger(name: str) -> logging.Logger:
    """
    Return a child logger under the AIWhiteboard root.

    Usage
    -----
    >>> from utils.logger import get_logger
    >>> log = get_logger(__name__)
    >>> log.info("Hello from %s", __name__)
    """
    _setup_root()
    return logging.getLogger(f"{_ROOT}.{name}")


# ──────────────────────────────────────────────────────────────────
#  PERFORMANCE TIMER
# ──────────────────────────────────────────────────────────────────
@contextmanager
def timer(label: str, logger: logging.Logger | None = None) -> Generator:
    """
    Context-manager that logs elapsed time in milliseconds.

    Usage
    -----
    >>> with timer("TrOCR inference", log):
    ...     result = model(image)
    """
    _setup_root()
    _log = logger or logging.getLogger(f"{_ROOT}.timer")
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        _log.debug("⏱  %-30s  %7.2f ms", label, elapsed_ms)
