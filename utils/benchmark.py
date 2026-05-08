"""
utils/benchmark.py — OCR engine benchmarking and performance reporting.

Usage (standalone)
------------------
  python -m utils.benchmark --image drawings/autosave.png --runs 5

Usage (programmatic)
---------------------
>>> from utils.benchmark import OCRBenchmark
>>> bench = OCRBenchmark()
>>> bench.run_all(canvas_bgr)
>>> bench.print_report()
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from utils.logger import get_logger

log = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════
@dataclass
class EngineResult:
    name:       str
    text:       str
    confidence: float
    latency_ms: float
    success:    bool
    error:      str = ""


@dataclass
class BenchmarkReport:
    engine_results: List[EngineResult]   = field(default_factory=list)
    image_path:     Optional[str]        = None
    runs:           int                  = 1

    def best(self) -> Optional[EngineResult]:
        """Return the result with the highest confidence."""
        ok = [r for r in self.engine_results if r.success]
        return max(ok, key=lambda r: r.confidence) if ok else None

    def fastest(self) -> Optional[EngineResult]:
        ok = [r for r in self.engine_results if r.success]
        return min(ok, key=lambda r: r.latency_ms) if ok else None


# ══════════════════════════════════════════════════════════════════
#  BENCHMARK RUNNER
# ══════════════════════════════════════════════════════════════════
class OCRBenchmark:
    """
    Run every OCR engine against the same canvas image and collect
    latency + confidence metrics.
    """

    def __init__(self) -> None:
        self._engines: Dict[str, Callable] = {}
        self._reports: List[BenchmarkReport] = []

    # ── Engine registration ────────────────────────────────────
    def register(self, name: str, fn: Callable) -> None:
        """
        Register an engine callable.

        fn signature: (canvas_bgr: np.ndarray) → (text: str, confidence: float)
        """
        self._engines[name] = fn
        log.debug("Benchmark: registered engine '%s'", name)

    # ── Single run ─────────────────────────────────────────────
    def _run_once(
        self,
        name: str,
        fn: Callable,
        img: np.ndarray,
    ) -> EngineResult:
        t0 = time.perf_counter()
        try:
            text, conf = fn(img)
            latency_ms = (time.perf_counter() - t0) * 1000
            return EngineResult(
                name=name,
                text=text,
                confidence=conf,
                latency_ms=latency_ms,
                success=True,
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000
            log.error("Benchmark '%s' failed: %s", name, exc)
            return EngineResult(
                name=name, text="", confidence=0.0,
                latency_ms=latency_ms, success=False, error=str(exc),
            )

    # ── Full benchmark ─────────────────────────────────────────
    def run_all(
        self,
        img: np.ndarray,
        runs: int = 1,
        image_path: Optional[str] = None,
    ) -> BenchmarkReport:
        """
        Run every registered engine *runs* times, average the latencies,
        keep the result from the last run.
        """
        results: List[EngineResult] = []

        for name, fn in self._engines.items():
            latencies: List[float] = []
            last: Optional[EngineResult] = None
            for _ in range(runs):
                last = self._run_once(name, fn, img)
                latencies.append(last.latency_ms)
            if last:
                last.latency_ms = statistics.mean(latencies)
                results.append(last)

        report = BenchmarkReport(
            engine_results=results,
            image_path=image_path,
            runs=runs,
        )
        self._reports.append(report)
        return report

    # ── Reporting ──────────────────────────────────────────────
    @staticmethod
    def print_report(report: BenchmarkReport) -> None:
        """Pretty-print a benchmark report to stdout."""
        sep = "─" * 64
        print(f"\n{sep}")
        print(f"  OCR BENCHMARK REPORT   (runs per engine: {report.runs})")
        if report.image_path:
            print(f"  Image: {report.image_path}")
        print(sep)
        print(f"  {'Engine':<14} {'Text':<26} {'Conf':>6} {'ms':>8}  Status")
        print(sep)
        for r in sorted(report.engine_results, key=lambda x: -x.confidence):
            status = "✓" if r.success else "✗"
            short  = (r.text[:24] + "…") if len(r.text) > 25 else r.text
            err    = f"  ERR: {r.error[:30]}" if r.error else ""
            print(
                f"  {r.name:<14} {short!r:<26} {r.confidence:6.3f} "
                f"{r.latency_ms:8.1f}  {status}{err}"
            )
        print(sep)
        best = report.best()
        fast = report.fastest()
        if best:
            print(f"  🏆  Best confidence : {best.name}  ({best.confidence:.3f})")
        if fast:
            print(f"  ⚡  Fastest         : {fast.name}  ({fast.latency_ms:.1f} ms)")
        print(f"{sep}\n")

    def last_report(self) -> Optional[BenchmarkReport]:
        return self._reports[-1] if self._reports else None


# ══════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════
def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark all OCR engines on a canvas image."
    )
    parser.add_argument("--image",  default="drawings/autosave.png",
                        help="Path to input PNG image")
    parser.add_argument("--runs",   type=int, default=3,
                        help="Number of timing runs per engine")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"[ERROR] Cannot read: {args.image}")
        return

    # Import here to avoid circular at module level
    from ocr.hybrid_ocr import HybridOCRPipeline
    pipeline = HybridOCRPipeline()
    bench    = OCRBenchmark()

    bench.register("tesseract", lambda i: pipeline.tesseract.recognize(i))
    bench.register("easyocr",   lambda i: pipeline.easyocr.recognize(i))
    bench.register("trocr",     lambda i: pipeline.trocr.recognize(i))

    report = bench.run_all(img, runs=args.runs, image_path=args.image)
    OCRBenchmark.print_report(report)


if __name__ == "__main__":
    _cli()
