"""
scanner.py — TRM real-time screen scanner.

  python -m scanner
  python -m scanner --region 0 130 1920 910   # crop below browser chrome
  python -m scanner --show-region              # draw a border around capture area
  python -m scanner --interval 2000 --gpu 0
"""
from __future__ import annotations

import queue
import signal
import threading
import time
import argparse
import traceback
from pathlib import Path
from typing import Dict

from .config import ScannerConfig
from .capture import ScreenCapture
from .ocr import OCRExtractor
from .classifier import ContentClassifier
from .overlay import ScannerOverlay
from .filters import DeduplicationFilter


class Scanner:
    def __init__(self, config: ScannerConfig) -> None:
        self._cfg        = config
        self._queue      = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()

        print("[scanner] Loading capture…")
        self._capture = ScreenCapture(region=config.capture_region)

        print("[scanner] Loading OCR…")
        self._ocr = OCRExtractor(language=config.ocr_language, psm=config.ocr_psm)

        print(f"[scanner] Loading models…")
        self._classifiers: Dict[str, ContentClassifier] = {}
        trms = list(config.model_paths.keys())
        if config.enabled_trms is not None:
            trms = [t for t in trms if t in config.enabled_trms]
        for trm_name in trms:
            model_path = config.model_paths[trm_name]
            if model_path.exists():
                print(f"[scanner] Loading {trm_name} model from {model_path}")
                self._classifiers[trm_name] = ContentClassifier(model_path=model_path, device=config.device)
            else:
                print(f"[scanner] WARNING: {trm_name} model not found at {model_path} — skipping")

        print("[scanner] Building overlay…")
        self._overlay = ScannerOverlay(config, self._queue)
        # Layer 4: skip re-classification when the screen hasn't meaningfully changed.
        self._dedup = DeduplicationFilter(similarity_threshold=0.85)

        region = config.capture_region
        if region:
            print(f"[scanner] Capture region: left={region[0]} top={region[1]} width={region[2]} height={region[3]}")
        else:
            print("[scanner] Capture region: full screen")
            print("[scanner] TIP: Use --region L T W H to crop below browser chrome (e.g. --region 0 130 1920 910)")
        print("[scanner] Ready.")

    # ── Scan loop (background thread) ─────────────────────────────────────────

    def _scan_loop(self) -> None:
        interval = self._cfg.refresh_interval_ms / 1000.0

        while not self._stop_event.is_set():
            if self._overlay.paused:
                self._dedup.reset()  # treat first post-pause frame as new content
                time.sleep(0.5)
                continue

            try:
                image = self._capture.capture()
                text  = self._ocr.extract(image)
                chars = len(text)

                print(f"[scan] OCR ({chars} chars):\n{text}\n", flush=True)

                if chars < self._cfg.min_text_chars:
                    result = {
                        "trm_results": {},
                        "status": f"Too little text ({chars} chars)",
                    }
                elif not self._dedup.is_new(text):
                    print("[scan] Duplicate frame — skipping.", flush=True)
                    result = None
                else:
                    trm_results: Dict[str, Dict] = {}
                    n_chunks = 1
                    for trm_name, clf in self._classifiers.items():
                        r = clf.classify(text)
                        if r["score"] < self._cfg.confidence_threshold:
                            r["label"] = "UNCERTAIN"
                        trm_results[trm_name] = r
                        n_chunks = max(n_chunks, r.get("n_chunks", 1))
                        print(
                            f"[scan] {trm_name:8s} → {r['label']} {r['score']:.1%}  "
                            + "  ".join(f"{k}:{v:.0%}" for k, v in r["all_scores"].items()),
                            flush=True,
                        )
                    result = {
                        "trm_results": trm_results,
                        "status": f"{chars} chars · {n_chunks} chunk(s)",
                    }

            except Exception as exc:
                traceback.print_exc()
                result = {
                    "trm_results": {},
                    "status": f"Error: {exc}",
                }

            # Drop stale result, push fresh one (skip on duplicate frames).
            if result is not None:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                self._queue.put_nowait(result)

            time.sleep(interval)

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self) -> None:
        # Route Ctrl-C through tkinter's event loop so the shutdown is clean.
        def _sigint_handler(*_):
            print("\n[scanner] Interrupted — shutting down…", flush=True)
            self._overlay.shutdown()

        signal.signal(signal.SIGINT, _sigint_handler)

        t = threading.Thread(target=self._scan_loop, name="trm-scan", daemon=True)
        t.start()
        try:
            self._overlay.run()
        finally:
            self._stop_event.set()
            self._capture.close()
            print("[scanner] Stopped.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(prog="scanner", description="TRM real-time screen scanner.")
    parser.add_argument("--models-dir", type=Path, default=None,
                        help="Base directory containing impact/flavor/purpose/lifespan subdirs")
    parser.add_argument("--region",      type=int,   nargs=4, metavar=("L","T","W","H"), default=None,
                        help="Capture region: left top width height")
    parser.add_argument("--interval",    type=int,   default=1500, metavar="MS")
    parser.add_argument("--threshold",   type=float, default=0.50)
    parser.add_argument("--gpu",         type=int,   default=None, metavar="IDX")
    parser.add_argument("--show-region", action="store_true",
                        help="Draw a visible border around the capture area.")

    args = parser.parse_args()

    cfg = ScannerConfig(
        refresh_interval_ms=args.interval,
        confidence_threshold=args.threshold,
        show_region=args.show_region,
    )
    if args.models_dir is not None:
        cfg.model_paths = {
            name: args.models_dir / name
            for name in cfg.model_paths
        }
    if args.region is not None:
        cfg.capture_region = tuple(args.region)
    if args.gpu is not None:
        cfg.device = args.gpu

    Scanner(cfg).run()

