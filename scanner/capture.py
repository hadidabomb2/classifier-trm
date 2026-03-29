"""
capture.py

Screen capture using mss — returns a PIL Image of either the full primary
monitor or a configured sub-region.
"""
from __future__ import annotations

from typing import Optional, Tuple

import mss
import mss.tools
from PIL import Image


class ScreenCapture:
    """
    Thin wrapper around mss for fast, low-overhead screen capture.

    Usage::

        with ScreenCapture(region=(0, 0, 960, 1080)) as cap:
            img = cap.capture()   # PIL Image (RGB)
    """

    def __init__(self, region: Optional[Tuple[int, int, int, int]] = None) -> None:
        """
        Args:
            region: (left, top, width, height) in screen pixels.
                    None captures the entire primary monitor.
        """
        self._region  = region
        self._sct     = None   # Created lazily on first capture() call (mss is not thread-safe).
        self._monitor = None

    def _ensure_sct(self) -> None:
        """Create the mss instance on the calling thread if it doesn't exist yet."""
        if self._sct is None:
            self._sct     = mss.mss()
            self._monitor = self._build_monitor()

    # ── Public API ────────────────────────────────────────────────────────────

    def capture(self) -> Image.Image:
        """Capture and return the configured region as a PIL RGB Image."""
        self._ensure_sct()
        raw = self._sct.grab(self._monitor)
        return Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")

    def close(self) -> None:
        if self._sct is not None:
            self._sct.close()
            self._sct = None

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "ScreenCapture":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── Internals ─────────────────────────────────────────────────────────────

    def _build_monitor(self) -> dict:
        if self._region:
            left, top, width, height = self._region
            return {"left": left, "top": top, "width": width, "height": height}
        # monitors[0] is the virtual "all monitors" bounding box.
        # monitors[1] is always the primary monitor.
        return self._sct.monitors[1]
