"""
ocr.py

Text extraction from a PIL Image using RapidOCR.

  pip install rapidocr-onnxruntime numpy
  (downloads ~20 MB ONNX models on first use, no binary install needed)

RapidOCR returns a list of detections, each shaped as:
  [bbox_points, text_string, confidence_score]

We filter by confidence so noise detections don't corrupt the classifier input.
"""
from __future__ import annotations

import re
import warnings
from typing import List

import numpy as np
from PIL import Image

from .filters import WidthRatioFilter, WordCountFilter, StopwordFilter

# Suppress the one-time ONNX provider info messages RapidOCR emits on import.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from rapidocr_onnxruntime import RapidOCR


class OCRExtractor:
    """
    Extracts visible text from a PIL Image using RapidOCR.

    Each RapidOCR detection has a confidence score in [0, 1].
    Detections below `min_confidence` are discarded so that blurry
    or partial text doesn't pollute the classifier.
    """

    def __init__(
        self,
        min_confidence: float = 0.5,
        # Legacy kwargs accepted for API compatibility — not used by RapidOCR.
        language: str = "eng",
        psm: int = 3,
    ) -> None:
        self._engine          = RapidOCR()
        self._min_confidence  = min_confidence
        # Content filtering pipeline (L1 → L2 → L3).
        self._width_filter    = WidthRatioFilter(min_ratio=0.20)
        self._word_filter     = WordCountFilter(min_words=4)
        self._stopword_filter = StopwordFilter(min_ratio=0.05, min_words=15)

    # ── Public API ────────────────────────────────────────────────────────────

    def extract(self, image: Image.Image) -> str:
        """
        Run OCR on *image* and return a single cleaned text string.
        Returns an empty string when nothing legible is found or when the
        content looks like UI chrome rather than real prose.
        """
        arr            = np.array(image.convert("RGB"))
        detections, _ = self._engine(arr)

        if not detections:
            return ""

        joined = self._filter_and_join(detections, capture_width=image.width)
        if not joined:
            return ""

        cleaned = self._clean(joined)

        # Layer 3: reject blocks that look like UI label lists (no stopwords).
        if not self._stopword_filter.passes(cleaned):
            return ""

        return cleaned

    # ── Internals ─────────────────────────────────────────────────────────────

    # Detections whose y-centers are within this many pixels are treated as
    # the same visual line.  20 px covers most font sizes on a 1080p display.
    _LINE_TOL_PX = 20

    def _filter_and_join(self, detections: List, capture_width: int) -> str:
        """
        Apply L1 (width ratio) and L2 (word count) pre-filters, then sort
        surviving detections into natural reading order and group into lines.
        Preserving line breaks gives the classifier the same structure the
        model saw during training and makes console output readable.
        """
        # Layer 1: discard narrow detections (UI chrome, button labels).
        detections = self._width_filter.apply(detections, capture_width)
        # Layer 2: discard detections with too few words (single tokens).
        detections = self._word_filter.apply(detections)

        candidates = []
        for det in detections:
            # det format: [bbox_4pts, text, score]  (score absent on some builds)
            if len(det) < 2:
                continue
            bbox  = det[0]   # [[x0,y0],[x1,y1],[x2,y2],[x3,y3]]
            text  = det[1]
            score = float(det[2]) if len(det) > 2 else 1.0
            if not (score >= self._min_confidence and text and text.strip()):
                continue
            try:
                xs = [float(p[0]) for p in bbox]
                ys = [float(p[1]) for p in bbox]
                cx = sum(xs) / len(xs)
                cy = sum(ys) / len(ys)
            except (TypeError, IndexError, ZeroDivisionError):
                # Fallback: append without spatial info, will land at end.
                candidates.append((float("inf"), 0.0, text.strip()))
                continue
            candidates.append((cy, cx, text.strip()))

        if not candidates:
            return ""

        # Primary sort: y-center (top → bottom), secondary: x-center (left → right).
        candidates.sort(key=lambda item: (item[0], item[1]))

        # Group into visual lines using y-center proximity.
        lines: List[List[str]] = []
        current_y   = candidates[0][0]
        current_line: List[str] = [candidates[0][2]]

        for cy, _cx, text in candidates[1:]:
            if abs(cy - current_y) <= self._LINE_TOL_PX:
                current_line.append(text)
            else:
                lines.append(current_line)
                current_line = [text]
                current_y = cy
        lines.append(current_line)

        return "\n".join(" ".join(line) for line in lines)

    @staticmethod
    def _clean(text: str) -> str:
        """Normalise intra-line spaces; preserve newlines for paragraph structure."""
        cleaned = []
        for line in text.splitlines():
            line = re.sub(r" {2,}", " ", line).strip()
            if line:
                cleaned.append(line)
        return "\n".join(cleaned)
