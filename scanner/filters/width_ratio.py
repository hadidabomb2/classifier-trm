"""
filters/width_ratio.py — Layer 1: Bounding box width ratio filter.

Principle
---------
Real content text (paragraphs, articles, chat messages) spans a meaningful
fraction of the capture area.  UI chrome elements — tab titles, toolbar
buttons, sidebar labels, icon captions — are narrow, isolated detections.

Any RapidOCR detection whose bounding-box width is less than
`min_ratio × capture_width` is discarded before text assembly.

Tuning
------
- 0.20 (default): requires the text to span ≥ 20 % of the capture width.
  Works well for full-screen or near-full-screen captures.
- Lower (0.10–0.15): safer for narrow-column layouts (chat, code, sidebars).
- Raise (0.25–0.35): more aggressive; good for standard article layouts.
"""
from __future__ import annotations

from typing import List


class WidthRatioFilter:
    """
    Drop individual OCR detections that are narrower than a fraction of the
    full capture width.

    Args:
        min_ratio: Minimum (bbox_width / capture_width) to keep a detection.
    """

    def __init__(self, min_ratio: float = 0.20) -> None:
        self._min_ratio = min_ratio

    def apply(self, detections: List, capture_width: int) -> List:
        """Return only detections whose bounding box is wide enough."""
        if capture_width <= 0 or not detections:
            return detections

        min_px = self._min_ratio * capture_width
        kept   = []

        for det in detections:
            if len(det) < 1:
                continue
            try:
                xs     = [float(p[0]) for p in det[0]]
                bbox_w = max(xs) - min(xs)
                if bbox_w >= min_px:
                    kept.append(det)
            except (TypeError, IndexError, ValueError):
                # Malformed bbox: keep the detection rather than silently drop it.
                kept.append(det)

        return kept
