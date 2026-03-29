"""
scanner/filters — Content filtering pipeline for OCR output.

Filters are applied in layers:

  L1  WidthRatioFilter  — drop narrow detections (buttons, tabs, icon labels)
  L2  WordCountFilter   — drop detections with too few words (single tokens)
  L3  StopwordFilter    — reject full text blocks that look like UI label lists
  L4  DeduplicationFilter — skip re-classification when the screen hasn't changed

L1–L3 live inside OCRExtractor (per-frame, per-detection).
L4 lives in Scanner (between frames, stateful).
"""
from .width_ratio import WidthRatioFilter
from .word_count import WordCountFilter
from .stopword import StopwordFilter
from .dedup import DeduplicationFilter

__all__ = [
    "WidthRatioFilter",
    "WordCountFilter",
    "StopwordFilter",
    "DeduplicationFilter",
]
