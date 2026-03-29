"""
filters/word_count.py — Layer 2: Minimum word count per detection.

Principle
---------
Real content always has multiple words per reading unit.  The following are
all single-token or very-short OCR detections that are useless for classification:

  - Tab titles: "Home", "Notifications", "For You"
  - Timestamps: "12:34", "2 min ago"
  - Button labels: "Follow", "Like", "Share", "Subscribe"
  - Icon badges: "3", "99+"
  - Navigation items: "Settings", "Profile", "Explore"

Any detection with fewer than `min_words` whitespace-separated tokens is
discarded before spatial ordering.

Tuning
------
- 4 (default): eliminates most single-word and two-word UI fragments while
  keeping legitimate short sentences ("This is great.", "He said yes.").
- Lower (2–3): gentler; useful if your target content has lots of short captions.
- Raise (5–6): more aggressive; only keeps proper multi-word prose.
"""
from __future__ import annotations

from typing import List


class WordCountFilter:
    """
    Drop individual OCR detections that contain fewer than `min_words` words.

    Args:
        min_words: Minimum whitespace-separated token count to keep a detection.
    """

    def __init__(self, min_words: int = 4) -> None:
        self._min_words = min_words

    def apply(self, detections: List) -> List:
        """Return only detections that have enough words."""
        kept = []
        for det in detections:
            if len(det) < 2:
                continue
            text = det[1] or ""
            if len(text.split()) >= self._min_words:
                kept.append(det)
        return kept
