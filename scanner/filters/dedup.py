"""
filters/dedup.py — Layer 4: Inter-scan deduplication.

Principle
---------
Between scan cycles the screen may not have changed at all, or have changed
only trivially (cursor blink, timestamp tick, scroll indicator update).
Re-classifying identical content wastes CPU and causes the overlay to flash
the same result repeatedly.

We compare the new text to the last accepted text using
`difflib.SequenceMatcher`, which is character-level, dependency-free, and
fast enough for text up to a few thousand characters.  If the similarity
ratio exceeds `similarity_threshold`, the frame is treated as a duplicate
and classification is skipped — the overlay keeps displaying the last result.

State update
------------
The internal `_last_text` is updated **only when a frame is accepted** (i.e.
when `is_new()` returns True).  This means a sequence of near-identical
frames chains correctly: each is compared to the last *different* content,
not the last frame.

Tuning
------
- 0.85 (default): ignores changes smaller than ~15 % of the text.
  Handles cursor blinks, minor UI badge updates, timestamp changes.
- Lower (0.70–0.80): more sensitive; fires on smaller content changes.
  Useful if you want to catch paragraph-by-paragraph scroll.
- Raise (0.90–0.95): less sensitive; only re-classifies on major page changes.
"""
from __future__ import annotations

import difflib


class DeduplicationFilter:
    """
    Stateful filter that returns False when new text is too similar to the
    last accepted text.

    Args:
        similarity_threshold: SequenceMatcher ratio above which text is
                              considered a duplicate and should be skipped.
    """

    def __init__(self, similarity_threshold: float = 0.85) -> None:
        self._threshold  = similarity_threshold
        self._last_text  = ""

    def is_new(self, text: str) -> bool:
        """
        Return True  → content is new enough, proceed with classification.
        Return False → content is a duplicate, skip this frame.

        Always returns True on the first non-empty call.
        """
        if not self._last_text:
            self._last_text = text
            return True

        ratio = difflib.SequenceMatcher(
            None, self._last_text, text, autojunk=False
        ).ratio()

        if ratio >= self._threshold:
            return False  # duplicate

        self._last_text = text
        return True

    def reset(self) -> None:
        """Force the next frame to be treated as new (e.g. after unpausing)."""
        self._last_text = ""
