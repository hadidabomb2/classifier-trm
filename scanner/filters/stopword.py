"""
filters/stopword.py — Layer 3: Stopword presence check.

Principle
---------
Real English prose always contains common function words — "the", "a", "is",
"it", "this", "was", "are", etc.  These are called *stopwords* and make up
roughly 30–50 % of natural text.

UI text is almost entirely nouns and verbs: "Settings", "Notifications",
"Follow", "Share", "Trending", "Explore".  After L1 and L2 have eliminated
narrow and short detections, if the *remaining* text still has a stopword
ratio near zero, it's a collection of label fragments — not real prose.

This filter operates on the **complete joined text block**, not individual
detections.  The check is only meaningful once enough words have accumulated,
so the filter passes through anything below `min_words` unconditionally.

Tuning
------
- min_ratio 0.05 (default): passes if at least 5 % of words are stopwords.
  Very permissive — catches only pure label dumps.  Prose sits at 30–50 %.
- min_words 15 (default): don't judge text that's too short to be reliable.
- Raise min_ratio to 0.10–0.15 if UI fragments are still slipping through.
- Lower min_words to 10 if you want to catch shorter garbage blocks.
"""
from __future__ import annotations

import re

# fmt: off
_STOPWORDS = frozenset({
    "a", "an", "the",
    "and", "but", "or", "nor", "for", "yet", "so",
    "in", "on", "at", "to", "of", "with", "by", "from",
    "about", "into", "through", "during", "before", "after",
    "above", "below", "between", "out", "off", "over", "under",
    "i", "me", "my", "we", "our", "you", "your",
    "he", "him", "his", "she", "her", "it", "its",
    "they", "them", "their", "this", "that", "these", "those",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "shall", "can",
    "not", "no", "as", "if", "then", "than",
    "there", "here", "when", "where", "which", "who", "how",
    "all", "each", "every", "both", "few", "more", "most",
    "other", "some", "such", "only", "just", "also", "very",
})
# fmt: on

# Punctuation to strip when testing if a token is a stopword.
_PUNCT = re.compile(r"[^\w]")


class StopwordFilter:
    """
    Reject the entire text block when the stopword ratio is below `min_ratio`.

    Args:
        min_ratio:  Minimum fraction of words that must be stopwords to keep
                    the block.  Default 0.05 (5 %).
        min_words:  Minimum total word count before the check is applied.
                    Blocks shorter than this are always passed through.
    """

    def __init__(self, min_ratio: float = 0.05, min_words: int = 15) -> None:
        self._min_ratio = min_ratio
        self._min_words = min_words

    def passes(self, text: str) -> bool:
        """
        Return True  → text looks like real prose, keep it.
        Return False → text looks like a label dump, discard it.
        """
        words = text.lower().split()
        if len(words) < self._min_words:
            return True  # too short to judge reliably — let it through

        stopword_count = sum(
            1 for w in words
            if _PUNCT.sub("", w) in _STOPWORDS
        )
        ratio = stopword_count / len(words)
        return ratio >= self._min_ratio
