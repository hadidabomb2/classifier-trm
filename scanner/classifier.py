"""
classifier.py

Thin wrapper around a HuggingFace text-classification pipeline
for the exported TRM model.

Long texts are split into 512-token chunks and scores are averaged across
all chunks before picking the top label — no content is discarded.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List


# DistilBERT hard token limit (includes [CLS] + [SEP]).
# We use 510 for content so the tokenizer always has room for special tokens.
_CHUNK_TOKENS = 510


class ContentClassifier:
    """
    Loads a TRM model once and classifies text strings.

    Args:
        model_path: Path to the exported model directory.
        device:     -1 = CPU (default).  0, 1, … = GPU index.
    """

    def __init__(self, model_path: Path, device: int = -1) -> None:
        from transformers import AutoTokenizer, pipeline as hf_pipeline

        model_str = str(model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(model_str)
        self._pipe = hf_pipeline(
            "text-classification",
            model=model_str,
            tokenizer=self._tokenizer,
            device=device,
            top_k=None,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def classify(self, text: str, max_length: int = 2048) -> Dict:
        """
        Split *text* into 512-token chunks, classify each chunk, and return
        the label with the highest *average* score across all chunks.

        Returns label="UNCERTAIN" / score=0.0 when *text* is empty.
        """
        if not text or not text.strip():
            return {"label": "UNCERTAIN", "score": 0.0, "all_scores": {}}

        chunks = self._chunk(text)

        # Batch all chunks in a single pipeline call — efficient on both CPU and GPU.
        batch_results: List[List[Dict]] = self._pipe(
            chunks,
            truncation=True,
            max_length=512,
            batch_size=len(chunks),
        )

        # Average per-label softmax scores across chunks.
        totals: Dict[str, float] = {}
        for chunk_scores in batch_results:
            for item in chunk_scores:
                totals[item["label"]] = totals.get(item["label"], 0.0) + item["score"]

        n = len(chunks)
        all_scores = {label: round(score / n, 4) for label, score in totals.items()}
        top_label  = max(all_scores, key=all_scores.__getitem__)

        return {
            "label":      top_label,
            "score":      all_scores[top_label],
            "all_scores": all_scores,
            "n_chunks":   n,          # avoids re-chunking in scanner.py
        }

    # ── Internals ─────────────────────────────────────────────────────────────

    def _chunk(self, text: str) -> List[str]:
        """
        Tokenize *text*, split into _CHUNK_TOKENS-token windows, decode back
        to strings.  Returns a list of at least one string.

        Note: The transformers tokenizer may warn if the raw text exceeds 512
        tokens — that warning is expected and harmless here because we are
        intentionally encoding the full text *before* chunking it ourselves.
        """
        import warnings
        # Suppress only the "Token indices sequence length > 512" UserWarning
        # that transformers raises when we encode a long document.  We handle
        # the length ourselves via windowing, so the warning is misleading.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Token indices sequence length",
                category=UserWarning,
            )
            ids = self._tokenizer.encode(text, add_special_tokens=False)

        if not ids:
            return [text]

        windows = [ids[i: i + _CHUNK_TOKENS] for i in range(0, len(ids), _CHUNK_TOKENS)]
        return [self._tokenizer.decode(w, skip_special_tokens=True) for w in windows]
