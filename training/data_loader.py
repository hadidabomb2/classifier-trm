"""
data_loader.py

Responsible for:
  1. Downloading HuggingFace datasets defined in DatasetSourceConfig entries.
  2. Applying text preprocessors and label mappings.
  3. Length-filtering and deduplication.
  4. Writing per-source JSONL datadumps (cached on disk to avoid re-downloading).
  5. Merging and balancing all sources into a single merged datadump.
  6. Loading the merged datadump and splitting it into train / val / test sets.

Only JSONL datadumps are used for training — raw dataset objects are never
passed directly to the trainer.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from datasets import load_dataset
from tqdm import tqdm

from .config import DataConfig, DatasetSourceConfig, TRMConfig
from .utils import get_logger, read_jsonl, write_jsonl

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Text preprocessors
# ─────────────────────────────────────────────────────────────────────────────

def first_paragraph(text: str, max_chars: int = 1_500) -> str:
    """Return the first non-empty paragraph, capped at max_chars."""
    for para in text.split("\n\n"):
        para = para.strip()
        if len(para) >= 40:
            return para[:max_chars]
    return text[:max_chars]


def last_assistant_turn(text: str) -> str:
    """
    Extract the final assistant response from an Anthropic HH-RLHF conversation.

    Format:  "\\n\\nHuman: ...\\n\\nAssistant: ...\\n\\nHuman: ...\\n\\nAssistant: ..."
    Returns only the last assistant segment, trimmed of any trailing Human turn.
    """
    marker = "\n\nAssistant:"
    parts = text.split(marker)
    if len(parts) < 2:
        return text.strip()
    last = parts[-1]
    # Drop any trailing Human: turn
    human_idx = last.find("\n\nHuman:")
    if human_idx != -1:
        last = last[:human_idx]
    return last.strip()


# Registry of available preprocessors keyed by name used in DatasetSourceConfig.
_PREPROCESSORS: Dict[str, Callable[[str], str]] = {
    "first_paragraph":      first_paragraph,
    "last_assistant_turn":  last_assistant_turn,
}


# ─────────────────────────────────────────────────────────────────────────────
# DatasetBuilder
# ─────────────────────────────────────────────────────────────────────────────

class DatasetBuilder:
    """
    Downloads, preprocesses, and caches all dataset sources defined in a
    TRMConfig into JSONL datadumps.  The merged datadump is keyed to the TRM
    name and is used as the sole input to the training pipeline.

    Usage::

        builder = DatasetBuilder(trm_config)
        merged_path = builder.build_datadumps(force_rebuild=False)
        splits = builder.load_splits()   # returns {"train": [...], "val": [...], "test": [...]}
    """

    def __init__(self, trm_config: TRMConfig) -> None:
        self.cfg: TRMConfig     = trm_config
        self.data_cfg: DataConfig = trm_config.data
        self.label_set           = set(trm_config.labels)
        self._ensure_dirs()

    # ── Public API ────────────────────────────────────────────────────────────

    def build_datadumps(self, force_rebuild: bool = False) -> Path:
        """
        Build the merged JSONL datadump from all sources.

        Each source is cached individually under data/raw/.
        The balanced merged dump lives at data/datadumps/<trm-name>_merged.jsonl.

        Args:
            force_rebuild: Re-download and reprocess all sources even if cached.

        Returns:
            Path to the merged datadump file.
        """
        merged_path = self._merged_path()

        if merged_path.exists() and not force_rebuild:
            logger.info(
                "Merged datadump already exists (%s). Pass force_rebuild=True to regenerate.",
                merged_path,
            )
            return merged_path

        all_records: List[Dict[str, str]] = []
        for source in self.data_cfg.sources:
            records = self._load_or_download_source(source, force_rebuild=force_rebuild)
            logger.info(
                "  %-45s → %4d records  [label subset: %s]",
                f"{source.hf_path}/{source.hf_name or ''}",
                len(records),
                set(r["label"] for r in records),
            )
            all_records.extend(records)

        balanced = self._balance_and_shuffle(all_records)

        write_jsonl(merged_path, balanced)
        self._log_distribution(balanced)
        logger.info(
            "Merged datadump written → %s  (%d total records)", merged_path, len(balanced)
        )
        return merged_path

    def load_splits(
        self,
        val_split: Optional[float] = None,
        test_split: Optional[float] = None,
    ) -> Dict[str, List[Dict[str, str]]]:
        """
        Load the merged datadump and return stratified train / val / test splits.

        Stratification is performed per-label so every label appears in every split.

        Returns:
            {"train": [...], "val": [...], "test": [...]}
        """
        merged_path = self._merged_path()
        if not merged_path.exists():
            raise FileNotFoundError(
                f"Datadump not found at '{merged_path}'. "
                "Run build_datadumps() first."
            )

        records   = read_jsonl(merged_path)
        val_pct   = val_split  if val_split  is not None else self.data_cfg.val_split
        test_pct  = test_split if test_split is not None else self.data_cfg.test_split
        train, val, test = self._stratified_split(records, val_pct, test_pct)

        logger.info(
            "Split sizes — train: %d  val: %d  test: %d",
            len(train), len(val), len(test),
        )
        return {"train": train, "val": val, "test": test}

    # ── Internal: Source handling ─────────────────────────────────────────────

    def _load_or_download_source(
        self,
        source: DatasetSourceConfig,
        force_rebuild: bool,
    ) -> List[Dict[str, str]]:
        cache_path = self._source_cache_path(source)
        if cache_path.exists() and not force_rebuild:
            return read_jsonl(cache_path)
        return self._download_source(source, cache_path)

    def _download_source(
        self,
        source: DatasetSourceConfig,
        cache_path: Path,
    ) -> List[Dict[str, str]]:
        logger.info(
            "Downloading %-40s  split=%-6s  streaming=%s",
            f"{source.hf_path} ({source.hf_name or 'default'})",
            source.split,
            source.streaming,
        )
        try:
            ds = load_dataset(
                source.hf_path,
                source.hf_name,
                split=source.split,
                streaming=source.streaming,
                trust_remote_code=False,
                **source.hf_kwargs,
            )
        except Exception as exc:
            logger.warning(
                "Could not load '%s' (%s): %s — skipping source.",
                source.hf_path, source.hf_name, exc,
            )
            return []

        preprocessor: Optional[Callable[[str], str]] = (
            _PREPROCESSORS.get(source.preprocessor) if source.preprocessor else None
        )

        records: List[Dict[str, str]] = []
        seen_hashes: set = set()

        for item in tqdm(
            ds,
            desc=f"  {source.hf_path}/{source.hf_name or ''}",
            leave=False,
        ):
            if len(records) >= source.max_samples:
                break

            raw_text: str = str(item.get(source.text_column) or "")
            if preprocessor:
                raw_text = preprocessor(raw_text)
            raw_text = raw_text.strip()

            # Length filter
            if not (source.filter_min_chars <= len(raw_text) <= source.filter_max_chars):
                continue

            # Exact-duplicate filter (cheap hash)
            h = hashlib.md5(raw_text.encode("utf-8"), usedforsecurity=False).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            # Label resolution
            label = self._resolve_label(item, source)
            if label is None:
                continue

            records.append({"text": raw_text, "label": label})

        write_jsonl(cache_path, records)
        return records

    @staticmethod
    def _resolve_label(
        item: Dict[str, Any],
        source: DatasetSourceConfig,
    ) -> Optional[str]:
        """Map a raw dataset item's label to a TRM label string, or None to skip."""
        if source.label_column is None:
            return source.label_mapping.get(None)

        raw = item.get(source.label_column)
        # Try int, then string, then string of int
        for key in (raw, str(raw)):
            if key in source.label_mapping:
                return source.label_mapping[key]
        return None

    # ── Internal: Balancing & splitting ──────────────────────────────────────

    def _balance_and_shuffle(
        self, records: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Cap each label at max_samples_per_label and shuffle the result."""
        by_label: Dict[str, List[Dict[str, str]]] = {lbl: [] for lbl in self.label_set}
        for r in records:
            lbl = r.get("label", "")
            if lbl in by_label:
                by_label[lbl].append(r)

        limit = self.data_cfg.max_samples_per_label
        rng   = random.Random(self.data_cfg.seed)

        balanced: List[Dict[str, str]] = []
        for lbl, items in by_label.items():
            rng.shuffle(items)
            balanced.extend(items[:limit])

        rng.shuffle(balanced)
        return balanced

    def _stratified_split(
        self,
        records: List[Dict[str, str]],
        val_pct: float,
        test_pct: float,
    ) -> tuple[List[Dict], List[Dict], List[Dict]]:
        """Per-label stratified split to guarantee label coverage in all sets."""
        by_label: Dict[str, List[Dict[str, str]]] = {}
        for r in records:
            by_label.setdefault(r["label"], []).append(r)

        rng = random.Random(self.data_cfg.seed)
        train, val, test = [], [], []

        for items in by_label.values():
            rng.shuffle(items)
            n       = len(items)
            n_test  = max(1, round(n * test_pct))
            n_val   = max(1, round(n * val_pct))
            n_train = n - n_test - n_val
            if n_train < 1:
                # Degenerate: too few samples — put everything in train
                train.extend(items)
                continue
            test.extend(items[:n_test])
            val.extend(items[n_test : n_test + n_val])
            train.extend(items[n_test + n_val :])

        rng.shuffle(train)
        rng.shuffle(val)
        rng.shuffle(test)
        return train, val, test

    # ── Internal: Paths & dirs ────────────────────────────────────────────────

    def _merged_path(self) -> Path:
        return Path(self.data_cfg.datadump_dir) / f"{self.cfg.name}_merged.jsonl"

    def _source_cache_path(self, source: DatasetSourceConfig) -> Path:
        # Include text_column so sources that share the same path/split but read
        # different columns (e.g. hh-rlhf "chosen" vs "rejected") don't collide.
        key = (
            f"{source.hf_path}__{source.hf_name or 'default'}"
            f"__{source.split}__{source.text_column}"
        )
        # Also hash any extra kwargs (e.g. data_dir) that further disambiguate the source.
        if source.hf_kwargs:
            kwargs_hash = hashlib.md5(
                json.dumps(source.hf_kwargs, sort_keys=True).encode("utf-8"),
                usedforsecurity=False,
            ).hexdigest()[:8]
            key += f"__{kwargs_hash}"
        safe = key.replace("/", "-").replace(" ", "_")
        return Path(self.data_cfg.raw_dir) / f"{safe}.jsonl"

    def _ensure_dirs(self) -> None:
        for d in (
            self.data_cfg.raw_dir,
            self.data_cfg.datadump_dir,
            self.data_cfg.processed_dir,
        ):
            Path(d).mkdir(parents=True, exist_ok=True)

    def _log_distribution(self, records: List[Dict[str, str]]) -> None:
        from collections import Counter
        dist = Counter(r["label"] for r in records)
        logger.info("Label distribution:")
        for lbl in sorted(dist):
            logger.info("  %-10s %d", lbl, dist[lbl])
