"""
utils.py

Shared utilities used across all pipeline stages:
  - Structured logging setup (one-time, stdout)
  - Device detection (CUDA > MPS > CPU)
  - Reproducibility seeding
  - Lightweight JSON-Lines I/O helpers
"""
from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, Iterator, List

import numpy as np
import torch

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
_LOG_FORMAT  = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_logging_configured = False


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger once.  Safe to call multiple times."""
    global _logging_configured
    if _logging_configured:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)
    # Silence overly verbose third-party loggers
    for noisy in ("urllib3", "filelock", "datasets.utils", "transformers.tokenization_utils"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _logging_configured = True


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a named logger, ensuring the root handler is configured."""
    setup_logging(level)
    return logging.getLogger(name)


# ─────────────────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────────────────
def resolve_device() -> torch.device:
    """Return the best available torch device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_summary() -> str:
    device = resolve_device()
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        return f"cuda ({props.name}, {props.total_memory // 1_048_576} MB)"
    return device.type


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# JSON-Lines helpers
# ─────────────────────────────────────────────────────────────────────────────
def write_jsonl(path: Path, records: List[Dict]) -> None:
    """Write a list of dicts to a JSON-Lines file (UTF-8, one object per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict]:
    """Load all records from a JSON-Lines file."""
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def stream_jsonl(path: Path) -> Iterator[Dict]:
    """Memory-efficient generator over a JSON-Lines file."""
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


# ─────────────────────────────────────────────────────────────────────────────
# Param count helper
# ─────────────────────────────────────────────────────────────────────────────
def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
