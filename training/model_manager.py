"""
model_manager.py

Full lifecycle management for a TRM model:
  - Load base / fine-tuned model + tokenizer from HuggingFace or local path.
  - Attach a classification head sized to the TRM's label count.
  - Device placement (CUDA / MPS / CPU auto-detected).
  - Versioned checkpoint save / load / listing.
  - "Best" checkpoint promotion (copied to .../best/).
  - Final model export to the output directory.
  - Clean resource release (GPU memory, references).
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional, Tuple

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .config import ModelConfig, TRMConfig
from .utils import count_parameters, device_summary, get_logger, resolve_device

logger = get_logger(__name__)


class ModelManager:
    """
    Manages the model lifecycle for one TRM.

    Typical usage::

        manager = ModelManager(trm_config)
        model, tokenizer = manager.load()
        # … training …
        manager.save_checkpoint(model, tokenizer, tag="epoch-3", is_best=True)
        manager.close()

    All checkpoint artefacts live under::

        <checkpoint_dir>/<trm-name>/<tag>/   ← versioned checkpoint
        <checkpoint_dir>/<trm-name>/best/    ← current best (symlink-style copy)
    """

    def __init__(self, trm_config: TRMConfig) -> None:
        self.cfg: TRMConfig    = trm_config
        self.model_cfg: ModelConfig = trm_config.model
        self.device: torch.device   = resolve_device()
        self._model: Optional[PreTrainedModel]          = None
        self._tokenizer: Optional[PreTrainedTokenizerBase] = None
        self._ensure_dirs()
        logger.info("ModelManager initialised  device=%s", device_summary())

    # ── Public API ────────────────────────────────────────────────────────────

    def load(
        self,
        checkpoint_path: Optional[Path] = None,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """
        Load model and tokenizer.

        If *checkpoint_path* is given the model weights are loaded from that
        directory (must contain a valid ``config.json`` + weights file).
        Otherwise the pre-trained base model named in ``ModelConfig.model_name``
        is fetched from HuggingFace (cached locally).

        The tokenizer is always loaded from the original base model name so that
        the vocabulary is consistent regardless of which checkpoint is loaded.

        Returns:
            (model, tokenizer) placed on the active device.
        """
        cache_dir  = str(self.model_cfg.cache_dir) if self.model_cfg.cache_dir else None
        model_src  = str(checkpoint_path) if checkpoint_path else self.model_cfg.model_name

        logger.info("Loading tokenizer  ← %s", self.model_cfg.model_name)
        tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            self.model_cfg.model_name,
            cache_dir=cache_dir,
            use_fast=True,
        )

        id2label: dict[int, str] = {i: lbl for i, lbl in enumerate(self.cfg.labels)}
        label2id: dict[str, int] = {lbl: i for i, lbl in id2label.items()}

        logger.info("Loading model      ← %s", model_src)
        model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
            model_src,
            num_labels=self.model_cfg.num_labels,
            id2label=id2label,
            label2id=label2id,
            cache_dir=cache_dir,
            # Resize the classification head when loading a base model that
            # was pre-trained with a different (or no) num_labels.
            ignore_mismatched_sizes=True,
        )
        model = model.to(self.device)

        stats = count_parameters(model)
        logger.info(
            "Model ready  total=%dM  trainable=%dM  device=%s",
            stats["total"] // 1_000_000,
            stats["trainable"] // 1_000_000,
            self.device,
        )

        self._model     = model
        self._tokenizer = tokenizer
        return model, tokenizer

    def save_checkpoint(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        tag: str,
        is_best: bool = False,
    ) -> Path:
        """
        Persist model + tokenizer to a versioned checkpoint directory.

        Args:
            tag:      Sub-directory name, e.g. ``"epoch-3"`` or ``"step-500"``.
            is_best:  If True, also copy the checkpoint to the ``best/`` slot.

        Returns:
            Path to the saved checkpoint directory.
        """
        ckpt_dir = self._checkpoint_root() / tag
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(ckpt_dir))
        tokenizer.save_pretrained(str(ckpt_dir))
        logger.info("Checkpoint saved → %s", ckpt_dir)

        if is_best:
            best_dir = self._checkpoint_root() / "best"
            if best_dir.exists():
                shutil.rmtree(best_dir)
            shutil.copytree(ckpt_dir, best_dir)
            logger.info("Best checkpoint updated → %s", best_dir)

        return ckpt_dir

    def load_best_checkpoint(self) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """
        Load the ``best/`` checkpoint if it exists, else load the base model.
        """
        best_dir = self._checkpoint_root() / "best"
        if best_dir.exists():
            logger.info("Loading best checkpoint ← %s", best_dir)
            return self.load(checkpoint_path=best_dir)
        logger.info("No best checkpoint found — loading base model.")
        return self.load()

    def list_checkpoints(self) -> list[Path]:
        """Return all checkpoint subdirectories for this TRM, sorted by name."""
        root = self._checkpoint_root()
        if not root.exists():
            return []
        return sorted(p for p in root.iterdir() if p.is_dir() and p.name != "best")

    def export_final(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
    ) -> Path:
        """
        Copy the final model to the configured output directory.

        Suitable as the artefact directory for deployment / inference.

        Returns:
            Path to the exported model directory.
        """
        out_dir = Path(self.cfg.training.output_dir) / self.cfg.name
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))
        logger.info("Final model exported → %s", out_dir)
        return out_dir

    def close(self) -> None:
        """
        Release device memory and Python references.

        Safe to call even if ``load()`` was never called.
        """
        if self._model is not None:
            self._model.cpu()
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("ModelManager closed — device memory released.")

    # ── Internals ─────────────────────────────────────────────────────────────

    def _checkpoint_root(self) -> Path:
        return Path(self.cfg.training.checkpoint_dir) / self.cfg.name

    def _ensure_dirs(self) -> None:
        for d in (
            self._checkpoint_root(),
            Path(self.cfg.training.output_dir) / self.cfg.name,
        ):
            d.mkdir(parents=True, exist_ok=True)
        if self.model_cfg.cache_dir:
            Path(self.model_cfg.cache_dir).mkdir(parents=True, exist_ok=True)
