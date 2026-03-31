"""
orchestrator.py

Entry-point and pipeline orchestrator for TRM training.

Ties together DatasetBuilder, ModelManager, and TRMTrainer into a
reproducible, step-by-step pipeline that can be controlled via CLI flags
or called programmatically.

Pipeline stages (in order):
  data    — Download sources and build the merged JSONL datadump.
  train   — Load model, run fine-tuning, save best checkpoint.
  eval    — Evaluate the best checkpoint on the held-out test split.
  export  — Copy the best checkpoint to the final output directory.

CLI usage::

    python -m training.orchestrator --trm impact
    python -m training.orchestrator --trm impact --steps data,train
    python -m training.orchestrator --trm flavor --force-rebuild
    python -m training.orchestrator --trm lifespan --checkpoint path/to/ckpt

Programmatic usage::

    from training.orchestrator import Orchestrator
    from training.config import TRMRegistry

    cfg  = TRMRegistry.get("impact")
    orch = Orchestrator(cfg)
    orch.run_pipeline(steps=["data", "train", "eval", "export"])
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from .config import TRMConfig, TRMRegistry
from .data_loader import DatasetBuilder
from .model_manager import ModelManager
from .trainer import TRMTrainer
from .utils import get_logger, setup_logging

logger = get_logger(__name__)

VALID_STEPS: tuple[str, ...] = ("data", "train", "eval", "export")


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class Orchestrator:
    """
    Coordinates all pipeline stages for a given TRM.

    Each stage is idempotent where possible:
      - ``run_data``   skips downloading if the merged datadump already exists
        (pass ``force_rebuild=True`` to override).
      - ``run_eval``   always reloads the best checkpoint before evaluating.
      - ``run_export`` always overwrites the output directory.
    """

    def __init__(self, trm_config: TRMConfig) -> None:
        self.cfg     = trm_config
        self.builder = DatasetBuilder(trm_config)
        self.manager = ModelManager(trm_config)

    # ── Stages ────────────────────────────────────────────────────────────────

    def run_data(self, force_rebuild: bool = False) -> None:
        """Stage 1 — build JSONL datadumps from all configured sources."""
        logger.info("══════════  STAGE: data  [TRM: %s]  ══════════", self.cfg.name)
        dump = self.builder.build_datadumps(force_rebuild=force_rebuild)
        logger.info("Datadump ready  →  %s", dump)

    def run_train(self, checkpoint: Optional[Path] = None) -> None:
        """
        Stage 2 — load data splits, fine-tune the model, save best checkpoint.

        Args:
            checkpoint: Optional path to an existing checkpoint to resume from.
                        When None the base pre-trained model is loaded.
        """
        logger.info("══════════  STAGE: train  [TRM: %s]  ══════════", self.cfg.name)

        splits          = self.builder.load_splits()
        model, tokenizer = self.manager.load(checkpoint_path=checkpoint)
        engine          = TRMTrainer(self.cfg, model, tokenizer)
        trainer         = engine.train(splits)

        best_hf_ckpt = trainer.state.best_model_checkpoint
        logger.info("HF Trainer best checkpoint: %s", best_hf_ckpt)

        # Reload best HF-internal checkpoint as our canonical "best"
        if best_hf_ckpt:
            model.config.update({"_best_hf_checkpoint": best_hf_ckpt})
            best_model, best_tok = self.manager.load(
                checkpoint_path=Path(best_hf_ckpt)
            )
            self.manager.save_checkpoint(
                best_model, best_tok,
                tag=f"epoch-{self.cfg.training.num_train_epochs}",
                is_best=True,
            )
        else:
            self.manager.save_checkpoint(
                model, tokenizer,
                tag=f"epoch-{self.cfg.training.num_train_epochs}",
                is_best=True,
            )

    def run_eval(self) -> None:
        """Stage 3 — reload best checkpoint and evaluate on the test split."""
        logger.info("══════════  STAGE: eval  [TRM: %s]  ══════════", self.cfg.name)

        splits               = self.builder.load_splits()
        model, tokenizer      = self.manager.load_best_checkpoint()
        engine               = TRMTrainer(self.cfg, model, tokenizer)
        dataset_dict         = engine._dicts_to_dataset_dict(splits)
        tokenised            = engine._tokenise(dataset_dict)
        args                 = engine._build_training_args()

        from transformers import DataCollatorWithPadding, Trainer

        collator = DataCollatorWithPadding(tokenizer=tokenizer)
        eval_trainer = Trainer(
            model=model,
            args=args,
            eval_dataset=tokenised["test"],
            processing_class=tokenizer,
            data_collator=collator,
            compute_metrics=engine._compute_metrics,
        )
        results = eval_trainer.evaluate(metric_key_prefix="eval")
        logger.info("Evaluation results: %s", results)

    def run_export(self) -> None:
        """Stage 4 — export the best checkpoint to the final output directory."""
        logger.info("══════════  STAGE: export  [TRM: %s]  ══════════", self.cfg.name)
        model, tokenizer = self.manager.load_best_checkpoint()
        out = self.manager.export_final(model, tokenizer)
        logger.info("Exported  →  %s", out)

    # ── Pipeline ──────────────────────────────────────────────────────────────

    def run_pipeline(
        self,
        steps: List[str],
        force_rebuild: bool = False,
        checkpoint: Optional[Path] = None,
    ) -> None:
        """
        Execute one or more named pipeline stages in the given order.

        Args:
            steps:         Ordered list of stage names from VALID_STEPS.
            force_rebuild: Passed to ``run_data``; forces source re-download.
            checkpoint:    Starting checkpoint path, passed to ``run_train``.
        """
        unknown = set(steps) - set(VALID_STEPS)
        if unknown:
            raise ValueError(
                f"Unknown pipeline steps: {unknown}.  Valid steps: {VALID_STEPS}"
            )

        try:
            for step in steps:
                if step == "data":
                    self.run_data(force_rebuild=force_rebuild)
                elif step == "train":
                    self.run_train(checkpoint=checkpoint)
                elif step == "eval":
                    self.run_eval()
                elif step == "export":
                    self.run_export()
        finally:
            # Always release device memory on exit, even after an exception.
            self.manager.close()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m training.orchestrator",
        description="TRM training pipeline — data → train → eval → export",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--trm",
        type=str,
        default="impact",
        metavar="NAME",
        help=(
            "Name of the TRM to train.\n"
            f"Registered TRMs: {TRMRegistry.list_names()}\n"
            "(default: impact)"
        ),
    )
    parser.add_argument(
        "--steps",
        type=str,
        default=",".join(VALID_STEPS),
        metavar="STAGE[,STAGE,...]",
        help=(
            f"Comma-separated pipeline stages to run.\n"
            f"Options: {', '.join(VALID_STEPS)}\n"
            "(default: all stages)"
        ),
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force re-download and rebuild of all dataset datadumps.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to an existing checkpoint directory to resume training from.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_parser()
    args   = parser.parse_args(argv)

    setup_logging(getattr(logging, args.log_level))

    try:
        trm_config = TRMRegistry.get(args.trm)
    except KeyError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    steps      = [s.strip() for s in args.steps.split(",") if s.strip()]
    checkpoint = Path(args.checkpoint) if args.checkpoint else None

    orchestrator = Orchestrator(trm_config)
    orchestrator.run_pipeline(
        steps=steps,
        force_rebuild=args.force_rebuild,
        checkpoint=checkpoint,
    )


if __name__ == "__main__":
    main()
