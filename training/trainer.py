"""
trainer.py

Training engine for TRM fine-tuning.

Responsibilities:
  - Convert raw JSONL splits (dicts with "text" / "label" keys) into
    tokenised HuggingFace Dataset objects.
  - Build a TrainingArguments from TRMConfig.
  - Wire up HuggingFace Trainer with:
      • Dynamic padding (DataCollatorWithPadding)
      • EarlyStoppingCallback
      • compute_metrics: accuracy + macro-F1 + per-class F1
  - Run training, evaluate on the test set, and print a full
    classification report.
  - Return the fitted Trainer so the orchestrator can inspect
    the best checkpoint path and training logs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import (
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from .config import TrainingConfig, TRMConfig
from .utils import get_logger, seed_everything

logger = get_logger(__name__)


class TRMTrainer:
    """
    Fine-tuning engine for a single Tiny Recursive Model.

    Typical usage::

        engine = TRMTrainer(trm_config, model, tokenizer)
        trainer = engine.train(splits)       # splits = {"train": [...], "val": [...], ...}
        best_ckpt = trainer.state.best_model_checkpoint
    """

    def __init__(
        self,
        trm_config: TRMConfig,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        self.cfg       = trm_config
        self.model     = model
        self.tokenizer = tokenizer
        self.train_cfg: TrainingConfig = trm_config.training

        # Bidirectional label ↔ index maps
        self.label_to_id: Dict[str, int] = {
            lbl: i for i, lbl in enumerate(trm_config.labels)
        }
        self.id_to_label: Dict[int, str] = {
            i: lbl for lbl, i in self.label_to_id.items()
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def train(self, splits: Dict[str, List[Dict[str, str]]]) -> Trainer:
        """
        Run the full training loop.

        Args:
            splits: Dict with keys "train", "val", and optionally "test".
                    Each value is a list of {"text": str, "label": str} dicts
                    loaded directly from the JSONL datadumps.

        Returns:
            The fitted HuggingFace Trainer (inspect .state for metrics & log history).
        """
        seed_everything(self.train_cfg.seed)

        dataset_dict = self._dicts_to_dataset_dict(splits)
        tokenised    = self._tokenise(dataset_dict)
        train_size   = len(tokenised["train"])
        args         = self._build_training_args(train_size=train_size)
        collator     = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            pad_to_multiple_of=8 if self.train_cfg.fp16 else None,
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=tokenised["train"],
            eval_dataset=tokenised["val"],
            processing_class=self.tokenizer,
            data_collator=collator,
            compute_metrics=self._compute_metrics,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.train_cfg.early_stopping_patience
                )
            ],
        )

        logger.info(
            "Training '%s'  train=%d  val=%d  epochs=%d  device=%s",
            self.cfg.name,
            len(tokenised["train"]),
            len(tokenised["val"]),
            self.train_cfg.num_train_epochs,
            next(self.model.parameters()).device,
        )

        trainer.train()
        logger.info("Training complete.")

        if "test" in tokenised:
            self._run_test_evaluation(trainer, tokenised["test"])

        return trainer

    # ── Tokenisation ──────────────────────────────────────────────────────────

    def _dicts_to_dataset_dict(
        self, splits: Dict[str, List[Dict[str, str]]]
    ) -> DatasetDict:
        return DatasetDict(
            {name: Dataset.from_list(records) for name, records in splits.items()}
        )

    def _tokenise(self, dataset_dict: DatasetDict) -> DatasetDict:
        max_len = self.cfg.model.max_length

        def _encode(batch: Dict[str, Any]) -> Dict[str, Any]:
            encoded = self.tokenizer(
                batch["text"],
                truncation=True,
                max_length=max_len,
                padding=False,   # DataCollatorWithPadding handles per-batch padding
            )
            encoded["labels"] = [self.label_to_id[lbl] for lbl in batch["label"]]
            return encoded

        return dataset_dict.map(
            _encode,
            batched=True,
            remove_columns=["text", "label"],
            desc="Tokenising",
        )

    # ── Metrics ───────────────────────────────────────────────────────────────

    def _compute_metrics(self, eval_pred: Any) -> Dict[str, float]:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        acc       = float(accuracy_score(labels, preds))
        macro_f1  = float(f1_score(labels, preds, average="macro",  zero_division=0))
        per_class = f1_score(labels, preds, average=None, zero_division=0)

        metrics: Dict[str, float] = {"accuracy": acc, "f1": macro_f1}
        label_names = [self.id_to_label[i] for i in range(len(self.cfg.labels))]
        for name, score in zip(label_names, per_class):
            metrics[f"f1_{name.lower()}"] = float(score)

        return metrics

    def _run_test_evaluation(self, trainer: Trainer, test_dataset: Dataset) -> None:
        """Evaluate on the held-out test split and log a full classification report."""
        results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
        logger.info("Test results: %s", results)

        pred_output = trainer.predict(test_dataset)
        preds       = np.argmax(pred_output.predictions, axis=-1)
        label_names = [self.id_to_label[i] for i in range(len(self.cfg.labels))]
        report      = classification_report(
            pred_output.label_ids,
            preds,
            target_names=label_names,
            zero_division=0,
        )
        logger.info("\nClassification report (test set):\n%s", report)

    # ── TrainingArguments ─────────────────────────────────────────────────────

    def _build_training_args(self, train_size: int = 0) -> TrainingArguments:
        cfg     = self.train_cfg
        out_dir = str(Path(cfg.output_dir) / self.cfg.name / "runs")

        # Compute warmup_steps from ratio so we don't rely on the deprecated warmup_ratio param.
        steps_per_epoch = max(1, train_size // (cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps))
        total_steps     = steps_per_epoch * cfg.num_train_epochs
        warmup_steps    = max(1, round(total_steps * cfg.warmup_ratio))

        return TrainingArguments(
            output_dir=out_dir,
            num_train_epochs=cfg.num_train_epochs,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            per_device_eval_batch_size=cfg.per_device_eval_batch_size,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            warmup_steps=warmup_steps,
            fp16=cfg.fp16,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            eval_strategy=cfg.eval_strategy,
            save_strategy=cfg.save_strategy,
            load_best_model_at_end=cfg.load_best_model_at_end,
            metric_for_best_model=cfg.metric_for_best_model,
            greater_is_better=cfg.greater_is_better,
            logging_steps=cfg.logging_steps,
            save_total_limit=cfg.save_total_limit,
            seed=cfg.seed,
            report_to=cfg.report_to,
            dataloader_num_workers=cfg.dataloader_num_workers,
        )


