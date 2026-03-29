"""
training — TRM (Tiny Recursive Model) training package.

Public surface:
  TRMRegistry         — look up registered TRM configurations by name.
  TRMConfig           — top-level config dataclass.
  ContentLabel        — enum of classification targets.
  DatasetBuilder      — build and load JSONL datadumps.
  ModelManager        — model lifecycle (load / save / export / close).
  TRMTrainer          — fine-tuning engine.
  Orchestrator        — end-to-end pipeline coordinator.

Quick start::

    python -m training.orchestrator --help
    python -m training.orchestrator --trm content-quality
"""
from .config import (
    CONTENT_QUALITY_TRM,
    ContentLabel,
    DataConfig,
    DatasetSourceConfig,
    ModelConfig,
    TrainingConfig,
    TRMConfig,
    TRMRegistry,
)
from .data_loader import DatasetBuilder
from .model_manager import ModelManager
from .orchestrator import Orchestrator
from .trainer import TRMTrainer

__all__ = [
    # Config
    "TRMRegistry",
    "TRMConfig",
    "ContentLabel",
    "DatasetSourceConfig",
    "DataConfig",
    "ModelConfig",
    "TrainingConfig",
    "CONTENT_QUALITY_TRM",
    # Pipeline components
    "DatasetBuilder",
    "ModelManager",
    "TRMTrainer",
    "Orchestrator",
]
