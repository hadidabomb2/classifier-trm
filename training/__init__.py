"""
training — TRM (Tiny Recursive Model) training package.

Public surface:
  TRMRegistry         — look up registered TRM configurations by name.
  TRMConfig           — top-level config dataclass.
  ImpactLabel         — labels for the impact TRM.
  FlavorLabel         — labels for the flavor TRM.
  PurposeLabel        — labels for the purpose TRM.
  LifespanLabel       — labels for the lifespan TRM.
  DatasetBuilder      — build and load JSONL datadumps.
  ModelManager        — model lifecycle (load / save / export / close).
  TRMTrainer          — fine-tuning engine.

Registered TRMs:
  impact              STATIC · CLARIFYING · PROVOCATIVE · TRANSFORMATIVE · TOXIC
  flavor              RAW · PROCESSED · SPICY · NOURISHING · EMPTY_CALORIES
  purpose             DIRECTIVE · PERFORMATIVE · SPECULATIVE · CONFESSIONAL · DECORATIVE
  lifespan            INSTANT · DAILY · SEASONAL · DECADAL · EVERGREEN

Quick start::

    python -m training.orchestrator --help
    python -m training.orchestrator --trm impact
    python -m training.orchestrator --trm flavor
    python -m training.orchestrator --trm purpose
    python -m training.orchestrator --trm lifespan
"""
from .config import (
    IMPACT_TRM,
    FLAVOR_TRM,
    PURPOSE_TRM,
    LIFESPAN_TRM,
    ImpactLabel,
    FlavorLabel,
    PurposeLabel,
    LifespanLabel,
    DataConfig,
    DatasetSourceConfig,
    ModelConfig,
    TrainingConfig,
    TRMConfig,
    TRMRegistry,
)
from .data_loader import DatasetBuilder
from .model_manager import ModelManager
from .trainer import TRMTrainer

__all__ = [
    # Config
    "TRMRegistry",
    "TRMConfig",
    "ImpactLabel",
    "FlavorLabel",
    "PurposeLabel",
    "LifespanLabel",
    "DatasetSourceConfig",
    "DataConfig",
    "ModelConfig",
    "TrainingConfig",
    "IMPACT_TRM",
    "FLAVOR_TRM",
    "PURPOSE_TRM",
    "LIFESPAN_TRM",
    # Pipeline components
    "DatasetBuilder",
    "ModelManager",
    "TRMTrainer",
]
