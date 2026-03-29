"""
config.py

Central configuration and constants for all TRM (Tiny Recursive Model) pipelines.

Design pattern:
  - Every pipeline stage reads from a TRMConfig instance.
  - TRMRegistry allows multiple TRMs to coexist and be selected by name.
  - To add a second TRM, create a new TRMConfig and call TRMRegistry.register().
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Root
# ─────────────────────────────────────────────────────────────────────────────
TRAINING_ROOT: Path = Path(__file__).parent


# ─────────────────────────────────────────────────────────────────────────────
# Labels
# ─────────────────────────────────────────────────────────────────────────────
class ContentLabel(str, Enum):
    """Classification targets for the content-quality TRM."""
    SLOP   = "SLOP"    # Low-effort filler, generic, padding, copy-paste
    CRINGE = "CRINGE"  # Embarrassing, try-hard, socially awkward
    BOT    = "BOT"     # Automated, templated, spam-like
    STUPID = "STUPID"  # Misinformation, conspiracy, factually wrong, incoherent
    CLEAN  = "CLEAN"   # High quality, genuine, informative, well-written


CONTENT_LABEL_LIST: List[str] = [lbl.value for lbl in ContentLabel]
CONTENT_LABEL_TO_ID: Dict[str, int] = {lbl.value: i for i, lbl in enumerate(ContentLabel)}
CONTENT_ID_TO_LABEL: Dict[int, str] = {i: lbl.value for i, lbl in enumerate(ContentLabel)}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset source config
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DatasetSourceConfig:
    """
    Describes a single HuggingFace dataset source and how to map its labels
    into TRM label space.

    label_mapping keys are the raw values returned by the source dataset's
    label column (can be int or str).  Values are ContentLabel.value strings.
    A key of None means the label is fixed (use when label_column is None).

    hf_kwargs can pass extra keyword arguments straight to load_dataset(),
    e.g. {"data_dir": "helpful-base"} for Anthropic/hh-rlhf sub-splits.
    """
    hf_path: str
    split: str
    text_column: str
    label_mapping: Dict[Any, str]
    hf_name: Optional[str]        = None
    hf_kwargs: Dict[str, Any]     = field(default_factory=dict)
    label_column: Optional[str]   = None   # None → use label_mapping[None] as fixed label
    max_samples: int               = 2_000
    streaming: bool                = False
    filter_min_chars: int          = 40
    filter_max_chars: int          = 2_048
    preprocessor: Optional[str]    = None  # name of a function in data_loader._PREPROCESSORS


# ─────────────────────────────────────────────────────────────────────────────
# Sub-configs
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DataConfig:
    """Paths and sampling parameters for the data pipeline."""
    raw_dir: Path                        = TRAINING_ROOT / "data" / "raw"
    datadump_dir: Path                   = TRAINING_ROOT / "data" / "datadumps"
    processed_dir: Path                  = TRAINING_ROOT / "data" / "processed"
    max_samples_per_label: int           = 5_000
    val_split: float                     = 0.1
    test_split: float                    = 0.1
    seed: int                            = 42
    sources: List[DatasetSourceConfig]   = field(default_factory=list)


@dataclass
class ModelConfig:
    """HuggingFace model identity and tokenisation settings."""
    model_name: str           = "distilbert/distilbert-base-uncased"
    num_labels: int           = len(ContentLabel)
    max_length: int           = 256
    cache_dir: Optional[Path] = TRAINING_ROOT / ".cache" / "models"


@dataclass
class TrainingConfig:
    """HuggingFace TrainingArguments equivalents and extra knobs."""
    output_dir: Path                   = TRAINING_ROOT / "outputs"
    checkpoint_dir: Path               = TRAINING_ROOT / "checkpoints"
    num_train_epochs: int              = 5
    per_device_train_batch_size: int   = 16
    per_device_eval_batch_size: int    = 32
    learning_rate: float               = 2e-5
    weight_decay: float                = 0.01
    warmup_ratio: float                = 0.1
    fp16: bool                         = False   # set True if CUDA GPU is available
    gradient_accumulation_steps: int   = 1
    eval_strategy: str                 = "epoch"
    save_strategy: str                 = "epoch"
    load_best_model_at_end: bool       = True
    metric_for_best_model: str         = "f1"
    greater_is_better: bool            = True
    early_stopping_patience: int       = 3
    logging_steps: int                 = 50
    save_total_limit: int              = 3
    seed: int                          = 42
    report_to: str                     = "none"  # "wandb" | "tensorboard" | "none"
    dataloader_num_workers: int        = 0


# ─────────────────────────────────────────────────────────────────────────────
# Top-level TRM config
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TRMConfig:
    """
    Complete configuration for one Tiny Recursive Model.

    Compose ModelConfig, DataConfig, and TrainingConfig here.
    The TRMRegistry keyed on 'name' allows multiple TRMs to coexist.
    """
    name: str
    description: str
    labels: List[str]
    model: ModelConfig       = field(default_factory=ModelConfig)
    data: DataConfig         = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────
class TRMRegistry:
    """
    Singleton registry.  Call TRMRegistry.register(cfg) once per TRM,
    then retrieve with TRMRegistry.get("my-trm-name").
    """
    _registry: Dict[str, TRMConfig] = {}

    @classmethod
    def register(cls, config: TRMConfig) -> TRMConfig:
        if config.name in cls._registry:
            raise ValueError(f"TRM '{config.name}' is already registered.")
        cls._registry[config.name] = config
        return config

    @classmethod
    def get(cls, name: str) -> TRMConfig:
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise KeyError(
                f"TRM '{name}' not found. Registered TRMs: {available}"
            )
        return cls._registry[name]

    @classmethod
    def list_names(cls) -> List[str]:
        return list(cls._registry.keys())

    @classmethod
    def all(cls) -> Dict[str, TRMConfig]:
        return dict(cls._registry)


# ─────────────────────────────────────────────────────────────────────────────
# Pre-configured TRM: content-quality
#
# Dataset strategy (proxy mappings — reviewed sources, graceful skip on load
# failure enforced in data_loader.py):
#
#   CLEAN   wikimedia/wikipedia          first paragraph of article
#   CLEAN   Anthropic/hh-rlhf chosen    last assistant turn (helpful-base)
#   BOT     ucirvine/sms_spam            spam SMS messages (ClassLabel int 1)
#   BOT     SetFit/enron_spam            spam email (ClassLabel int 1)
#   BOT     Deysi/spam-detection-dataset spam emails (graceful skip if 404)
#   SLOP    Anthropic/hh-rlhf rejected  last assistant turn — low-quality
#   SLOP    mediabiasgroup/BABE          biased/sensationalist articles
#   SLOP    yelp_review_full             1-star reviews (label 0) — low-effort
#   STUPID  GonzaloA/fake_news           fake news articles (label 1)
#   STUPID  cardiffnlp/tweet_eval hate   hate speech tweets (label 1)
#   CRINGE  cardiffnlp/tweet_eval offnsv offensive tweets (label 1)
#   CRINGE  cardiffnlp/tweet_eval irony  ironic tweets (label 1)
#   CRINGE  dair-ai/emotion              anger tweets (label 3)
# ─────────────────────────────────────────────────────────────────────────────
_CONTENT_SOURCES: List[DatasetSourceConfig] = [

    # ── CLEAN ──────────────────────────────────────────────────────────────
    DatasetSourceConfig(
        hf_path="wikimedia/wikipedia",
        hf_name="20231101.en",
        split="train",
        text_column="text",
        label_column=None,
        label_mapping={None: ContentLabel.CLEAN.value},
        max_samples=2_500,
        streaming=True,
        filter_min_chars=200,
        filter_max_chars=1_500,
        preprocessor="first_paragraph",
    ),
    DatasetSourceConfig(
        hf_path="Anthropic/hh-rlhf",
        hf_name=None,
        hf_kwargs={"data_dir": "helpful-base"},
        split="train",
        text_column="chosen",
        label_column=None,
        label_mapping={None: ContentLabel.CLEAN.value},
        max_samples=2_500,
        filter_min_chars=60,
        preprocessor="last_assistant_turn",
    ),

    # ── BOT ────────────────────────────────────────────────────────────────
    DatasetSourceConfig(
        hf_path="ucirvine/sms_spam",
        hf_name="plain_text",
        split="train",
        text_column="sms",
        label_column="label",
        # ClassLabel integers: 0 = ham, 1 = spam
        label_mapping={1: ContentLabel.BOT.value},
        max_samples=2_500,
    ),
    DatasetSourceConfig(
        hf_path="SetFit/enron_spam",
        hf_name=None,
        split="train",
        text_column="text",
        label_column="label",
        # 0 = ham, 1 = spam
        label_mapping={1: ContentLabel.BOT.value},
        max_samples=2_500,
    ),
    DatasetSourceConfig(
        hf_path="Deysi/spam-detection-dataset",
        hf_name=None,
        split="train",
        text_column="text",
        label_column="label",
        # String or int label — try both keys; graceful skip if dataset 404s
        label_mapping={"spam": ContentLabel.BOT.value, 1: ContentLabel.BOT.value},
        max_samples=2_500,
    ),

    # ── SLOP ───────────────────────────────────────────────────────────────
    DatasetSourceConfig(
        hf_path="Anthropic/hh-rlhf",
        hf_name=None,
        hf_kwargs={"data_dir": "helpful-base"},
        split="train",
        text_column="rejected",
        label_column=None,
        label_mapping={None: ContentLabel.SLOP.value},
        max_samples=2_500,
        filter_min_chars=60,
        preprocessor="last_assistant_turn",
    ),
    DatasetSourceConfig(
        hf_path="mediabiasgroup/BABE",
        hf_name=None,
        split="train",
        text_column="text",
        label_column="label",
        # Label 1 = biased — biased / sensationalist text is often slop
        label_mapping={1: ContentLabel.SLOP.value},
        max_samples=2_500,
    ),
    DatasetSourceConfig(
        hf_path="yelp_review_full",
        hf_name=None,
        split="train",
        text_column="text",
        label_column="label",
        # ClassLabel: 0 = 1-star, 4 = 5-star; 1-star reviews are low-effort SLOP
        label_mapping={0: ContentLabel.SLOP.value},
        max_samples=2_500,
        filter_min_chars=40,
        filter_max_chars=1_024,
    ),

    # ── STUPID ─────────────────────────────────────────────────────────────
    DatasetSourceConfig(
        hf_path="GonzaloA/fake_news",
        hf_name=None,
        split="train",
        text_column="text",
        label_column="label",
        label_mapping={1: ContentLabel.STUPID.value},
        max_samples=2_500,
        filter_min_chars=80,
    ),
    DatasetSourceConfig(
        hf_path="cardiffnlp/tweet_eval",
        hf_name="hate",
        split="train",
        text_column="text",
        label_column="label",
        label_mapping={1: ContentLabel.STUPID.value},
        max_samples=2_500,
    ),

    # ── CRINGE ─────────────────────────────────────────────────────────────
    DatasetSourceConfig(
        hf_path="cardiffnlp/tweet_eval",
        hf_name="offensive",
        split="train",
        text_column="text",
        label_column="label",
        label_mapping={1: ContentLabel.CRINGE.value},
        max_samples=2_500,
    ),
    DatasetSourceConfig(
        hf_path="cardiffnlp/tweet_eval",
        hf_name="irony",
        split="train",
        text_column="text",
        label_column="label",
        label_mapping={1: ContentLabel.CRINGE.value},
        max_samples=2_500,
    ),
    DatasetSourceConfig(
        hf_path="dair-ai/emotion",
        hf_name=None,
        split="train",
        text_column="text",
        label_column="label",
        # ClassLabel: 0=sadness 1=joy 2=love 3=anger 4=fear 5=surprise
        # Anger-driven posts map well to CRINGE (aggressive try-hard content)
        label_mapping={3: ContentLabel.CRINGE.value},
        max_samples=2_500,
    ),
]

_CONTENT_DATA_CFG = DataConfig(
    sources=_CONTENT_SOURCES,
    max_samples_per_label=5_000,
)

# Register at module import time
CONTENT_QUALITY_TRM: TRMConfig = TRMRegistry.register(
    TRMConfig(
        name="content-quality",
        description=(
            "Five-class content classifier: SLOP (low-effort filler), "
            "CRINGE (embarrassing/try-hard), BOT (automated/spam), "
            "STUPID (misinformation/incoherent), CLEAN (high quality)."
        ),
        labels=CONTENT_LABEL_LIST,
        model=ModelConfig(
            model_name="distilbert/distilbert-base-uncased",
            num_labels=len(ContentLabel),
            max_length=256,
        ),
        data=_CONTENT_DATA_CFG,
        training=TrainingConfig(
            num_train_epochs=5,
            per_device_train_batch_size=16,
            learning_rate=2e-5,
        ),
    )
)
