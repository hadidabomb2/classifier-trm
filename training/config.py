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
# Dataset source config
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DatasetSourceConfig:
    """
    Describes a single HuggingFace dataset source and how to map its labels
    into TRM label space.

    label_mapping keys are the raw values returned by the source dataset's
    label column (can be int or str).  Values are TRM label strings.
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
    num_labels: int           = 5
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
    fp16: bool                         = False   # overridden at runtime: auto-enabled when CUDA is available
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
    dataloader_num_workers: int        = 4
    max_grad_norm: float               = 1.0


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
# TRM: impact
#
# Classifies text by how it affects the reader after reading.
# Dataset mappings are proxy approximations — see per-label comments.
# ─────────────────────────────────────────────────────────────────────────────
class ImpactLabel(str, Enum):
    STATIC         = "STATIC"         # Dead-end text — leaves you where you started
    CLARIFYING     = "CLARIFYING"     # Removes confusion, simplifies a messy idea
    PROVOCATIVE    = "PROVOCATIVE"    # Makes you angry, excited, or defensive
    TRANSFORMATIVE = "TRANSFORMATIVE" # Actually changes how you think or behave
    TOXIC          = "TOXIC"          # Spreads negativity, lies, or mental exhaustion


IMPACT_LABEL_LIST: List[str] = [lbl.value for lbl in ImpactLabel]

_IMPACT_SOURCES: List[DatasetSourceConfig] = [

    # ── STATIC ─────────────────────────────────────────────────────────────
    # Very short, low-information messages that leave you exactly where you started.
    DatasetSourceConfig(
        hf_path="ucirvine/sms_spam",
        hf_name="plain_text",
        split="train",
        text_column="sms",
        label_column="label",
        label_mapping={0: ImpactLabel.STATIC.value},  # ham = casual chitchat ("K", "On my way")
        max_samples=2_500,
        filter_min_chars=4,
        filter_max_chars=120,
    ),
    DatasetSourceConfig(
        hf_path="cardiffnlp/tweet_eval",
        hf_name="sentiment",
        split="train",
        text_column="text",
        label_column="label",
        label_mapping={1: ImpactLabel.STATIC.value},  # neutral sentiment = low-impact filler
        max_samples=2_500,
    ),

    # ── CLARIFYING ─────────────────────────────────────────────────────────
    DatasetSourceConfig(
        hf_path="wikimedia/wikipedia",
        hf_name="20231101.en",
        split="train",
        text_column="text",
        label_column=None,
        label_mapping={None: ImpactLabel.CLARIFYING.value},
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
        label_mapping={None: ImpactLabel.CLARIFYING.value},
        max_samples=2_500,
        filter_min_chars=60,
        preprocessor="last_assistant_turn",
    ),

    # ── PROVOCATIVE ────────────────────────────────────────────────────────
    DatasetSourceConfig(
        hf_path="cardiffnlp/tweet_eval",
        hf_name="offensive",
        split="train",
        text_column="text",
        label_column="label",
        label_mapping={1: ImpactLabel.PROVOCATIVE.value},
        max_samples=2_500,
    ),
    DatasetSourceConfig(
        hf_path="mediabiasgroup/BABE",
        hf_name=None,
        split="train",
        text_column="text",
        label_column="label",
        label_mapping={1: ImpactLabel.PROVOCATIVE.value},  # biased articles provoke a reaction
        max_samples=2_500,
    ),
    DatasetSourceConfig(
        hf_path="dair-ai/emotion",
        hf_name=None,
        split="train",
        text_column="text",
        label_column="label",
        label_mapping={3: ImpactLabel.PROVOCATIVE.value},  # anger tweets = provoke a strong reaction
        max_samples=2_500,
    ),

    # ── TRANSFORMATIVE ─────────────────────────────────────────────────────
    DatasetSourceConfig(
        hf_path="Anthropic/hh-rlhf",
        hf_name=None,
        hf_kwargs={"data_dir": "helpful-online"},
        split="train",
        text_column="chosen",
        label_column=None,
        label_mapping={None: ImpactLabel.TRANSFORMATIVE.value},
        max_samples=2_500,
        filter_min_chars=100,
        preprocessor="last_assistant_turn",
    ),
    DatasetSourceConfig(
        hf_path="yelp_review_full",
        hf_name=None,
        split="train",
        text_column="text",
        label_column="label",
        label_mapping={4: ImpactLabel.TRANSFORMATIVE.value},  # detailed 5-star reviews describing life impact
        max_samples=2_500,
        filter_min_chars=200,
    ),

    # ── TOXIC ──────────────────────────────────────────────────────────────
    DatasetSourceConfig(
        hf_path="GonzaloA/fake_news",
        hf_name=None,
        split="train",
        text_column="text",
        label_column="label",
        label_mapping={1: ImpactLabel.TOXIC.value},
        max_samples=2_500,
        filter_min_chars=80,
    ),
    DatasetSourceConfig(
        hf_path="cardiffnlp/tweet_eval",
        hf_name="hate",
        split="train",
        text_column="text",
        label_column="label",
        label_mapping={1: ImpactLabel.TOXIC.value},
        max_samples=2_500,
    ),
]

IMPACT_TRM: TRMConfig = TRMRegistry.register(
    TRMConfig(
        name="impact",
        description=(
            "Five-class impact classifier: how text affects you after reading. "
            "STATIC (dead-end), CLARIFYING (removes confusion), PROVOCATIVE (triggers reaction), "
            "TRANSFORMATIVE (changes thinking), TOXIC (spreads negativity/lies)."
        ),
        labels=IMPACT_LABEL_LIST,
        model=ModelConfig(
            model_name="distilbert/distilbert-base-uncased",
            num_labels=len(ImpactLabel),
            max_length=256,
        ),
        data=DataConfig(sources=_IMPACT_SOURCES, max_samples_per_label=5_000),
        training=TrainingConfig(
            num_train_epochs=5,
            per_device_train_batch_size=16,
            learning_rate=2e-5,
        ),
    )
)


# ─────────────────────────────────────────────────────────────────────────────
# TRM: flavor
#
# Classifies the sensory "feel" of text — like food groups.
# ─────────────────────────────────────────────────────────────────────────────
class FlavorLabel(str, Enum):
    RAW            = "RAW"            # Unfiltered, messy thoughts (journals, voice-to-text)
    PROCESSED      = "PROCESSED"      # Clean and professional but soulless (press releases)
    SPICY          = "SPICY"          # Controversial, edgy, or intentionally bold
    NOURISHING     = "NOURISHING"     # Deep, thoughtful, healthy for your perspective
    EMPTY_CALORIES = "EMPTY_CALORIES" # Entertaining but mindless (gossip, clickbait)


FLAVOR_LABEL_LIST: List[str] = [lbl.value for lbl in FlavorLabel]

_FLAVOR_SOURCES: List[DatasetSourceConfig] = [

    # ── RAW ────────────────────────────────────────────────────────────────
    DatasetSourceConfig(
        hf_path="dair-ai/emotion",
        hf_name=None,
        split="train",
        text_column="text",
        label_column=None,
        label_mapping={None: FlavorLabel.RAW.value},  # raw emotional Twitter posts
        max_samples=2_500,
    ),
    DatasetSourceConfig(
        hf_path="cardiffnlp/tweet_eval",
        hf_name="irony",
        split="train",
        text_column="text",
        label_column="label",
        label_mapping={1: FlavorLabel.RAW.value},  # ironic tweets = unpolished, unfiltered expression
        max_samples=2_500,
    ),
    DatasetSourceConfig(
        hf_path="cardiffnlp/tweet_eval",
        hf_name="sentiment",
        split="train",
        text_column="text",
        label_column="label",
        label_mapping={0: FlavorLabel.RAW.value},  # negative emotional tweets = raw unfiltered expression
        max_samples=2_500,
    ),

    # ── PROCESSED ──────────────────────────────────────────────────────────
    DatasetSourceConfig(
        hf_path="SetFit/enron_spam",
        hf_name=None,
        split="train",
        text_column="text",
        label_column="label",
        label_mapping={0: FlavorLabel.PROCESSED.value},  # ham = corporate emails, clean but lifeless
        max_samples=2_500,
    ),
    DatasetSourceConfig(
        hf_path="fancyzhx/ag_news",
        hf_name=None,
        split="train",
        text_column="text",
        label_column=None,
        label_mapping={None: FlavorLabel.PROCESSED.value},  # wire-service news = polished but formulaic
        max_samples=2_500,
        filter_min_chars=80,
    ),

    # ── SPICY ──────────────────────────────────────────────────────────────
    DatasetSourceConfig(
        hf_path="cardiffnlp/tweet_eval",
        hf_name="offensive",
        split="train",
        text_column="text",
        label_column="label",
        label_mapping={1: FlavorLabel.SPICY.value},
        max_samples=2_500,
    ),
    DatasetSourceConfig(
        hf_path="mediabiasgroup/BABE",
        hf_name=None,
        split="train",
        text_column="text",
        label_column="label",
        label_mapping={1: FlavorLabel.SPICY.value},  # biased articles = intentionally bold takes
        max_samples=2_500,
    ),
    DatasetSourceConfig(
        hf_path="cardiffnlp/tweet_eval",
        hf_name="hate",
        split="train",
        text_column="text",
        label_column="label",
        label_mapping={1: FlavorLabel.SPICY.value},  # hate speech = deliberately bold/edgy content
        max_samples=2_500,
    ),

    # ── NOURISHING ─────────────────────────────────────────────────────────
    DatasetSourceConfig(
        hf_path="Anthropic/hh-rlhf",
        hf_name=None,
        hf_kwargs={"data_dir": "helpful-base"},
        split="train",
        text_column="chosen",
        label_column=None,
        label_mapping={None: FlavorLabel.NOURISHING.value},
        max_samples=2_500,
        filter_min_chars=60,
        preprocessor="last_assistant_turn",
    ),
    DatasetSourceConfig(
        hf_path="wikimedia/wikipedia",
        hf_name="20231101.en",
        split="train",
        text_column="text",
        label_column=None,
        label_mapping={None: FlavorLabel.NOURISHING.value},
        max_samples=2_500,
        streaming=True,
        filter_min_chars=200,
        filter_max_chars=1_500,
        preprocessor="first_paragraph",
    ),

    # ── EMPTY_CALORIES ─────────────────────────────────────────────────────
    DatasetSourceConfig(
        hf_path="yelp_review_full",
        hf_name=None,
        split="train",
        text_column="text",
        label_column="label",
        label_mapping={4: FlavorLabel.EMPTY_CALORIES.value},  # breathless 5-star gushing
        max_samples=2_500,
        filter_min_chars=40,
        filter_max_chars=400,
    ),
    DatasetSourceConfig(
        hf_path="GonzaloA/fake_news",
        hf_name=None,
        split="train",
        text_column="text",
        label_column="label",
        label_mapping={1: FlavorLabel.EMPTY_CALORIES.value},  # clickbait fake news
        max_samples=2_500,
        filter_min_chars=80,
    ),
]

FLAVOR_TRM: TRMConfig = TRMRegistry.register(
    TRMConfig(
        name="flavor",
        description=(
            "Five-class flavor classifier: the sensory feel of text. "
            "RAW (unfiltered), PROCESSED (soulless clean), SPICY (intentionally bold), "
            "NOURISHING (deep/thoughtful), EMPTY_CALORIES (entertaining but mindless)."
        ),
        labels=FLAVOR_LABEL_LIST,
        model=ModelConfig(
            model_name="distilbert/distilbert-base-uncased",
            num_labels=len(FlavorLabel),
            max_length=256,
        ),
        data=DataConfig(sources=_FLAVOR_SOURCES, max_samples_per_label=5_000),
        training=TrainingConfig(
            num_train_epochs=5,
            per_device_train_batch_size=16,
            learning_rate=2e-5,
        ),
    )
)


# ─────────────────────────────────────────────────────────────────────────────
# TRM: purpose
#
# Classifies the intent behind the writing.
# ─────────────────────────────────────────────────────────────────────────────
class PurposeLabel(str, Enum):
    DIRECTIVE    = "DIRECTIVE"    # Telling you what to do (manuals, laws, signs)
    PERFORMATIVE = "PERFORMATIVE" # Showing off or building a brand (CVs, social bios)
    SPECULATIVE  = "SPECULATIVE"  # Asking "What if?" (sci-fi, philosophy, theories)
    CONFESSIONAL = "CONFESSIONAL" # Revealing a personal truth (memoirs, letters)
    DECORATIVE   = "DECORATIVE"   # Just there to look pretty (greeting cards, filler)


PURPOSE_LABEL_LIST: List[str] = [lbl.value for lbl in PurposeLabel]

_PURPOSE_SOURCES: List[DatasetSourceConfig] = [

    # ── DIRECTIVE ──────────────────────────────────────────────────────────
    DatasetSourceConfig(
        hf_path="tatsu-lab/alpaca",
        hf_name=None,
        split="train",
        text_column="instruction",
        label_column=None,
        label_mapping={None: PurposeLabel.DIRECTIVE.value},  # instruction-following prompts
        max_samples=5_000,
        filter_min_chars=20,
        filter_max_chars=500,
    ),
    DatasetSourceConfig(
        hf_path="ucirvine/sms_spam",
        hf_name="plain_text",
        split="train",
        text_column="sms",
        label_column="label",
        label_mapping={1: PurposeLabel.DIRECTIVE.value},  # spam = directive ("call now", "buy this")
        max_samples=2_500,
    ),

    # ── PERFORMATIVE ───────────────────────────────────────────────────────
    DatasetSourceConfig(
        hf_path="yelp_review_full",
        hf_name=None,
        split="train",
        text_column="text",
        label_column="label",
        label_mapping={4: PurposeLabel.PERFORMATIVE.value},  # performative 5-star gushing
        max_samples=2_500,
        filter_max_chars=400,
    ),
    DatasetSourceConfig(
        hf_path="Anthropic/hh-rlhf",
        hf_name=None,
        hf_kwargs={"data_dir": "helpful-base"},
        split="train",
        text_column="rejected",
        label_column=None,
        label_mapping={None: PurposeLabel.PERFORMATIVE.value},  # rejected = try-hard, showing off
        max_samples=2_500,
        filter_min_chars=60,
        preprocessor="last_assistant_turn",
    ),

    # ── SPECULATIVE ────────────────────────────────────────────────────────
    DatasetSourceConfig(
        hf_path="Anthropic/hh-rlhf",
        hf_name=None,
        hf_kwargs={"data_dir": "helpful-online"},
        split="train",
        text_column="chosen",
        label_column=None,
        label_mapping={None: PurposeLabel.SPECULATIVE.value},  # online discussions = often hypothetical
        max_samples=2_500,
        filter_min_chars=80,
        preprocessor="last_assistant_turn",
    ),
    DatasetSourceConfig(
        hf_path="wikimedia/wikipedia",
        hf_name="20231101.en",
        split="train",
        text_column="text",
        label_column=None,
        label_mapping={None: PurposeLabel.SPECULATIVE.value},  # encyclopedic theory / science articles
        max_samples=2_500,
        streaming=True,
        filter_min_chars=200,
        filter_max_chars=1_500,
        preprocessor="first_paragraph",
    ),

    # ── CONFESSIONAL ───────────────────────────────────────────────────────
    DatasetSourceConfig(
        hf_path="dair-ai/emotion",
        hf_name=None,
        split="train",
        text_column="text",
        label_column=None,
        label_mapping={None: PurposeLabel.CONFESSIONAL.value},  # raw personal emotional expression
        max_samples=2_500,
    ),
    DatasetSourceConfig(
        hf_path="cardiffnlp/tweet_eval",
        hf_name="sentiment",
        split="train",
        text_column="text",
        label_column="label",
        label_mapping={0: PurposeLabel.CONFESSIONAL.value},  # negative sentiment = confessional outpouring
        max_samples=2_500,
    ),

    # ── DECORATIVE ─────────────────────────────────────────────────────────
    DatasetSourceConfig(
        hf_path="cardiffnlp/tweet_eval",
        hf_name="irony",
        split="train",
        text_column="text",
        label_column="label",
        label_mapping={0: PurposeLabel.DECORATIVE.value},  # non-ironic tweets = light chitchat filler
        max_samples=2_500,
    ),
    DatasetSourceConfig(
        hf_path="ucirvine/sms_spam",
        hf_name="plain_text",
        split="train",
        text_column="sms",
        label_column="label",
        label_mapping={0: PurposeLabel.DECORATIVE.value},  # ham small talk = decorative noise
        max_samples=2_500,
        filter_max_chars=80,
    ),
    DatasetSourceConfig(
        hf_path="cardiffnlp/tweet_eval",
        hf_name="sentiment",
        split="train",
        text_column="text",
        label_column="label",
        label_mapping={2: PurposeLabel.DECORATIVE.value},  # positive casual tweets = light decorative content
        max_samples=2_500,
    ),
]

PURPOSE_TRM: TRMConfig = TRMRegistry.register(
    TRMConfig(
        name="purpose",
        description=(
            "Five-class purpose classifier: the intent behind the writing. "
            "DIRECTIVE (commands/instructions), PERFORMATIVE (showing off), "
            "SPECULATIVE (what-if thinking), CONFESSIONAL (personal truth), DECORATIVE (filler)."
        ),
        labels=PURPOSE_LABEL_LIST,
        model=ModelConfig(
            model_name="distilbert/distilbert-base-uncased",
            num_labels=len(PurposeLabel),
            max_length=256,
        ),
        data=DataConfig(sources=_PURPOSE_SOURCES, max_samples_per_label=5_000),
        training=TrainingConfig(
            num_train_epochs=5,
            per_device_train_batch_size=16,
            learning_rate=2e-5,
        ),
    )
)


# ─────────────────────────────────────────────────────────────────────────────
# TRM: lifespan
#
# Classifies how long text stays "fresh" before it expires.
# ─────────────────────────────────────────────────────────────────────────────
class LifespanLabel(str, Enum):
    INSTANT  = "INSTANT"  # Expires in seconds (OTPs, "I'm here", quick reactions)
    DAILY    = "DAILY"    # Useless by tomorrow (breaking news, weather, scores)
    SEASONAL = "SEASONAL" # Good for a few months (trend reports, local reviews)
    DECADAL  = "DECADAL"  # Relevant for years (career advice, how-to guides, laws)
    EVERGREEN = "EVERGREEN" # Never dies (encyclopedic knowledge, timeless wisdom)


LIFESPAN_LABEL_LIST: List[str] = [lbl.value for lbl in LifespanLabel]

_LIFESPAN_SOURCES: List[DatasetSourceConfig] = [

    # ── INSTANT ────────────────────────────────────────────────────────────
    DatasetSourceConfig(
        hf_path="ucirvine/sms_spam",
        hf_name="plain_text",
        split="train",
        text_column="sms",
        label_column="label",
        label_mapping={0: LifespanLabel.INSTANT.value},  # very short ham = "K", "Be there in 5"
        max_samples=2_500,
        filter_min_chars=4,
        filter_max_chars=60,
    ),
    DatasetSourceConfig(
        hf_path="dair-ai/emotion",
        hf_name=None,
        split="train",
        text_column="text",
        label_column=None,
        label_mapping={None: LifespanLabel.INSTANT.value},  # in-the-moment emotional reactions
        max_samples=2_500,
        filter_max_chars=80,
    ),

    # ── DAILY ──────────────────────────────────────────────────────────────
    DatasetSourceConfig(
        hf_path="fancyzhx/ag_news",
        hf_name=None,
        split="train",
        text_column="text",
        label_column=None,
        label_mapping={None: LifespanLabel.DAILY.value},  # breaking news = expires tomorrow
        max_samples=2_500,
        filter_min_chars=80,
    ),
    DatasetSourceConfig(
        hf_path="GonzaloA/fake_news",
        hf_name=None,
        split="train",
        text_column="text",
        label_column="label",
        label_mapping={0: LifespanLabel.DAILY.value},  # real news articles = daily freshness
        max_samples=2_500,
        filter_min_chars=80,
    ),

    # ── SEASONAL ───────────────────────────────────────────────────────────
    DatasetSourceConfig(
        hf_path="yelp_review_full",
        hf_name=None,
        split="train",
        text_column="text",
        label_column=None,
        label_mapping={None: LifespanLabel.SEASONAL.value},  # restaurant/product reviews = seasonally relevant
        max_samples=2_500,
        filter_min_chars=80,
        filter_max_chars=600,
    ),
    DatasetSourceConfig(
        hf_path="mediabiasgroup/BABE",
        hf_name=None,
        split="train",
        text_column="text",
        label_column=None,
        label_mapping={None: LifespanLabel.SEASONAL.value},  # topical journalism = month-scale relevance
        max_samples=2_500,
    ),

    # ── DECADAL ────────────────────────────────────────────────────────────
    DatasetSourceConfig(
        hf_path="tatsu-lab/alpaca",
        hf_name=None,
        split="train",
        text_column="output",
        label_column=None,
        label_mapping={None: LifespanLabel.DECADAL.value},  # how-to answers = relevant for years
        max_samples=2_500,
        filter_min_chars=60,
    ),
    DatasetSourceConfig(
        hf_path="Anthropic/hh-rlhf",
        hf_name=None,
        hf_kwargs={"data_dir": "helpful-base"},
        split="train",
        text_column="chosen",
        label_column=None,
        label_mapping={None: LifespanLabel.DECADAL.value},  # expert Q&A advice = multi-year relevance
        max_samples=2_500,
        filter_min_chars=80,
        preprocessor="last_assistant_turn",
    ),

    # ── EVERGREEN ──────────────────────────────────────────────────────────
    DatasetSourceConfig(
        hf_path="wikimedia/wikipedia",
        hf_name="20231101.en",
        split="train",
        text_column="text",
        label_column=None,
        label_mapping={None: LifespanLabel.EVERGREEN.value},
        max_samples=2_500,
        streaming=True,
        filter_min_chars=200,
        filter_max_chars=1_500,
        preprocessor="first_paragraph",
    ),
    DatasetSourceConfig(
        hf_path="Anthropic/hh-rlhf",
        hf_name=None,
        hf_kwargs={"data_dir": "helpful-online"},
        split="train",
        text_column="chosen",
        label_column=None,
        label_mapping={None: LifespanLabel.EVERGREEN.value},  # timeless practical knowledge
        max_samples=2_500,
        filter_min_chars=80,
        preprocessor="last_assistant_turn",
    ),
]

LIFESPAN_TRM: TRMConfig = TRMRegistry.register(
    TRMConfig(
        name="lifespan",
        description=(
            "Five-class lifespan classifier: how long text stays fresh. "
            "INSTANT (seconds), DAILY (expires tomorrow), SEASONAL (months), "
            "DECADAL (years), EVERGREEN (never dies)."
        ),
        labels=LIFESPAN_LABEL_LIST,
        model=ModelConfig(
            model_name="distilbert/distilbert-base-uncased",
            num_labels=len(LifespanLabel),
            max_length=256,
        ),
        data=DataConfig(sources=_LIFESPAN_SOURCES, max_samples_per_label=5_000),
        training=TrainingConfig(
            num_train_epochs=5,
            per_device_train_batch_size=16,
            learning_rate=2e-5,
        ),
    )
)
