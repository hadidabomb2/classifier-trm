# classifier-trm

Local TRM (Tiny Recursive Model) training and inference pipeline.

## What it does

Hosts four fine-tuned DistilBERT (67M param) classifiers, each analysing text from a different angle:

| TRM | `--trm` name | Labels |
|-----|-------------|--------|
| Reader impact | `impact` | `STATIC` · `CLARIFYING` · `PROVOCATIVE` · `TRANSFORMATIVE` · `TOXIC` |
| Sensory flavor | `flavor` | `RAW` · `PROCESSED` · `SPICY` · `NOURISHING` · `EMPTY_CALORIES` |
| Writing purpose | `purpose` | `DIRECTIVE` · `PERFORMATIVE` · `SPECULATIVE` · `CONFESSIONAL` · `DECORATIVE` |
| Content lifespan | `lifespan` | `INSTANT` · `DAILY` · `SEASONAL` · `DECADAL` · `EVERGREEN` |

### Label definitions

**impact** — how it affects you after reading
| Label | Meaning |
|-------|---------|
| `STATIC` | Dead-end — leaves you exactly where you started |
| `CLARIFYING` | Removes confusion, simplifies a messy idea |
| `PROVOCATIVE` | Makes you angry, excited, or defensive |
| `TRANSFORMATIVE` | Actually changes how you think or behave |
| `TOXIC` | Spreads negativity, lies, or mental exhaustion |

**flavor** — the sensory feel (text as food)
| Label | Meaning |
|-------|---------|
| `RAW` | Unfiltered, messy thoughts — journals, voice-to-text, rough drafts |
| `PROCESSED` | Clean and professional but soulless — press releases, HR emails |
| `SPICY` | Controversial, edgy, or intentionally bold |
| `NOURISHING` | Deep, thoughtful, healthy for your perspective |
| `EMPTY_CALORIES` | Entertaining but mindless — gossip, clickbait, brainrot |

**purpose** — the intent behind the writing
| Label | Meaning |
|-------|---------|
| `DIRECTIVE` | Telling you what to do — manuals, laws, signs |
| `PERFORMATIVE` | Showing off or building a brand — social bios, CVs |
| `SPECULATIVE` | Asking "What if?" — sci-fi, philosophy, theories |
| `CONFESSIONAL` | Revealing a personal truth — memoirs, letters, secrets |
| `DECORATIVE` | Just there to look pretty — greeting cards, filler |

**lifespan** — how long the text stays fresh
| Label | Meaning |
|-------|---------|
| `INSTANT` | Expires in seconds — OTPs, "I'm here", quick reactions |
| `DAILY` | Useless by tomorrow — breaking news, weather reports |
| `SEASONAL` | Good for a few months — trend reports, local reviews |
| `DECADAL` | Relevant for years — career advice, how-to guides |
| `EVERGREEN` | Never dies — encyclopedic knowledge, timeless wisdom |

---

## Project structure

```
classifier-trm/
├── training/
│   ├── config.py          # All config, constants, TRMRegistry
│   ├── utils.py           # Logging, device detection, JSONL helpers
│   ├── data_loader.py     # Dataset download, preprocessing, datadumps
│   ├── model_manager.py   # Model lifecycle: load / save / export / close
│   ├── trainer.py         # Fine-tuning engine (HuggingFace Trainer)
│   ├── orchestrator.py    # CLI + pipeline coordinator
│   ├── __init__.py        # Public API surface
│   └── requirements.txt   # Python dependencies
│
├── training/data/
│   ├── raw/               # Per-source JSONL caches (gitignored)
│   └── datadumps/         # Merged balanced JSONL used for training (gitignored)
│
├── training/checkpoints/  # Versioned + best/ checkpoints (gitignored)
├── training/outputs/      # Final exported model (gitignored)
├── training/.cache/       # HuggingFace model cache (gitignored)
│
├── scanner/
│   ├── config.py          # ScannerConfig dataclass + label styles
│   ├── capture.py         # Screen capture via mss → PIL Image
│   ├── ocr.py             # RapidOCR wrapper + 4-layer filter pipeline
│   ├── classifier.py      # HuggingFace pipeline wrapper → label + scores
│   ├── overlay.py         # Frameless tkinter always-on-top overlay
│   ├── scanner.py         # Orchestrator + CLI (main entry point)
│   ├── __init__.py        # Public API: Scanner, ScannerConfig
│   ├── __main__.py        # python -m scanner entry point
│   ├── requirements.txt   # mss, rapidocr-onnxruntime, Pillow
│   └── filters/
│       ├── width_ratio.py # L1: drop narrow detections
│       ├── word_count.py  # L2: drop single-word detections
│       ├── stopword.py    # L3: reject UI label blocks
│       └── dedup.py       # L4: skip unchanged frames
```

---

## Training

### Setup

```bash
pip install -r training/requirements.txt
```

### Run the full pipeline

```bash
# Train a specific TRM (all stages: data → train → eval → export)
python -m training.orchestrator --trm impact
python -m training.orchestrator --trm flavor
python -m training.orchestrator --trm purpose
python -m training.orchestrator --trm lifespan
```

### Run individual stages

```bash
# 1. Download datasets and build datadumps only
python -m training.orchestrator --trm impact --steps data

# 2. Train the model
python -m training.orchestrator --trm impact --steps train

# 3. Evaluate the best checkpoint on the test set
python -m training.orchestrator --trm impact --steps eval

# 4. Export the final model to outputs/
python -m training.orchestrator --trm impact --steps export

# Force re-download all datasets
python -m training.orchestrator --trm flavor --steps data --force-rebuild

# Resume training from a checkpoint
python -m training.orchestrator --trm lifespan --steps train --checkpoint checkpoints/lifespan/epoch-5
```

### CLI reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--trm NAME` | `impact` | TRM to train (see `TRMRegistry`) |
| `--steps STAGE,...` | all | Comma-separated: `data`, `train`, `eval`, `export` |
| `--force-rebuild` | — | Re-download all datasets |
| `--checkpoint PATH` | — | Resume training from checkpoint |
| `--log-level LEVEL` | `INFO` | Logging verbosity |

### Pipeline details

1. **`data`** — Downloads from HuggingFace dataset sources per TRM, applies text preprocessors, length-filters (40–2048 chars), deduplicates via MD5 hash, caps each label at 5,000 samples, produces a balanced shuffled JSONL.
2. **`train`** — Tokenises with dynamic padding (max 256 tokens), trains for 5 epochs with early stopping (patience=3) on macro F1, saves best + versioned checkpoints. fp16 is auto-enabled when a CUDA GPU is detected.
3. **`eval`** — Reloads the best checkpoint and prints a full classification report on the held-out test set.
4. **`export`** — Copies the best checkpoint to `training/outputs/<trm-name>/` ready for inference.

---

## Real-time screen scanner

The `scanner/` module captures your screen, OCRs the text with RapidOCR, and
classifies it live using the exported model.

### Setup

```bash
pip install -r scanner/requirements.txt
```

> RapidOCR downloads ~20 MB of ONNX model weights on first use — no separate
> binary install required.

### Run

```bash
# Full screen, default settings (1.5 s refresh)
python -m scanner

# Capture a specific region (left top width height)
python -m scanner --region 0 0 1280 720

# Use a custom model, scan every 2 s, show capture border
python -m scanner --model training/outputs/impact --interval 2000 --show-region

# Raise confidence threshold, use GPU 0
python -m scanner --threshold 0.75 --gpu 0
```

A small floating overlay appears (always-on-top, draggable) showing the
current label, confidence score, and all five class probabilities.
Use ⏸ to pause and ✕ to close.

### CLI reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--model PATH` | `training/outputs/impact` | Path to exported model |
| `--region L T W H` | full screen | Capture region in pixels |
| `--interval MS` | `1500` | Refresh interval in milliseconds |
| `--threshold FLOAT` | `0.50` | Confidence threshold (0.0–1.0); below this shows UNCERTAIN |
| `--gpu IDX` | `-1` (CPU) | GPU index; `-1` for CPU |
| `--show-region` | — | Draw a visible border around the capture area |

### 4-layer filter pipeline

OCR output passes through these filters before classification:

| Layer | Filter | Purpose |
|-------|--------|---------|
| **L1** | `WidthRatioFilter` | Drop detections narrower than 20% of the capture width (buttons, tabs) |
| **L2** | `WordCountFilter` | Drop detections with fewer than 4 words (timestamps, badges) |
| **L3** | `StopwordFilter` | Reject UI label blocks with no stopwords |
| **L4** | `DeduplicationFilter` | Skip re-classification when the screen hasn't changed (85% similarity threshold) |

### Long-text chunking

Texts longer than 510 tokens are split into overlapping windows. Scores are
averaged across all chunks before the winning label is selected — no content
is discarded.

---

## Adding a new TRM

Create a new label enum and `TRMConfig` in `training/config.py`, then register it:

```python
class MyLabel(str, Enum):
    A = "A"
    B = "B"
    C = "C"

MY_TRM = TRMRegistry.register(TRMConfig(
    name="my-classifier",
    description="...",
    labels=[lbl.value for lbl in MyLabel],
    model=ModelConfig(model_name="distilbert/distilbert-base-uncased", num_labels=3),
    data=DataConfig(sources=[...]),
    training=TrainingConfig(num_train_epochs=5),
))
```

Then run:

```bash
python -m training.orchestrator --trm my-classifier
```
