# classifier-trm

Local TRM (Tiny Recursive Model) training and inference pipeline.

## What it does

Fine-tunes `distilbert/distilbert-base-uncased` (67M params) to classify text into five categories:

| Label | Meaning |
|-------|---------|
| `CLEAN` | High quality — genuine, informative, well-written |
| `SLOP` | Low-effort filler — copy-paste, padding, generic |
| `CRINGE` | Embarrassing / try-hard / socially awkward |
| `BOT` | Automated, templated, spam-like |
| `STUPID` | Misinformation, incoherent, factually wrong |

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
python -m training.orchestrator --trm content-quality
```

### Run individual stages

```bash
# 1. Download datasets and build datadumps only
python -m training.orchestrator --trm content-quality --steps data

# 2. Train the model
python -m training.orchestrator --trm content-quality --steps train

# 3. Evaluate the best checkpoint on the test set
python -m training.orchestrator --trm content-quality --steps eval

# 4. Export the final model to outputs/
python -m training.orchestrator --trm content-quality --steps export

# Force re-download all datasets
python -m training.orchestrator --trm content-quality --steps data --force-rebuild

# Resume training from a checkpoint
python -m training.orchestrator --trm content-quality --steps train --checkpoint checkpoints/content-quality/epoch-5
```

### CLI reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--trm NAME` | `content-quality` | TRM to train (see `TRMRegistry`) |
| `--steps STAGE,...` | all | Comma-separated: `data`, `train`, `eval`, `export` |
| `--force-rebuild` | — | Re-download all datasets |
| `--checkpoint PATH` | — | Resume training from checkpoint |
| `--log-level LEVEL` | `INFO` | Logging verbosity |

### Pipeline details

1. **`data`** — Downloads from 13 HuggingFace dataset sources, applies text preprocessors, length-filters (40–2048 chars), deduplicates via MD5 hash, caps each label at 5,000 samples, produces a balanced shuffled JSONL.
2. **`train`** — Tokenises with dynamic padding (max 256 tokens), trains for 5 epochs with early stopping (patience=3) on macro F1, saves best + versioned checkpoints.
3. **`eval`** — Reloads the best checkpoint and prints a full classification report on the held-out test set.
4. **`export`** — Copies the best checkpoint to `training/outputs/` ready for inference.

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
python -m scanner --model training/outputs/content-quality --interval 2000 --show-region

# Raise confidence threshold, use GPU 0
python -m scanner --threshold 0.75 --gpu 0
```

A small floating overlay appears (always-on-top, draggable) showing the
current label, confidence score, and all five class probabilities.
Use ⏸ to pause and ✕ to close.

### CLI reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--model PATH` | `training/outputs/content-quality` | Path to exported model |
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

## Adding a second TRM

Create a new `TRMConfig` in `training/config.py` and register it:

```python
MY_TRM = TRMRegistry.register(TRMConfig(
    name="my-classifier",
    description="...",
    labels=["A", "B", "C"],
    model=ModelConfig(model_name="distilbert/distilbert-base-uncased", num_labels=3),
    data=DataConfig(sources=[...]),
    training=TrainingConfig(num_train_epochs=3),
))
```

Then run:

```bash
python -m training.orchestrator --trm my-classifier
```
