# Models

Local TRM (Tiny Recursive Model) training pipeline.

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

## Setup

```bash
pip install -r training/requirements.txt
```

---

## Run the full pipeline

```bash
# From this directory (models/)
python -m training.orchestrator --trm content-quality
```

## Run individual stages

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

---

## Project structure

```
models/
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
│   ├── ocr.py             # Tesseract OCR wrapper → clean text
│   ├── classifier.py      # HuggingFace pipeline wrapper → label + scores
│   ├── overlay.py         # Frameless tkinter always-on-top overlay
│   ├── scanner.py         # Orchestrator + CLI (main entry point)
│   ├── __init__.py        # Public API: Scanner, ScannerConfig
│   ├── __main__.py        # python -m scanner entry point
│   └── requirements.txt   # mss, pytesseract, Pillow
```

---

## Real-time screen scanner

The `scanner/` module captures your screen, OCRs the text, and classifies it
live using the exported model.

### Install

```bash
pip install -r scanner/requirements.txt
```

Also install the **Tesseract binary**:
- Windows: <https://github.com/UB-Mannheim/tesseract/wiki> — add to PATH
- macOS: `brew install tesseract`
- Ubuntu: `sudo apt install tesseract-ocr`

### Run

```bash
# Full screen, default settings (1.5 s refresh)
python -m scanner

# Capture a specific region  (left top width height)
python -m scanner --region 0 0 1280 720

# Scan every 2 seconds, use GPU 0
python -m scanner --interval 2000 --gpu 0
```

A small floating overlay appears (always-on-top, draggable) showing the
current label, confidence score, and all five class probabilities.
Use ⏸ to pause and ✕ to close.

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
