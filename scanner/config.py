"""
config.py

All configuration for the TRM real-time screen scanner.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Model paths — one output directory per TRM
# ─────────────────────────────────────────────────────────────────────────────
_OUTPUTS_DIR = Path(__file__).parent.parent / "training" / "outputs"

_DEFAULT_MODEL_PATHS: Dict[str, Path] = {
    "impact":   _OUTPUTS_DIR / "impact",
    "flavor":   _OUTPUTS_DIR / "flavor",
    "purpose":  _OUTPUTS_DIR / "purpose",
    "lifespan": _OUTPUTS_DIR / "lifespan",
}

# ─────────────────────────────────────────────────────────────────────────────
# Label presentation  (hex_color, display_icon)
# ─────────────────────────────────────────────────────────────────────────────
LABEL_META: dict[str, tuple[str, str]] = {
    # ── impact ────────────────────────────────────────────────────────────
    "STATIC":         ("#7f8c8d", "–"),
    "CLARIFYING":     ("#3498db", "◎"),
    "PROVOCATIVE":    ("#e67e22", "!"),
    "TRANSFORMATIVE": ("#2ecc71", "✦"),
    "TOXIC":          ("#e74c3c", "☠"),
    # ── flavor ────────────────────────────────────────────────────────────
    "RAW":            ("#e67e22", "●"),
    "PROCESSED":      ("#95a5a6", "▣"),
    "SPICY":          ("#e74c3c", "~"),
    "NOURISHING":     ("#27ae60", "♦"),
    "EMPTY_CALORIES": ("#7f8c8d", "○"),
    # ── purpose ───────────────────────────────────────────────────────────
    "DIRECTIVE":      ("#3498db", "▶"),
    "PERFORMATIVE":   ("#9b59b6", "★"),
    "SPECULATIVE":    ("#f39c12", "?"),
    "CONFESSIONAL":   ("#e91e8c", "♡"),
    "DECORATIVE":     ("#95a5a6", "◌"),
    # ── lifespan ──────────────────────────────────────────────────────────
    "INSTANT":        ("#e74c3c", "⚡"),
    "DAILY":          ("#f39c12", "↻"),
    "SEASONAL":       ("#2ecc71", "◐"),
    "DECADAL":        ("#3498db", "◑"),
    "EVERGREEN":      ("#27ae60", "∞"),
    # ── fallback ──────────────────────────────────────────────────────────
    "UNCERTAIN":      ("#7f8c8d", "?"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Scanner config
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ScannerConfig:
    """
    Central config for the screen scanner.  All values have sensible defaults.

    capture_region:
        None   → captures the entire primary monitor.
        Tuple  → (left, top, width, height) in screen pixels.
                 e.g. (0, 0, 960, 1080) captures only the left half.

    refresh_interval_ms:
        How often the background thread runs a new scan cycle.
        1 500 ms recommended on CPU — gives the model time to finish.

    confidence_threshold:
        If the top label's probability is below this, the result is shown as
        UNCERTAIN rather than a potentially misleading classification.

    ocr_psm:
        Tesseract page segmentation mode.
        3 = fully automatic (good for general screens)
        6 = assume a single block of text (good if you pin a chat window)
    """
    model_paths: Dict[str, Path]                   = field(default_factory=lambda: dict(_DEFAULT_MODEL_PATHS))
    capture_region: Optional[Tuple[int,int,int,int]] = None
    refresh_interval_ms: int                      = 1_500
    min_text_chars: int                           = 30
    max_text_chars: int                           = 2_048
    confidence_threshold: float                   = 0.50
    ocr_language: str                             = "eng"
    ocr_psm: int                                  = 3
    device: int                                   = -1     # -1=CPU, 0=first GPU

    # Set to a subset of TRM names to only load those models (None = all four)
    enabled_trms: Optional[tuple] = None

    # Overlay window
    overlay_width: int                            = 420
    overlay_height: int                           = 250
    overlay_alpha: float                          = 0.93   # 0.0 transparent – 1.0 opaque
    overlay_x: int                                = 40
    overlay_y: int                                = 40

    # Region box — draws a visible border around the capture area on screen
    show_region: bool                             = False
    region_box_color: str                         = "#00ff88"  # bright green
