"""
config.py

All configuration for the TRM real-time screen scanner.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Model path — points at the exported content-quality model
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULT_MODEL_PATH = (
    Path(__file__).parent.parent / "training" / "outputs" / "content-quality"
)

# ─────────────────────────────────────────────────────────────────────────────
# Label presentation
# ─────────────────────────────────────────────────────────────────────────────
# Each label maps to (hex_color, display_icon)
LABEL_STYLE: dict[str, tuple[str, str]] = {
    "CLEAN":     ("#27ae60", "✓"),
    "SLOP":      ("#f39c12", "~"),
    "CRINGE":    ("#e67e22", "!"),
    "BOT":       ("#e74c3c", "⚠"),
    "STUPID":    ("#c0392b", "✗"),
    "UNCERTAIN": ("#7f8c8d", "?"),
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
    model_path: Path                              = _DEFAULT_MODEL_PATH
    capture_region: Optional[Tuple[int,int,int,int]] = None
    refresh_interval_ms: int                      = 1_500
    min_text_chars: int                           = 30
    max_text_chars: int                           = 2_048
    confidence_threshold: float                   = 0.50
    ocr_language: str                             = "eng"
    ocr_psm: int                                  = 3
    device: int                                   = -1     # -1=CPU, 0=first GPU

    # Overlay window
    overlay_width: int                            = 330
    overlay_height: int                           = 185
    overlay_alpha: float                          = 0.93   # 0.0 transparent – 1.0 opaque
    overlay_x: int                                = 40
    overlay_y: int                                = 40

    # Region box — draws a visible border around the capture area on screen
    show_region: bool                             = False
    region_box_color: str                         = "#00ff88"  # bright green
