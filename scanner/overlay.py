"""
overlay.py

Always-on-top tkinter overlay that displays TRM classification results
in real time, plus an optional region box showing exactly what area is
being captured.

Threading contract
------------------
- `ScannerOverlay` MUST be created and `run()` called on the **main thread**.
- The background scanner thread communicates via a `queue.Queue(maxsize=1)`.
- The overlay drains the queue every 100 ms via `after()` — fully thread-safe.
"""
from __future__ import annotations

import queue
import tkinter as tk
from typing import TYPE_CHECKING, Dict, Optional, Tuple

if TYPE_CHECKING:
    from .config import ScannerConfig


# ── Label styling ─────────────────────────────────────────────────────────────

_LABEL_META: Dict[str, Dict] = {
    "CLEAN":     {"color": "#2ecc71", "icon": "✓"},
    "SLOP":      {"color": "#f39c12", "icon": "~"},
    "CRINGE":    {"color": "#e67e22", "icon": "!"},
    "BOT":       {"color": "#e74c3c", "icon": "⚠"},
    "STUPID":    {"color": "#c0392b", "icon": "✗"},
    "UNCERTAIN": {"color": "#7f8c8d", "icon": "?"},
}

_BG_DARK = "#1e1e2e"
_BG_BAR  = "#13131f"
_FG_MAIN = "#cdd6f4"
_FG_DIM  = "#585b70"
_FONT_UI = ("Segoe UI", 9)

# A colour that will be made fully transparent on Windows.
# Must not appear in any real UI element.
_TRANSPARENT_KEY = "#010203"


# ── Region box ────────────────────────────────────────────────────────────────

class RegionBox:
    """
    Frameless, always-on-top window with a coloured border and a transparent
    interior, positioned exactly over the capture region so you can see what
    the scanner is looking at.

    Works on Windows via the -transparentcolor attribute.
    On other platforms the interior will be a dark near-black instead.
    """

    _BORDER = 3  # px

    def __init__(
        self,
        root: tk.Tk,
        region: Optional[Tuple[int, int, int, int]],
        color: str,
        screen_w: int,
        screen_h: int,
    ) -> None:
        if region:
            left, top, w, h = region
        else:
            left, top, w, h = 0, 0, screen_w, screen_h

        win = tk.Toplevel(root)
        win.overrideredirect(True)
        win.attributes("-topmost", True)
        win.geometry(f"{w}x{h}+{left}+{top}")
        win.configure(bg=_TRANSPARENT_KEY)

        try:
            # Windows: make the key colour fully transparent (punch-through).
            win.attributes("-transparentcolor", _TRANSPARENT_KEY)
        except tk.TclError:
            pass  # macOS / Linux — interior stays near-black, border still visible.

        b = self._BORDER
        canvas = tk.Canvas(win, bg=_TRANSPARENT_KEY, highlightthickness=0)
        canvas.pack(fill="both", expand=True)
        # Draw only the border — interior stays transparent.
        canvas.create_rectangle(b, b, w - b, h - b, outline=color, width=b * 2, fill="")

        self._win = win

    def destroy(self) -> None:
        try:
            self._win.destroy()
        except tk.TclError:
            pass


# ── Result overlay ────────────────────────────────────────────────────────────

class ScannerOverlay:
    """
    Floating, frameless, draggable result overlay.

    Layout::

        ╔═══════════════════════════════╗
        ║ TRM Scanner          ⏸  ✕   ║  ← draggable title bar
        ╠═══════════════════════════════╣
        ║         ⚠  BOT                ║  ← label coloured by class
        ║       98.7% confidence        ║
        ║  BOT:98% CLEAN:1% SLOP:0%    ║  ← all scores, small
        ╠═══════════════════════════════╣
        ║  342 chars · 2 chunks         ║  ← status bar
        ╚═══════════════════════════════╝
    """

    def __init__(self, config: "ScannerConfig", result_queue: queue.Queue) -> None:
        self._cfg    = config
        self._queue  = result_queue
        self._paused = False

        self._root = tk.Tk()

        # Show region box before building the main UI (both need same Tk root).
        self._region_box: Optional[RegionBox] = None
        if config.show_region:
            import mss
            with mss.mss() as sct:
                m = sct.monitors[1]
                sw, sh = m["width"], m["height"]
            self._region_box = RegionBox(
                self._root,
                config.capture_region,
                config.region_box_color,
                sw, sh,
            )

        self._build_ui()
        self._poll()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = self._root
        w, h = self._cfg.overlay_width, self._cfg.overlay_height
        x, y = self._cfg.overlay_x, self._cfg.overlay_y
        root.title("TRM Scanner")
        root.geometry(f"{w}x{h}+{x}+{y}")
        root.configure(bg=_BG_DARK)
        root.attributes("-topmost", True)
        root.attributes("-alpha", self._cfg.overlay_alpha)
        root.overrideredirect(True)

        self._build_title_bar(root)
        self._build_content(root)
        self._build_status_bar(root)

    def _build_title_bar(self, parent: tk.Tk) -> None:
        bar = tk.Frame(parent, bg=_BG_BAR, height=26)
        bar.pack(fill="x", side="top")
        bar.pack_propagate(False)
        bar.bind("<ButtonPress-1>", self._drag_start)
        bar.bind("<B1-Motion>",     self._drag_motion)

        tk.Label(bar, text=" TRM Scanner", bg=_BG_BAR, fg=_FG_MAIN, font=_FONT_UI).pack(side="left")

        tk.Button(
            bar, text="✕", width=3,
            bg=_BG_BAR, fg="#f38ba8", activebackground=_BG_BAR,
            relief="flat", bd=0, font=_FONT_UI,
            command=self._close,
        ).pack(side="right")

        self._pause_btn = tk.Button(
            bar, text="⏸", width=3,
            bg=_BG_BAR, fg=_FG_MAIN, activebackground=_BG_BAR,
            relief="flat", bd=0, font=_FONT_UI,
            command=self._toggle_pause,
        )
        self._pause_btn.pack(side="right")

    def _build_content(self, parent: tk.Tk) -> None:
        content = tk.Frame(parent, bg=_BG_DARK)
        content.pack(fill="both", expand=True, padx=10, pady=6)

        self._label_var = tk.StringVar(value="SCANNING…")
        self._label_widget = tk.Label(
            content, textvariable=self._label_var,
            font=("Segoe UI", 24, "bold"), bg=_BG_DARK, fg=_FG_DIM,
        )
        self._label_widget.pack()

        self._score_var = tk.StringVar(value="")
        tk.Label(content, textvariable=self._score_var,
                 font=("Segoe UI", 10), bg=_BG_DARK, fg=_FG_DIM).pack()

        self._all_var = tk.StringVar(value="")
        tk.Label(
            content, textvariable=self._all_var,
            font=("Segoe UI", 8), bg=_BG_DARK, fg=_FG_DIM,
            wraplength=self._cfg.overlay_width - 20, justify="center",
        ).pack(pady=(4, 0))

    def _build_status_bar(self, parent: tk.Tk) -> None:
        self._status_var = tk.StringVar(value="Initialising…")
        tk.Label(
            parent, textvariable=self._status_var,
            bg=_BG_BAR, fg=_FG_DIM, font=("Segoe UI", 7), anchor="w",
        ).pack(fill="x", side="bottom", padx=4)

    # ── Queue polling ─────────────────────────────────────────────────────────

    def _poll(self) -> None:
        try:
            result = self._queue.get_nowait()
            self._apply_result(result)
        except queue.Empty:
            pass
        self._root.after(100, self._poll)

    def _apply_result(self, result: Dict) -> None:
        label      = result.get("label", "UNCERTAIN")
        score      = result.get("score", 0.0)
        all_scores = result.get("all_scores", {})
        status     = result.get("status", "")

        meta  = _LABEL_META.get(label, _LABEL_META["UNCERTAIN"])
        self._label_var.set(f"{meta['icon']}  {label}")
        self._label_widget.configure(fg=meta["color"])
        self._score_var.set(f"{score:.1%} confidence" if score else "")

        if all_scores:
            parts = [f"{k}:{v:.0%}" for k, v in sorted(all_scores.items(), key=lambda x: -x[1])]
            self._all_var.set("  ".join(parts))
        else:
            self._all_var.set("")

        if status:
            self._status_var.set(f"  {status}")

    # ── Controls ──────────────────────────────────────────────────────────────

    def _close(self) -> None:
        if self._region_box:
            self._region_box.destroy()
        self._root.destroy()

    def _toggle_pause(self) -> None:
        self._paused = not self._paused
        self._pause_btn.configure(text="▶" if self._paused else "⏸")
        if self._paused:
            self._status_var.set("  Paused")
            self._label_var.set("⏸  PAUSED")
            self._label_widget.configure(fg=_FG_DIM)
            self._score_var.set("")
            self._all_var.set("")

    @property
    def paused(self) -> bool:
        return self._paused

    # ── Drag ──────────────────────────────────────────────────────────────────

    def _drag_start(self, event: tk.Event) -> None:
        self._drag_offset_x = event.x_root - self._root.winfo_x()
        self._drag_offset_y = event.y_root - self._root.winfo_y()

    def _drag_motion(self, event: tk.Event) -> None:
        x = event.x_root - self._drag_offset_x
        y = event.y_root - self._drag_offset_y
        self._root.geometry(f"+{x}+{y}")

    # ── Entry point ───────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """Request a graceful close from any thread (thread-safe via after())."""
        self._root.after(0, self._close)

    def run(self) -> None:
        self._root.mainloop()

