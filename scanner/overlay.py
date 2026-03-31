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
    # ── impact
    "STATIC":         {"color": "#7f8c8d", "icon": "–",  "tip": "Dead-end text — leaves you exactly where you started."},
    "CLARIFYING":     {"color": "#3498db", "icon": "◎",  "tip": "Removes confusion and simplifies a messy idea."},
    "PROVOCATIVE":    {"color": "#e67e22", "icon": "!",  "tip": "Makes you angry, excited, or defensive."},
    "TRANSFORMATIVE": {"color": "#2ecc71", "icon": "✦",  "tip": "Actually changes how you think or behave after reading."},
    "TOXIC":          {"color": "#e74c3c", "icon": "☠",  "tip": "Spreads negativity, lies, or mental exhaustion."},
    # ── flavor
    "RAW":            {"color": "#e67e22", "icon": "●",  "tip": "Unfiltered, messy thoughts — journals, voice notes."},
    "PROCESSED":      {"color": "#95a5a6", "icon": "▣",  "tip": "Clean and professional but soulless — press releases, corp speak."},
    "SPICY":          {"color": "#e74c3c", "icon": "~",  "tip": "Controversial, edgy, or intentionally bold."},
    "NOURISHING":     {"color": "#27ae60", "icon": "♦",  "tip": "Deep, thoughtful, and healthy for your perspective."},
    "EMPTY_CALORIES": {"color": "#7f8c8d", "icon": "○",  "tip": "Entertaining but mindless — gossip, clickbait, filler."},
    # ── purpose
    "DIRECTIVE":      {"color": "#3498db", "icon": "▶",  "tip": "Telling you what to do — manuals, instructions, commands."},
    "PERFORMATIVE":   {"color": "#9b59b6", "icon": "★",  "tip": "Showing off or building a brand — CVs, bios, marketing."},
    "SPECULATIVE":    {"color": "#f39c12", "icon": "?",  "tip": "Asking 'What if?' — philosophy, theories, sci-fi."},
    "CONFESSIONAL":   {"color": "#e91e8c", "icon": "♡",  "tip": "Revealing a personal truth — memoirs, diary entries, letters."},
    "DECORATIVE":     {"color": "#95a5a6", "icon": "◌",  "tip": "Just there to look pretty — filler, greeting-card fluff."},
    # ── lifespan
    "INSTANT":        {"color": "#e74c3c", "icon": "⚡",  "tip": "Expires in seconds — OTPs, 'I'm here', quick reactions."},
    "DAILY":          {"color": "#f39c12", "icon": "↻",  "tip": "Useless by tomorrow — breaking news, weather, scores."},
    "SEASONAL":       {"color": "#2ecc71", "icon": "◐",  "tip": "Good for a few months — trend reports, local reviews."},
    "DECADAL":        {"color": "#3498db", "icon": "◑",  "tip": "Relevant for years — career advice, how-to guides, laws."},
    "EVERGREEN":      {"color": "#27ae60", "icon": "∞",  "tip": "Never dies — encyclopedic knowledge, timeless wisdom."},
    # ── fallback
    "UNCERTAIN":      {"color": "#7f8c8d", "icon": "?",  "tip": "Confidence too low to classify reliably."},
}

_BG_DARK    = "#1e1e2e"
_BG_BAR     = "#13131f"
_BG_TOOLTIP = "#2a2a3e"
_FG_MAIN    = "#cdd6f4"
_FG_DIM     = "#585b70"
_FONT_UI    = ("Segoe UI", 11)


# ── Tooltip ───────────────────────────────────────────────────────────────────

class _Tooltip:
    """Lightweight hover tooltip that follows the label's live text."""

    _DELAY_MS = 400   # ms before popup appears
    _PAD      = 6

    def __init__(self, widget: tk.Widget, text_fn) -> None:
        """*text_fn* is a zero-arg callable returning the tooltip string."""
        self._widget  = widget
        self._text_fn = text_fn
        self._tip: Optional[tk.Toplevel] = None
        self._after_id: Optional[str]   = None
        widget.bind("<Enter>", self._schedule)
        widget.bind("<Leave>", self._cancel)

    def _schedule(self, _event: tk.Event) -> None:
        self._cancel(None)
        self._after_id = self._widget.after(self._DELAY_MS, self._show)

    def _cancel(self, _event) -> None:
        if self._after_id is not None:
            self._widget.after_cancel(self._after_id)
            self._after_id = None
        if self._tip is not None:
            try:
                self._tip.destroy()
            except tk.TclError:
                pass
            self._tip = None

    def _show(self) -> None:
        text = self._text_fn()
        if not text:
            return
        widget = self._widget
        x = widget.winfo_rootx() + widget.winfo_width() + 6
        y = widget.winfo_rooty()
        self._tip = tk.Toplevel(widget)
        self._tip.overrideredirect(True)
        self._tip.attributes("-topmost", True)
        lbl = tk.Label(
            self._tip, text=text, justify="left",
            bg=_BG_TOOLTIP, fg=_FG_MAIN,
            font=("Segoe UI", 10), wraplength=260,
            padx=self._PAD, pady=self._PAD // 2,
        )
        lbl.pack()
        self._tip.update_idletasks()
        # Nudge left if it would bleed off the right edge of the screen
        sw = widget.winfo_screenwidth()
        tw = self._tip.winfo_width()
        if x + tw > sw:
            x = widget.winfo_rootx() - tw - 6
        self._tip.geometry(f"+{x}+{y}")

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

        self._trm_rows: Dict[str, Dict] = {}
        for trm_name in ("impact", "flavor", "purpose", "lifespan"):
            row = tk.Frame(content, bg=_BG_DARK)
            row.pack(fill="x", pady=3)

            tk.Label(
                row, text=trm_name.upper(), width=8, anchor="w",
                font=("Segoe UI", 10), bg=_BG_DARK, fg=_FG_DIM,
            ).pack(side="left")

            label_var = tk.StringVar(value="…")
            label_widget = tk.Label(
                row, textvariable=label_var,
                font=("Segoe UI", 13, "bold"), bg=_BG_DARK, fg=_FG_DIM, anchor="w",
                cursor="question_arrow",
            )
            label_widget.pack(side="left", fill="x", expand=True)

            score_var = tk.StringVar(value="")
            tk.Label(
                row, textvariable=score_var,
                font=("Segoe UI", 10), bg=_BG_DARK, fg=_FG_DIM, width=5, anchor="e",
            ).pack(side="right")

            row_data: Dict = {
                "label_var":     label_var,
                "label_widget":  label_widget,
                "score_var":     score_var,
                "current_label": "",
            }
            _Tooltip(
                label_widget,
                lambda rd=row_data: _LABEL_META.get(rd["current_label"], {}).get("tip", ""),
            )
            self._trm_rows[trm_name] = row_data

    def _build_status_bar(self, parent: tk.Tk) -> None:
        self._status_var = tk.StringVar(value="Initialising…")
        tk.Label(
            parent, textvariable=self._status_var,
            bg=_BG_BAR, fg=_FG_DIM, font=("Segoe UI", 9), anchor="w",
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
        trm_results = result.get("trm_results", {})
        status      = result.get("status", "")

        for trm_name, row in self._trm_rows.items():
            r     = trm_results.get(trm_name)
            if r:
                label = r.get("label", "UNCERTAIN")
                score = r.get("score", 0.0)
                meta  = _LABEL_META.get(label, _LABEL_META["UNCERTAIN"])
                row["current_label"] = label
                row["label_var"].set(f"{meta['icon']}  {label}")
                row["label_widget"].configure(fg=meta["color"])
                row["score_var"].set(f"{score:.0%}")
            else:
                row["current_label"] = ""
                row["label_var"].set("–")
                row["label_widget"].configure(fg=_FG_DIM)
                row["score_var"].set("")

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
            for row in self._trm_rows.values():
                row["label_var"].set("⏸  PAUSED")
                row["label_widget"].configure(fg=_FG_DIM)
                row["score_var"].set("")
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

