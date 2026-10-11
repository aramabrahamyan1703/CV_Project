"""Colours, fonts and ttk styles for the main window.

The look is a quiet lab instrument: the image sits on 18% grey so a
filter's effect on colour and contrast reads true, the panels around it are
light graphite, and a single accent colour marks "where you are" -- the
current step, the tool that produced the image, the cursor on the rulers.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import font as tkfont, ttk


SURFACE = "#F4F5F6"  # toolbar and status bar
PANEL = "#E7E8EA"  # sidebar and results panel
HOVER = "#D9DBDF"
LINE = "#C5C8CD"
RULER = "#DCDEE1"
RULER_SPAN = "#EFF0F2"  # the stretch of a ruler the image covers
CANVAS = "#767676"  # 18% grey card: neutral, so colours are judged fairly
CANVAS_TEXT = "#F1F1F1"
CANVAS_HINT = "#D2D2D2"
INK = "#1E2227"
MUTED = "#5F6670"
FAINT = "#A0A5AC"
ACCENT = "#2346C4"
ACCENT_HOVER = "#1C3AA6"
ACCENT_PRESSED = "#172F88"
ACCENT_SOFT = "#DCE3F8"
ON_ACCENT = "#FFFFFF"

# First installed family wins, so the app looks intended on macOS, Windows
# and Linux alike.
UI_FAMILIES = ("Avenir Next", "Segoe UI", "Helvetica Neue", "DejaVu Sans")
MONO_FAMILIES = ("Menlo", "Consolas", "DejaVu Sans Mono")


def _first_family(root: tk.Misc, candidates: tuple[str, ...], fallback: str) -> str:
    available = set(tkfont.families(root))
    for family in candidates:
        if family in available:
            return family
    return tkfont.nametofont(fallback, root=root).actual("family")


class Fonts:
    """The app's type scale: one UI family plus a monospace for measurements."""

    def __init__(self, root: tk.Misc) -> None:
        ui = _first_family(root, UI_FAMILIES, "TkDefaultFont")
        mono = _first_family(root, MONO_FAMILIES, "TkFixedFont")
        # Tk on macOS draws a point as one pixel; elsewhere it is ~1.33 px,
        # so the same numbers would look a size too large there.
        base = 13 if root.tk.call("tk", "windowingsystem") == "aqua" else 10
        self.body = tkfont.Font(root, family=ui, size=base)
        self.strong = tkfont.Font(root, family=ui, size=base, weight="bold")
        self.heading = tkfont.Font(root, family=ui, size=base + 3, weight="bold")
        self.section = tkfont.Font(root, family=ui, size=base - 2, weight="bold")
        self.hint = tkfont.Font(root, family=ui, size=base - 1)
        self.mono = tkfont.Font(root, family=mono, size=base - 1)
        self.mono_small = tkfont.Font(root, family=mono, size=base - 3)


def apply_theme(root: tk.Misc) -> Fonts:
    """Configure every ttk style the window uses and return its fonts."""
    fonts = Fonts(root)
    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")  # the only built-in theme that honours colours everywhere

    flat = {"bordercolor": PANEL, "lightcolor": PANEL, "darkcolor": PANEL}
    # Borders and troughs stay clam's own, so scrollbars and sliders a tool
    # puts in its dialog keep their contrast; the styles below flatten their own.
    style.configure(
        ".",
        background=PANEL,
        foreground=INK,
        font=fonts.body,
        focuscolor=ACCENT,
        selectbackground=ACCENT,
        selectforeground=ON_ACCENT,
    )

    style.configure("TFrame", background=PANEL)
    style.configure("Surface.TFrame", background=SURFACE)
    style.configure("Line.TFrame", background=LINE)
    style.configure("Ruler.TFrame", background=RULER)

    style.configure("TLabel", background=PANEL, foreground=INK)
    style.configure("Heading.TLabel", font=fonts.heading)
    style.configure("Section.TLabel", font=fonts.section, foreground=MUTED)
    style.configure("Surface.TLabel", background=SURFACE)
    style.configure("Readout.TLabel", background=SURFACE, foreground=MUTED, font=fonts.mono)
    style.configure("Ruler.TLabel", background=RULER, foreground=MUTED, font=fonts.mono_small)

    # Dialog buttons (Apply / Cancel) and anything else unstyled.
    style.configure(
        "TButton",
        background=SURFACE,
        bordercolor=LINE,
        lightcolor=SURFACE,
        darkcolor=SURFACE,
        padding=(12, 4),
    )
    style.map(
        "TButton",
        background=[("disabled", SURFACE), ("pressed", ACCENT_SOFT), ("active", HOVER)],
        lightcolor=[("pressed", ACCENT_SOFT), ("active", HOVER)],
        darkcolor=[("pressed", ACCENT_SOFT), ("active", HOVER)],
        foreground=[("disabled", FAINT)],
    )

    surface_flat = {"bordercolor": SURFACE, "lightcolor": SURFACE, "darkcolor": SURFACE}
    style.configure("Toolbar.TButton", background=SURFACE, padding=(10, 5), **surface_flat)
    style.map(
        "Toolbar.TButton",
        background=[("disabled", SURFACE), ("pressed", ACCENT_SOFT), ("active", HOVER)],
        bordercolor=[("pressed", ACCENT_SOFT), ("active", HOVER)],
        lightcolor=[("pressed", ACCENT_SOFT), ("active", HOVER)],
        darkcolor=[("pressed", ACCENT_SOFT), ("active", HOVER)],
        foreground=[("disabled", FAINT)],
    )

    accent_flat = {"bordercolor": ACCENT, "lightcolor": ACCENT, "darkcolor": ACCENT}
    style.configure(
        "Primary.TButton",
        background=ACCENT,
        foreground=ON_ACCENT,
        font=fonts.strong,
        padding=(12, 5),
        focuscolor=ON_ACCENT,
        **accent_flat,
    )
    pressed_or_hover = [("pressed", ACCENT_PRESSED), ("active", ACCENT_HOVER)]
    style.map(
        "Primary.TButton",
        background=pressed_or_hover,
        bordercolor=pressed_or_hover,
        lightcolor=pressed_or_hover,
        darkcolor=pressed_or_hover,
    )

    # Sidebar rows: flat until hovered; disabled rows fade instead of vanishing.
    style.configure("Tool.TButton", anchor="w", background=PANEL, padding=(10, 5), **flat)
    hover = [("disabled", PANEL), ("pressed", ACCENT_SOFT), ("active", HOVER)]
    style.map(
        "Tool.TButton",
        background=hover,
        bordercolor=hover,
        lightcolor=hover,
        darkcolor=hover,
        foreground=[("disabled", FAINT)],
    )
    # The tool that produced the image on screen.
    soft = {"bordercolor": ACCENT_SOFT, "lightcolor": ACCENT_SOFT, "darkcolor": ACCENT_SOFT}
    style.configure(
        "Current.Tool.TButton",
        background=ACCENT_SOFT,
        foreground=ACCENT,
        font=fonts.strong,
        **soft,
    )
    style.map(
        "Current.Tool.TButton",
        background=[("active", ACCENT_SOFT)],
        bordercolor=[("active", ACCENT_SOFT)],
        lightcolor=[("active", ACCENT_SOFT)],
        darkcolor=[("active", ACCENT_SOFT)],
    )

    row_height = fonts.body.metrics("linespace") + 10
    style.configure(
        "Treeview",
        background=PANEL,
        fieldbackground=PANEL,
        foreground=INK,
        rowheight=row_height,
        borderwidth=0,
        indent=0,
    )
    style.map("Treeview", background=[("selected", HOVER)], foreground=[("selected", INK)])
    style.layout("Treeview", [("Treeview.treearea", {"sticky": "nswe"})])

    # A thin, arrowless scrollbar for the sidebar.
    style.layout(
        "Slim.Vertical.TScrollbar",
        [
            (
                "Vertical.Scrollbar.trough",
                {"sticky": "ns", "children": [("Vertical.Scrollbar.thumb", {"expand": "1", "sticky": "nswe"})]},
            )
        ],
    )
    style.configure(
        "Slim.Vertical.TScrollbar",
        troughcolor=PANEL,
        background=LINE,
        arrowsize=8,
        width=8,
        **flat,
    )
    style.map("Slim.Vertical.TScrollbar", background=[("active", FAINT)])

    style.configure("TPanedwindow", background=LINE)
    style.configure("Sash", sashthickness=2, gripcount=0, background=LINE)

    style.configure("TCheckbutton", background=PANEL)
    style.configure("TRadiobutton", background=PANEL)
    style.configure("TCombobox", fieldbackground=SURFACE, background=SURFACE, bordercolor=LINE)
    style.configure("TEntry", fieldbackground=SURFACE, bordercolor=LINE)

    if isinstance(root, (tk.Tk, tk.Toplevel)):
        root.configure(background=PANEL)
    return fonts
