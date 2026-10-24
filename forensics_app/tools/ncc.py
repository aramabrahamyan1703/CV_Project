"""Normalized cross-correlation response image (Set 4.1).

Matches the slide: ``match_template(..., pad_input=True)`` and a three-panel
sheet of Original | Template | Normalized cross-correlation image (gray).
"""

from __future__ import annotations

import math
import tkinter as tk
from tkinter import filedialog, messagebox

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image
from skimage import feature

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .masking import OPEN_TYPES


ACCEPTED_MODES = ("L", "RGB", "RGBA", "P")


def _as_gray_float(image: Image.Image) -> np.ndarray:
    """Float grayscale in ``[0, 1]`` for ``match_template``."""
    if image.mode not in ACCEPTED_MODES:
        raise ValueError(
            f"NCC needs an L, RGB, RGBA or P image; got mode {image.mode!r}."
        )
    if image.width < 1 or image.height < 1:
        raise ValueError("Image is empty.")
    return np.asarray(image.convert("L"), dtype=np.float64) / 255.0


def compute_ncc(scene: Image.Image, template: Image.Image) -> np.ndarray:
    """Return a full-size NCC response via ``skimage.feature.match_template``.

    ``pad_input=True`` so the map matches the scene size; each value is the
    correlation for a template centred on that pixel.
    """
    scene_g = _as_gray_float(scene)
    template_g = _as_gray_float(template)
    th, tw = template_g.shape
    sh, sw = scene_g.shape
    if th > sh or tw > sw:
        raise ValueError(
            f"The template ({tw}×{th}) is larger than the image ({sw}×{sh})."
        )
    if float(template_g.std()) < 1e-12:
        raise ValueError(
            "The template is constant, so normalized cross-correlation is undefined.\n\n"
            "Use a template with some texture or contrast."
        )
    return feature.match_template(scene_g, template_g, pad_input=True)


class NCCTool(ForensicsTool):
    tool_id = "ncc"
    title = "NCC image"
    category = "Matching"
    description = "Normalized cross-correlation response against a template (Set 4.1)."

    ACCEPTED_MODES = ACCEPTED_MODES
    PANEL_IN = 4.0
    TITLE_IN = 0.9
    MIN_DPI = 100
    MAX_DPI = 400
    MAX_PIXELS = 16_000_000

    def is_available(self, document: ImageDocument) -> bool:
        if document.current is None:
            return True
        if document.current.mode not in self.ACCEPTED_MODES:
            return False
        return document.last_tool_id != self.tool_id

    def unavailable_message(self, document: ImageDocument) -> str:
        current = document.current
        if current is not None and current.mode not in self.ACCEPTED_MODES:
            return (
                f"This image mode is not supported for NCC (mode {current.mode!r}).\n\n"
                "Open an L or RGB image."
            )
        return (
            "The NCC visualization is already shown.\n\n"
            "Undo or reset the image to run NCC again."
        )

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        source = document.current
        assert source is not None

        template = self._ask_template(parent)
        if template is None:
            return None

        try:
            ncc = compute_ncc(source, template)
        except ValueError as error:
            messagebox.showinfo(self.title, str(error), parent=parent)
            return None

        peak = float(np.nanmax(ncc))
        return ToolResult(
            image=self._render_sheet(source, template, ncc),
            message=f"Computed NCC image (peak {peak:.3f}).",
            details={
                "Operation": self.title,
                "Function": "skimage.feature.match_template",
                "Template size": f"{template.width}×{template.height}",
                "Peak score": f"{peak:.4f}",
                "Pad input": "yes",
            },
        )

    def _ask_template(self, parent: tk.Misc | None) -> Image.Image | None:
        path = filedialog.askopenfilename(
            parent=parent,
            title="Choose a template for NCC",
            filetypes=OPEN_TYPES,
        )
        if not path:
            return None
        try:
            with Image.open(path) as opened:
                return opened.copy()
        except OSError as error:
            messagebox.showinfo(self.title, f"Could not open template:\n{error}", parent=parent)
            return None

    def _render_sheet(
        self, source: Image.Image, template: Image.Image, ncc: np.ndarray
    ) -> Image.Image:
        """Slide layout: Original image | Template | Normalized cross-correlation image."""
        width, height = source.size
        panels = 3
        fig_w = panels * self.PANEL_IN
        fig_h = min(self.PANEL_IN * height / max(width, 1) + self.TITLE_IN, 2 * self.PANEL_IN)
        native_dpi = width / (self.PANEL_IN * 0.85)
        dpi = min(self.MAX_DPI, max(self.MIN_DPI, native_dpi))
        dpi = min(dpi, math.sqrt(self.MAX_PIXELS / (fig_w * fig_h)))

        figure = Figure(figsize=(fig_w, fig_h), dpi=int(dpi), layout="constrained")
        axes = figure.subplots(1, panels)

        original = source.convert("L")
        axes[0].imshow(np.asarray(original), cmap="gray", vmin=0, vmax=255)
        axes[0].set_title("Original image")

        tmpl = template.convert("L")
        axes[1].imshow(np.asarray(tmpl), cmap="gray", vmin=0, vmax=255)
        axes[1].set_title("Template")

        # Slide uses cmap='gray' for the correlation map.
        axes[2].imshow(ncc, cmap="gray")
        axes[2].set_title("Normalized cross-correlation image")

        for ax in axes:
            ax.set_axis_off()

        canvas = FigureCanvasAgg(figure)
        canvas.draw()
        image = Image.fromarray(np.asarray(canvas.buffer_rgba())).convert("RGB")
        plt.close(figure)
        return image
