"""Canny edge detection from ``skimage.feature``."""

from __future__ import annotations

import tkinter as tk
from tkinter import simpledialog

import numpy as np
from PIL import Image
from skimage import feature

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


ACCEPTED_MODES = ("L", "RGB", "RGBA")
MAX_SIGMA = 20.0


def canny_edges(image: Image.Image, sigma: float) -> Image.Image:
    """Binary Canny edge map: edges white (255), everything else black.

    ``sigma`` is the Gaussian smoothing applied before the gradient: small
    values keep fine detail (and noise), larger ones keep only strong
    outlines. Edges are found on the grayscale (luminance) image, and the
    hysteresis thresholds are skimage's defaults (0.1 and 0.2 of full scale).
    """
    if image.mode not in ACCEPTED_MODES:
        raise ValueError(
            f"Canny edge detection needs an L, RGB or RGBA image; got mode {image.mode!r}."
        )
    if image.width < 1 or image.height < 1:
        raise ValueError("Image is empty; nothing to detect.")
    if not 0.0 < sigma <= MAX_SIGMA:
        raise ValueError(f"Sigma must be above 0 and at most {MAX_SIGMA:g}; got {sigma:g}.")

    gray = np.asarray(image.convert("L")).astype(np.float64) / 255.0
    edges = feature.canny(gray, sigma=sigma)
    return Image.fromarray(edges.astype(np.uint8) * 255, mode="L")


class CannyEdgeTool(ForensicsTool):
    tool_id = "skimage_canny"
    title = "Canny edges (skimage)"
    category = "Filtering"
    description = "Thin edge map with skimage.feature.canny; choose sigma."

    ACCEPTED_MODES = ACCEPTED_MODES
    DEFAULT_SIGMA = 1.0

    def is_available(self, document: ImageDocument) -> bool:
        if document.current is None:
            return True
        return document.current.mode in self.ACCEPTED_MODES

    def unavailable_message(self, document: ImageDocument) -> str:
        current = document.current
        mode = current.mode if current is not None else "unknown"
        return (
            f"This image mode is not supported by {self.title} (mode {mode!r}).\n\n"
            "Open an L, RGB or RGBA image."
        )

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        source = document.current
        assert source is not None
        assert source.mode in self.ACCEPTED_MODES, f"unsupported mode {source.mode!r}"

        sigma = self._ask_sigma(parent)
        if sigma is None:
            return None

        output = canny_edges(source, sigma)
        edge_pixels = int(np.count_nonzero(np.asarray(output)))
        total = output.width * output.height
        return ToolResult(
            image=output,
            message=f"Detected Canny edges (sigma {sigma:g}).",
            details={
                "Operation": self.title,
                "Function": "skimage.feature.canny",
                "Sigma": sigma,
                "Thresholds": "skimage defaults (low 0.1, high 0.2)",
                "Edge pixels": f"{edge_pixels} / {total} ({100 * edge_pixels / total:.1f}%)",
                "Output mode": output.mode,
            },
        )

    def _ask_sigma(self, parent: tk.Misc | None) -> float | None:
        return simpledialog.askfloat(
            self.title,
            "Sigma (Gaussian smoothing before edge detection).\n\n"
            "Small values (about 1) find fine detail and noise;\n"
            "larger values (3–5) keep only strong outlines.",
            parent=parent,
            minvalue=0.1,
            maxvalue=MAX_SIGMA,
            initialvalue=self.DEFAULT_SIGMA,
        )
