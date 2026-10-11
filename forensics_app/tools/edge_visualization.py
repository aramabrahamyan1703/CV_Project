"""Superimpose detected edges on the working image (Set 3.4)."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, simpledialog

import numpy as np
from PIL import Image
from skimage import color, feature

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult

# Styles shown on the Set 3.4 slide.
STYLES = ("red", "dimmed", "contour")


def detect_edges(gray_float: np.ndarray, sigma: float) -> np.ndarray:
    """Boolean Canny edge map for a float grayscale image in ``[0, 1]``."""
    if gray_float.size == 0:
        raise ValueError("Image is empty; nothing to find edges on.")
    if sigma <= 0:
        raise ValueError(f"Sigma must be positive; got {sigma:g}.")
    return feature.canny(gray_float, sigma=sigma)


def superimpose_edges(
    image: Image.Image,
    *,
    sigma: float = 2.0,
    style: str = "red",
) -> tuple[Image.Image, dict[str, object]]:
    """Detect Canny edges and draw them on top of ``image``.

    Styles:
      - ``red``: solid red edge pixels on the original (slide option 1)
      - ``dimmed``: dimmed original with bright white edges (slide option 3)
      - ``contour``: thin red contours via a rendered plot-free pixel dilation
    """
    if image.mode not in ("L", "RGB", "RGBA", "P"):
        raise ValueError(
            f"Edge visualization needs an L, RGB or RGBA image; got mode {image.mode!r}."
        )
    style = style.lower().strip()
    if style not in STYLES:
        raise ValueError(f"Unknown style {style!r}; use one of {', '.join(STYLES)}.")

    rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    gray = color.rgb2gray(rgb)
    edges = detect_edges(gray, sigma)
    if not edges.any():
        raise ValueError(
            f"No edges found with sigma={sigma:g}.\n\n"
            "Try a smaller sigma, or a more detailed image."
        )

    if style == "red":
        out = rgb.copy()
        out[edges] = (1.0, 0.0, 0.0)
    elif style == "dimmed":
        out = rgb * 0.4
        out[edges] = (1.0, 1.0, 1.0)
    else:
        # Contour-like: thicken edge pixels slightly and paint red.
        out = rgb.copy()
        thick = edges.copy()
        thick[1:, :] |= edges[:-1, :]
        thick[:-1, :] |= edges[1:, :]
        thick[:, 1:] |= edges[:, :-1]
        thick[:, :-1] |= edges[:, 1:]
        out[thick] = (1.0, 0.0, 0.0)

    details = {
        "Sigma": sigma,
        "Style": style,
        "Edge pixels": f"{int(edges.sum())} / {edges.size}",
        "Output mode": "RGB",
    }
    # Same explicit uint8 path as sharpening — avoids skimage dtype surprises.
    pixels = (np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB"), details


class EdgeVisualizationTool(ForensicsTool):
    tool_id = "edge_visualization"
    title = "Edge visualization"
    category = "Filtering"
    description = "Canny edges superimposed on the original image (Set 3.4)."

    ACCEPTED_MODES = ("L", "RGB", "RGBA", "P")
    DEFAULT_SIGMA = 2.0
    DEFAULT_STYLE = "red"

    def is_available(self, document: ImageDocument) -> bool:
        if document.current is None:
            return True
        return document.current.mode in self.ACCEPTED_MODES

    def unavailable_message(self, document: ImageDocument) -> str:
        current = document.current
        mode = current.mode if current is not None else "unknown"
        return (
            f"This image mode is not supported for edge visualization (mode {mode!r}).\n\n"
            "Open an L or RGB image."
        )

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        source = document.current
        assert source is not None

        params = self._ask_params(parent)
        if params is None:
            return None
        sigma, style = params

        try:
            output, info = superimpose_edges(source, sigma=sigma, style=style)
        except ValueError as error:
            messagebox.showinfo(self.title, str(error), parent=parent)
            return None

        return ToolResult(
            image=output,
            message=f"Superimposed {style} edges (sigma={sigma:g}).",
            details={"Operation": self.title, **info},
        )

    def _ask_params(self, parent: tk.Misc | None) -> tuple[float, str] | None:
        sigma = simpledialog.askfloat(
            self.title,
            "Canny sigma (smoothing before edges).\n"
            "Larger sigma → fewer, smoother edges.",
            parent=parent,
            minvalue=0.1,
            initialvalue=self.DEFAULT_SIGMA,
        )
        if sigma is None:
            return None

        style = simpledialog.askstring(
            self.title,
            "Visualization style:\n"
            "  red     – red edges on the original\n"
            "  dimmed  – white edges on a dimmed image\n"
            "  contour – thicker red outline\n",
            parent=parent,
            initialvalue=self.DEFAULT_STYLE,
        )
        if style is None:
            return None
        style = style.strip().lower()
        if style not in STYLES:
            messagebox.showerror(
                self.title,
                f"Use one of: {', '.join(STYLES)}.",
                parent=parent,
            )
            return None
        return float(sigma), style
