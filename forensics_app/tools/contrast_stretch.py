"""Stretch a narrow intensity range to the full 0–255 scale."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, simpledialog

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


def stretch_contrast(
    image: Image.Image,
    low_percent: float = 2.0,
    high_percent: float = 98.0,
) -> Image.Image:
    """Linearly stretch intensities between two percentiles to 0..255.

    Same idea as ``skimage.exposure.rescale_intensity`` with an ``in_range``
    taken from percentiles (Set 2.5). Each colour band is stretched on its own;
    alpha is preserved when present.
    """
    if image.mode not in ("L", "RGB", "RGBA"):
        raise ValueError(
            f"Contrast stretching needs an L, RGB or RGBA image; got mode {image.mode!r}."
        )
    if not 0.0 <= low_percent < high_percent <= 100.0:
        raise ValueError(
            f"Need 0 ≤ low < high ≤ 100; got low={low_percent}, high={high_percent}."
        )
    if image.width < 1 or image.height < 1:
        raise ValueError("Image is empty; nothing to stretch.")

    arr = np.asarray(image).astype(np.float64)
    if image.mode == "L":
        bands = [arr]
    else:
        bands = [arr[..., i] for i in range(3)]

    stretched_bands: list[np.ndarray] = []
    for band in bands:
        lo, hi = np.percentile(band, (low_percent, high_percent))
        if hi <= lo:
            # Flat (or nearly flat) band: nothing meaningful to stretch.
            raise ValueError(
                "Cannot stretch contrast: the chosen percentiles collapse to "
                f"the same value ({lo:g}). Try a flatter image or wider percentiles."
            )
        scaled = (band - lo) / (hi - lo) * 255.0
        stretched_bands.append(np.clip(scaled, 0, 255))

    if image.mode == "L":
        out = stretched_bands[0].astype(np.uint8)
        return Image.fromarray(out, mode="L")

    out = arr.copy()
    for i, band in enumerate(stretched_bands):
        out[..., i] = band
    return Image.fromarray(out.astype(np.uint8), mode=image.mode)


class ContrastStretchTool(ForensicsTool):
    tool_id = "contrast_stretch"
    title = "Contrast stretching"
    category = "Intensity"
    description = "Stretch intensity percentiles to the full 0–255 range."

    ACCEPTED_MODES = ("L", "RGB", "RGBA")
    DEFAULT_LOW = 2.0
    DEFAULT_HIGH = 98.0

    def is_available(self, document: ImageDocument) -> bool:
        if document.current is None:
            return True
        return document.current.mode in self.ACCEPTED_MODES

    def unavailable_message(self, document: ImageDocument) -> str:
        current = document.current
        mode = current.mode if current is not None else "unknown"
        return (
            f"This image mode is not supported for contrast stretching (mode {mode!r}).\n\n"
            "Open an L, RGB or RGBA image."
        )

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        source = document.current
        assert source is not None
        assert source.mode in self.ACCEPTED_MODES, f"unsupported mode {source.mode!r}"

        bounds = self._ask_percentiles(parent)
        if bounds is None:
            return None
        low, high = bounds

        output = stretch_contrast(source, low, high)
        sample = np.asarray(output.convert("L"))
        return ToolResult(
            image=output,
            message=f"Stretched contrast using the {low:g}–{high:g} percentiles.",
            details={
                "Operation": self.title,
                "Low percentile": low,
                "High percentile": high,
                "Output min / max": f"{int(sample.min())} / {int(sample.max())}",
                "Output mode": output.mode,
            },
        )

    def _ask_percentiles(
        self, parent: tk.Misc | None
    ) -> tuple[float, float] | None:
        """Ask for low then high percentile; cancel either dialog aborts."""
        low = simpledialog.askfloat(
            self.title,
            "Low percentile (0–100).\n\n"
            "Intensities below this become black. Default 2 ignores dark outliers.",
            parent=parent,
            minvalue=0.0,
            maxvalue=100.0,
            initialvalue=self.DEFAULT_LOW,
        )
        if low is None:
            return None

        high = simpledialog.askfloat(
            self.title,
            "High percentile (0–100).\n\n"
            "Intensities above this become white. Default 98 ignores bright outliers.",
            parent=parent,
            minvalue=0.0,
            maxvalue=100.0,
            initialvalue=self.DEFAULT_HIGH,
        )
        if high is None:
            return None

        if not 0.0 <= low < high <= 100.0:
            messagebox.showerror(
                self.title,
                f"Need 0 ≤ low < high ≤ 100; got low={low:g}, high={high:g}.",
                parent=parent,
            )
            return None
        return float(low), float(high)
