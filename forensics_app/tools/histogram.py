"""Plot intensity histograms for the working image."""

from __future__ import annotations

import tkinter as tk

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


def compute_histogram(channel: np.ndarray) -> np.ndarray:
    """Return a length-256 count histogram for an 8-bit channel."""
    if channel.size == 0:
        raise ValueError("Cannot compute a histogram of an empty image.")
    counts, _ = np.histogram(channel.ravel(), bins=256, range=(0, 256))
    return counts.astype(np.int64)


def _figure_to_pil(figure: Figure) -> Image.Image:
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    return Image.fromarray(np.asarray(canvas.buffer_rgba())).convert("RGB")


class HistogramTool(ForensicsTool):
    tool_id = "histogram"
    title = "Histogram visualization"
    category = "Intensity"
    description = "Plot the intensity histogram (R/G/B or grayscale)."

    ACCEPTED_MODES = ("L", "RGB", "RGBA", "P")

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
                f"This image mode is not supported for a histogram (mode {current.mode!r}).\n\n"
                "Open an L or RGB image."
            )
        return (
            "The histogram is already shown.\n\n"
            "Undo or reset the image to plot it again."
        )

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult:
        source = document.current
        assert source is not None
        assert source.mode in self.ACCEPTED_MODES, f"unsupported mode {source.mode!r}"

        if source.width < 1 or source.height < 1:
            raise ValueError("Image is empty; nothing to plot.")

        if source.mode in ("RGB", "RGBA"):
            rgb = source if source.mode == "RGB" else source.convert("RGB")
            channels = {
                "Red": (np.asarray(rgb.getchannel("R")), "#e64646"),
                "Green": (np.asarray(rgb.getchannel("G")), "#46c85a"),
                "Blue": (np.asarray(rgb.getchannel("B")), "#508cf0"),
            }
        else:
            gray = source if source.mode == "L" else source.convert("L")
            channels = {"Gray": (np.asarray(gray), "#444444")}

        figure = Figure(figsize=(7.5, 4.5), dpi=120, layout="constrained")
        axis = figure.add_subplot(1, 1, 1)
        details: dict[str, object] = {"Operation": self.title, "Bins": 256}
        x = np.arange(256)

        for name, (channel, color) in channels.items():
            counts = compute_histogram(channel)
            axis.plot(x, counts, color=color, linewidth=1.2, label=name)
            details[f"{name} peak bin"] = int(counts.argmax())
            details[f"{name} peak count"] = int(counts.max())

        axis.set_xlim(0, 255)
        axis.set_xlabel("Pixel intensity")
        axis.set_ylabel("Frequency")
        axis.set_title("Intensity histogram")
        axis.legend(loc="upper right")
        axis.grid(True, alpha=0.3)

        return ToolResult(
            image=_figure_to_pil(figure),
            message="Plotted the intensity histogram.",
            details=details,
        )
