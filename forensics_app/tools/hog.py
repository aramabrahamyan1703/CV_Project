"""HOG feature extraction and visualization with ``skimage.feature.hog`` (Set 4.3)."""

from __future__ import annotations

import math
import tkinter as tk
from tkinter import messagebox

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image
from skimage import feature

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .param_dialog import IntField, ParamDialog, parse_int_fields


ACCEPTED_MODES = ("L", "RGB", "RGBA", "P")

# One row per whole-number HOG parameter in the dialog.
HOG_FIELDS = (
    IntField("Orientations", 9, 36, "gradient direction bins per cell"),
    IntField("Pixels per cell", 6, 64, "square cell side, in pixels"),
    IntField("Cells per block", 3, 8, "square block side, in cells"),
)


def parse_hog_params(
    orientations: str, pixels_per_cell: str, cells_per_block: str
) -> tuple[int, int, int]:
    """Validate the dialog's three text fields as whole numbers within limits."""
    values = parse_int_fields(HOG_FIELDS, (orientations, pixels_per_cell, cells_per_block))
    return values[0], values[1], values[2]


def descriptor_layout(
    size: tuple[int, int], *, orientations: int, pixels_per_cell: int, cells_per_block: int
) -> tuple[tuple[int, int], int]:
    """``((cells_x, cells_y), descriptor_length)`` HOG gives for a ``(width, height)`` image.

    Raises ``ValueError`` when the image cannot hold one block. Cells that do
    not fit whole are dropped, as skimage does.
    """
    width, height = size
    block_px = pixels_per_cell * cells_per_block
    if width < block_px or height < block_px:
        raise ValueError(
            f"The image ({width}×{height}) is smaller than one HOG block "
            f"({block_px}×{block_px} px = {cells_per_block}×{cells_per_block} cells "
            f"of {pixels_per_cell} px).\n\n"
            "Use smaller cells or fewer cells per block."
        )
    cells_x, cells_y = width // pixels_per_cell, height // pixels_per_cell
    blocks = (cells_x - cells_per_block + 1) * (cells_y - cells_per_block + 1)
    return (cells_x, cells_y), blocks * cells_per_block**2 * orientations


def extract_hog(
    image: Image.Image,
    *,
    orientations: int,
    pixels_per_cell: int,
    cells_per_block: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(feature_vector, hog_image)`` from ``skimage.feature.hog``.

    Colour images are passed as uint8 RGB with ``channel_axis=2`` exactly as on
    the slide, so skimage keeps the strongest-gradient channel per pixel;
    grayscale images use ``channel_axis=None``. Alpha and palettes are dropped.
    Cells and blocks are square.
    """
    if image.mode not in ACCEPTED_MODES:
        raise ValueError(
            f"HOG needs an L, RGB, RGBA or P image; got mode {image.mode!r}."
        )
    if pixels_per_cell < 1 or cells_per_block < 1:
        raise ValueError("Pixels per cell and cells per block must be at least 1.")
    descriptor_layout(
        image.size,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
    )

    if image.mode == "L":
        pixels, channel_axis = np.asarray(image), None
    else:
        pixels, channel_axis = np.asarray(image.convert("RGB")), 2

    return feature.hog(
        pixels,
        orientations=orientations,
        pixels_per_cell=(pixels_per_cell, pixels_per_cell),
        cells_per_block=(cells_per_block, cells_per_block),
        visualize=True,
        channel_axis=channel_axis,
    )


class HOGTool(ForensicsTool):
    tool_id = "hog"
    title = "HOG features"
    category = "Features"
    description = "Extract a HOG descriptor and visualize its gradients (Set 4.3)."

    ACCEPTED_MODES = ACCEPTED_MODES

    # Sheet geometry in inches, as in ChannelSplitTool: one panel per image,
    # height following the source aspect ratio, dpi chosen so each panel stays
    # near source resolution -- HOG strokes are one pixel wide and blur away on
    # a low-dpi sheet.
    PANEL_IN = 4.5
    TITLE_IN = 0.9
    MIN_DPI = 100
    MAX_DPI = 400
    MAX_PIXELS = 16_000_000

    def is_available(self, document: ImageDocument) -> bool:
        """Refuse unsupported modes and the tool's own output sheet."""
        if document.current is None:
            return True
        if document.current.mode not in self.ACCEPTED_MODES:
            return False
        return document.last_tool_id != self.tool_id

    def unavailable_message(self, document: ImageDocument) -> str:
        current = document.current
        if current is not None and current.mode not in self.ACCEPTED_MODES:
            return (
                f"This image mode is not supported for HOG (mode {current.mode!r}).\n\n"
                "Open an L or RGB image."
            )
        return (
            "The HOG visualization is already shown.\n\n"
            "Its gradients would describe the panel titles and axes, not the "
            "evidence. Undo or reset the image to run HOG again."
        )

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        source = document.current
        assert source is not None
        assert source.mode in self.ACCEPTED_MODES, f"unsupported mode {source.mode!r}"

        params = self._ask_params(parent, source)
        if params is None:
            return None
        orientations, pixels_per_cell, cells_per_block = params

        try:
            descriptor, hog_image = extract_hog(
                source,
                orientations=orientations,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block,
            )
        except ValueError as error:
            messagebox.showinfo(self.title, str(error), parent=parent)
            return None

        return ToolResult(
            image=self._render_sheet(source, orientations, hog_image),
            message=f"Extracted HOG features with orientations={orientations}.",
            details={
                "Operation": self.title,
                "Function": "skimage.feature.hog",
                "Orientations": orientations,
                "Pixels per cell": f"({pixels_per_cell}, {pixels_per_cell})",
                "Cells per block": f"({cells_per_block}, {cells_per_block})",
                "Channel axis": "None (grayscale)" if source.mode == "L" else "2 (RGB)",
                "Descriptor length": int(descriptor.size),
            },
        )

    def _render_sheet(
        self, source: Image.Image, orientations: int, hog_image: np.ndarray
    ) -> Image.Image:
        """The original and its HOG image side by side, laid out like the slide."""
        width, height = source.size
        panels = 2
        fig_w = panels * self.PANEL_IN
        fig_h = min(self.PANEL_IN * height / width + self.TITLE_IN, 2 * self.PANEL_IN)

        native_dpi = width / (self.PANEL_IN * 0.85)
        dpi = min(self.MAX_DPI, max(self.MIN_DPI, native_dpi))
        dpi = min(dpi, math.sqrt(self.MAX_PIXELS / (fig_w * fig_h)))

        figure = Figure(figsize=(fig_w, fig_h), dpi=int(dpi), layout="constrained")
        axes = figure.subplots(1, panels)

        original = source if source.mode in ("L", "RGB") else source.convert("RGB")
        if original.mode == "L":
            axes[0].imshow(np.asarray(original), cmap="gray", vmin=0, vmax=255)
        else:
            axes[0].imshow(np.asarray(original))
        axes[0].set_title("Original image")

        # Plain auto-scaled imshow, as on the slide: the brightest cell sets
        # white, so most strokes look dim against the black background.
        axes[1].imshow(hog_image, cmap="gray")
        axes[1].set_title(f"HoG image with orientation={orientations}")

        canvas = FigureCanvasAgg(figure)
        canvas.draw()
        image = Image.fromarray(np.asarray(canvas.buffer_rgba())).convert("RGB")
        plt.close(figure)
        return image

    def _ask_params(
        self, parent: tk.Misc | None, source: Image.Image
    ) -> tuple[int, int, int] | None:
        """One dialog for all HOG parameters; ``None`` on cancel.

        The footer shows the cell grid and descriptor length for ``source`` as
        the values change, and explains in place when they do not fit it.
        """
        dialog: ParamDialog[tuple[int, int, int]] = ParamDialog(
            parent,
            title=self.title,
            intro="Describe the image by the directions of its edges, "
            "and draw each cell's gradient histogram as a star of strokes.",
            action="Extract",
        )
        variables = dialog.int_fields(dialog.section("HOG descriptor"), HOG_FIELDS)

        def validate() -> tuple[tuple[int, int, int], str]:
            params = parse_hog_params(*(v.get() for v in variables))
            orientations, ppc, cpb = params
            (cells_x, cells_y), length = descriptor_layout(
                source.size, orientations=orientations, pixels_per_cell=ppc, cells_per_block=cpb
            )
            return params, f"{cells_x} × {cells_y} cells · {length:,} values"

        return dialog.show(validate)
