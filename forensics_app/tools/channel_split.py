"""Show the R, G and B channels of the working image side by side."""

from __future__ import annotations

import math
import tkinter as tk

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


class ChannelSplitTool(ForensicsTool):
    tool_id = "channel_split"
    title = "Channel split (RGB)"
    category = "Color"
    description = "Show the R, G and B channels side by side with labels."

    # One panel per channel, in the order PIL hands them out.
    LABELS = (
        ("Red",   "#e64646"),
        ("Green", "#46c85a"),
        ("Blue",  "#508cf0"),
    )
    PANELS = len(LABELS)

    ACCEPTED_MODES = ("RGB", "RGBA", "P")

    # Contact-sheet geometry, in inches. The sheet is PANELS panels wide; its
    # height follows the source aspect ratio, plus TITLE_IN for the row of
    # titles, and stops at MAX_FIG_IN so a very tall image cannot grow an
    # endless sheet.
    PANEL_IN = 5.0
    TITLE_IN = 0.6
    MAX_FIG_IN = PANELS * PANEL_IN

    # dpi grows with the source so panels stay near full resolution, then gives
    # way to MAX_PIXELS so the rendered sheet stays a reasonable size in memory.
    MIN_DPI = 100
    MAX_DPI = 400
    MAX_PIXELS = 12_000_000

    def is_available(self, document: ImageDocument) -> bool:
        """Require a colour image this tool has not already split.

        Splitting its own contact sheet would measure the labels and the white
        background rather than the evidence, and splitting a single-channel
        image would report one grey channel three times. Either way the sidebar
        greys the button out until the user undoes or resets.
        """
        if document.current is None:
            return True  # let the main window ask for an image first
        if document.current.mode not in self.ACCEPTED_MODES:
            return False
        return document.last_tool_id != self.tool_id

    def unavailable_message(self, document: ImageDocument) -> str:
        """Name whichever rule in :meth:`is_available` refused the image."""
        current = document.current
        if current is not None and current.mode not in self.ACCEPTED_MODES:
            # Only advise undo/reset when there is a colour version to go back
            # to; a file that opened as grayscale never had one.
            original = document.original
            if original is not None and original.mode in self.ACCEPTED_MODES:
                advice = "Undo or reset the image to get the colour version back."
            else:
                advice = "Open an RGB image to use this tool."
            return (
                f"This image is not RGB (mode {current.mode!r}).\n\n"
                "A channel split needs an image with separate red, green and "
                f"blue channels, and this one does not have them.\n\n{advice}"
            )
        return (
            "The channels are already split.\n\n"
            "The working image is the R/G/B contact sheet, so there is nothing "
            "left to split.\n\n"
            "Undo or reset the image to split it again."
        )

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult:
        """Render the three channels as one labelled sheet, with their ranges.

        Returns the sheet as the new working image, so the result is undoable
        like any other operation, plus each channel's min/max for the Results
        panel. Never cancels, hence the non-optional return type.
        """
        source = document.current
        # Both preconditions are guaranteed by is_available(), which the main
        # window checks before it enables the button.
        assert source is not None
        assert source.mode in self.ACCEPTED_MODES, f"unsupported mode {source.mode!r}"

        # The only conversion left to do is dropping an alpha band, so that
        # split() yields exactly the three colour channels. It keeps the stored
        # R/G/B values of transparent pixels rather than compositing them.
        rgb = source if source.mode == "RGB" else source.convert("RGB")

        # Equivalent to cv2.split(), minus the reordering: PIL hands back
        # R, G, B, whereas cv2.imread() gives B, G, R.
        channels = [np.asarray(band) for band in rgb.split()]

        figure = self._new_figure(*rgb.size)
        axes = figure.subplots(1, self.PANELS)

        details = {"Operation": self.title}
        for axis, (name, color), channel in zip(axes, self.LABELS, channels):
            # vmin/vmax fixed to 0..255 so the channels share one scale; plain
            # imshow would stretch each channel to its own min/max and make a
            # flat channel look like full-range detail.
            axis.imshow(channel, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            axis.set_title(f"{name} channel", color=color, fontweight="bold")
            axis.set_xticks([])
            axis.set_yticks([])

            details[f"{name} min / max"] = f"{channel.min()} / {channel.max()}"

        return ToolResult(
            image=self._to_pil(figure),
            message="Split into R, G and B channels.",
            details=details,
        )

    def _new_figure(self, width: int, height: int) -> Figure:
        """Build the off-screen figure the three panels are drawn into.

        Size and resolution are derived together: the figure is as tall as the
        source aspect ratio asks for, and the dpi is whatever keeps both the
        panel detail and the total pixel count within bounds.
        """
        fig_w = self.PANELS * self.PANEL_IN
        fig_h = min(self.PANEL_IN * height / width + self.TITLE_IN, self.MAX_FIG_IN)

        # A panel occupies roughly 90% of its box once constrained layout has
        # taken its margins, so this is the dpi at which one source pixel maps
        # to about one sheet pixel.
        native_dpi = width / (self.PANEL_IN * 0.9)
        dpi = min(self.MAX_DPI, max(self.MIN_DPI, native_dpi))
        dpi = min(dpi, math.sqrt(self.MAX_PIXELS / (fig_w * fig_h)))

        return Figure(figsize=(fig_w, fig_h), dpi=int(dpi), layout="constrained")

    @staticmethod
    def _to_pil(figure: Figure) -> Image.Image:
        """Render a matplotlib figure off-screen and return it as a PIL RGB image."""
        canvas = FigureCanvasAgg(figure)
        canvas.draw()
        return Image.fromarray(np.asarray(canvas.buffer_rgba())).convert("RGB")
