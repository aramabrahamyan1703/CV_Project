"""Reorder the R, G and B channels of the working image."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, simpledialog

from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


# Position of each channel letter in a PIL RGB image, and the inverse lookup.
CHANNEL_INDEX = {"R": 0, "G": 1, "B": 2}
CHANNEL_NAMES = ("R", "G", "B")

# The permutation that changes nothing, i.e. the order the image already has.
IDENTITY_ORDER = (0, 1, 2)


def parse_order(text: str) -> tuple[int, int, int]:
    """Turn an order such as ``"BGR"`` into source indices for each output slot.

    The letters name the *source* channel that lands in each output slot, so
    ``"BGR"`` means "red gets the source blue, green keeps green, blue gets the
    source red" and returns ``(2, 1, 0)``. Reading it the other way round -- as
    where each source channel goes -- gives the same answer for a reversal but
    the opposite one for a three-way rotation, so the direction matters.

    Raises:
        ValueError: with a message written for the user, for anything that is
            not a genuine reordering: a wrong length, a letter outside R/G/B, a
            repeated letter, or ``"RGB"`` itself.
    """
    letters = text.strip().upper()
    if len(letters) != 3:
        raise ValueError(
            f"Enter exactly three letters, one per channel; got {len(letters)}."
        )

    unknown = sorted(set(letters) - CHANNEL_INDEX.keys())
    if unknown:
        listed = ", ".join(repr(letter) for letter in unknown)
        raise ValueError(f"Use only the letters R, G and B; {listed} is not one of them.")

    # Rejects "RRG" and friends: dropping a channel twice over would lose a
    # third of the evidence rather than rearrange it.
    if len(set(letters)) != 3:
        raise ValueError("Use each of R, G and B exactly once, in any order.")

    red, green, blue = (CHANNEL_INDEX[letter] for letter in letters)
    order = (red, green, blue)
    if order == IDENTITY_ORDER:
        raise ValueError(
            "RGB is the order the image already has, so that would change nothing.\n\n"
            "Try BGR, GRB, BRG, GBR or RBG."
        )
    return order


def swap_channels(image: Image.Image, order: tuple[int, int, int]) -> Image.Image:
    """Return ``image`` with its colour channels taken in ``order``.

    Only the three colour bands move; an alpha band rides along untouched, so
    an RGBA image keeps its transparency and its mode. This is the opposite
    choice to :mod:`~forensics_app.tools.channel_split`, which drops alpha: a
    contact sheet of grey panels has no slot to show transparency in, whereas a
    swap is still an ordinary viewable image that should keep it.
    """
    assert image.mode in ("RGB", "RGBA"), f"unsupported mode {image.mode!r}"

    # split() is PIL's equivalent of cv2.split() and hands back R, G, B(, A).
    bands = image.split()
    swapped = [bands[index] for index in order]
    if image.mode == "RGBA":
        swapped.append(bands[3])
    return Image.merge(image.mode, swapped)


class ChannelSwapTool(ForensicsTool):
    tool_id = "channel_swap"
    title = "Channel swap (RGB)"
    category = "Color"
    description = "Reorder the red, green and blue channels, e.g. RGB to BGR."

    ACCEPTED_MODES = ("RGB", "RGBA", "P")

    DEFAULT_ORDER = "BGR"
    PROMPT = (
        "New channel order, for example BGR:\n\n"
        "Each letter names the source channel that goes into that output\n"
        "slot, so BGR moves the source blue channel into the red one."
    )

    def is_available(self, document: ImageDocument) -> bool:
        """Require an image that has three separate colour channels.

        Unlike :class:`~forensics_app.tools.channel_split.ChannelSplitTool`
        there is deliberately no ``document.last_tool_id`` rule here: a swap
        produces an ordinary image whose channels can be swapped again, and
        doing so is useful -- applying BGR twice is how you get back to RGB.
        """
        if document.current is None:
            return True  # let the main window ask for an image first
        return document.current.mode in self.ACCEPTED_MODES

    def unavailable_message(self, document: ImageDocument) -> str:
        """Explain the single rule in :meth:`is_available`."""
        current = document.current
        mode = current.mode if current is not None else "unknown"

        # Only advise undo/reset when there is a colour version to go back to;
        # a file that opened as grayscale never had one.
        original = document.original
        if original is not None and original.mode in self.ACCEPTED_MODES:
            advice = "Undo or reset the image to get the colour version back."
        else:
            advice = "Open an RGB image to use this tool."
        return (
            f"This image is not RGB (mode {mode!r}).\n\n"
            "Swapping channels needs an image with separate red, green and "
            f"blue channels, and this one does not have them.\n\n{advice}"
        )

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        """Ask for an order, apply it, and report which slot took which channel.

        Returns ``None`` when the user cancels the dialog.
        """
        source = document.current
        # Both preconditions are guaranteed by is_available(), which the main
        # window checks before it enables the button.
        assert source is not None
        assert source.mode in self.ACCEPTED_MODES, f"unsupported mode {source.mode!r}"

        order = self._ask_order(parent)
        if order is None:
            return None

        # A palette image stores indices rather than colours, so it has no
        # bands to reorder until it is expanded; RGB and RGBA already do.
        image = source if source.mode in ("RGB", "RGBA") else source.convert("RGB")
        output = swap_channels(image, order)

        order_text = "".join(CHANNEL_NAMES[index] for index in order)
        details = {"Operation": self.title, "New order": order_text}
        # One row per output slot, naming its source, so the Results panel says
        # which way round the swap went rather than leaving it to be inferred.
        for slot, index in zip(CHANNEL_NAMES, order):
            details[f"Output {slot}"] = f"source {CHANNEL_NAMES[index]}"
        details["Output mode"] = output.mode

        return ToolResult(
            image=output,
            message=f"Swapped the channels into {order_text} order.",
            details=details,
        )

    def _ask_order(self, parent: tk.Misc | None) -> tuple[int, int, int] | None:
        """Prompt until the user gives a valid order or cancels.

        Everything Tkinter touches lives here, which keeps :meth:`run` testable
        without a display and keeps a typo out of the error dialog the main
        window shows for a raised exception -- a mistyped order is a nudge to
        try again, not a failed tool.
        """
        text = self.DEFAULT_ORDER
        while True:
            text = simpledialog.askstring(self.title, self.PROMPT, parent=parent, initialvalue=text)
            if text is None:
                return None
            try:
                return parse_order(text)
            except ValueError as error:
                # Re-ask with the rejected text in place so it can be corrected.
                messagebox.showerror(self.title, str(error), parent=parent)
