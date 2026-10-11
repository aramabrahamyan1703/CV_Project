"""Unsharp-mask image sharpening (Set 3.5)."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, simpledialog

import numpy as np
from PIL import Image
from skimage.filters import gaussian

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


def sharpen_image(
    image: Image.Image,
    *,
    radius: float = 2.0,
    amount: float = 1.5,
) -> Image.Image:
    """Sharpen with the Set 3.5 formula: ``original + amount * (original - blur)``.

    Uses a Gaussian blur (``radius`` as sigma). Colour images are blurred per
    channel so RGB is not treated as a 3D volume (that path can go black).
    """
    if image.mode not in ("L", "RGB", "RGBA", "P"):
        raise ValueError(
            f"Sharpening needs an L, RGB or RGBA image; got mode {image.mode!r}."
        )
    if radius <= 0:
        raise ValueError(f"Radius must be positive; got {radius:g}.")
    if amount < 0:
        raise ValueError(f"Amount must be ≥ 0; got {amount:g}.")
    if image.width < 2 or image.height < 2:
        raise ValueError("Image is too small to sharpen.")

    if image.mode == "L":
        working = image
        mode = "L"
    else:
        working = image.convert("RGB")
        mode = "RGB"

    # Explicit [0, 1] float — avoids PIL/skimage dtype surprises.
    arr = np.asarray(working, dtype=np.float64) / 255.0
    channel_axis = None if arr.ndim == 2 else -1
    blurred = gaussian(arr, sigma=radius, channel_axis=channel_axis)
    sharpened = np.clip(arr + amount * (arr - blurred), 0.0, 1.0)
    out = Image.fromarray((sharpened * 255.0 + 0.5).astype(np.uint8), mode=mode)

    if image.mode == "RGBA":
        out = out.convert("RGBA")
        out.putalpha(image.getchannel("A"))
    return out


class SharpenTool(ForensicsTool):
    tool_id = "sharpen"
    title = "Image sharpening"
    category = "Filtering"
    description = "Unsharp-mask sharpening with configurable amount (Set 3.5)."

    ACCEPTED_MODES = ("L", "RGB", "RGBA", "P")
    DEFAULT_RADIUS = 2.0
    DEFAULT_AMOUNT = 1.5

    def is_available(self, document: ImageDocument) -> bool:
        if document.current is None:
            return True
        return document.current.mode in self.ACCEPTED_MODES

    def unavailable_message(self, document: ImageDocument) -> str:
        current = document.current
        mode = current.mode if current is not None else "unknown"
        return (
            f"This image mode is not supported for sharpening (mode {mode!r}).\n\n"
            "Open an L or RGB image."
        )

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        source = document.current
        assert source is not None

        params = self._ask_params(parent)
        if params is None:
            return None
        radius, amount = params

        try:
            output = sharpen_image(source, radius=radius, amount=amount)
        except ValueError as error:
            messagebox.showerror(self.title, str(error), parent=parent)
            return None

        return ToolResult(
            image=output,
            message=f"Sharpened image (radius={radius:g}, amount={amount:g}).",
            details={
                "Operation": self.title,
                "Radius": radius,
                "Amount": amount,
                "Output mode": output.mode,
            },
        )

    def _ask_params(self, parent: tk.Misc | None) -> tuple[float, float] | None:
        radius = simpledialog.askfloat(
            self.title,
            "Blur radius (Gaussian sigma before adding detail back).\n"
            "Typical value: 2",
            parent=parent,
            minvalue=0.1,
            initialvalue=self.DEFAULT_RADIUS,
        )
        if radius is None:
            return None

        amount = simpledialog.askfloat(
            self.title,
            "Sharpening amount (how much detail to add back).\n"
            "Typical value: 1.5",
            parent=parent,
            minvalue=0.0,
            initialvalue=self.DEFAULT_AMOUNT,
        )
        if amount is None:
            return None
        return float(radius), float(amount)
