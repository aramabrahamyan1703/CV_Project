"""Classic filters from ``skimage.filters``: Gaussian, median, Sobel, unsharp mask."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, simpledialog

import numpy as np
from PIL import Image
from skimage import filters

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


ACCEPTED_MODES = ("L", "RGB", "RGBA")

MAX_SIGMA = 50.0
MAX_MEDIAN_SIZE = 25
MAX_UNSHARP_RADIUS = 50.0
MAX_UNSHARP_AMOUNT = 10.0


def _check_image(image: Image.Image, operation: str) -> None:
    if image.mode not in ACCEPTED_MODES:
        raise ValueError(
            f"{operation} needs an L, RGB or RGBA image; got mode {image.mode!r}."
        )
    if image.width < 1 or image.height < 1:
        raise ValueError("Image is empty; nothing to filter.")


def _float_to_uint8(values: np.ndarray) -> np.ndarray:
    """Map skimage's 0..1 float output back to 0..255, rounding not truncating."""
    return np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8)


def _with_colour_bands(image: Image.Image, filtered: np.ndarray) -> Image.Image:
    """Put filtered L/RGB values back into an image, keeping alpha if present."""
    if image.mode == "L":
        return Image.fromarray(filtered, mode="L")
    out = np.asarray(image).copy()
    out[..., :3] = filtered
    return Image.fromarray(out, mode=image.mode)


def _colour_bands(image: Image.Image) -> np.ndarray:
    """The L band, or the three colour bands without alpha."""
    arr = np.asarray(image)
    return arr if image.mode == "L" else arr[..., :3]


def gaussian_blur(image: Image.Image, sigma: float) -> Image.Image:
    """Blur with ``skimage.filters.gaussian``; larger ``sigma`` blurs more.

    Colour images are blurred per channel (``channel_axis=-1``) so colours do
    not bleed into each other; alpha is preserved.
    """
    _check_image(image, "Gaussian blur")
    if not 0.0 < sigma <= MAX_SIGMA:
        raise ValueError(f"Sigma must be above 0 and at most {MAX_SIGMA:g}; got {sigma:g}.")

    bands = _colour_bands(image).astype(np.float64) / 255.0
    channel_axis = None if image.mode == "L" else -1
    blurred = filters.gaussian(bands, sigma=sigma, channel_axis=channel_axis)
    return _with_colour_bands(image, _float_to_uint8(blurred))


def median_filter(image: Image.Image, size: int) -> Image.Image:
    """Replace each pixel by the median of a ``size``×``size`` window.

    Uses ``skimage.filters.median``, which removes salt-and-pepper noise while
    keeping edges sharper than a blur would. Each channel is filtered on its
    own; alpha is preserved.

    ``behavior="rank"`` uses skimage's histogram-based median for 8-bit
    images: its cost barely grows with the window (a 9×9 window on a 12 MP
    photo takes seconds instead of ~25 s). At the image border it takes the
    median of the pixels inside the image only.
    """
    _check_image(image, "Median filter")
    if size < 3 or size % 2 == 0 or size > MAX_MEDIAN_SIZE:
        raise ValueError(
            f"Window size must be odd, between 3 and {MAX_MEDIAN_SIZE}; got {size}."
        )

    footprint = np.ones((size, size), dtype=bool)
    # A writable copy: the rank median refuses PIL's read-only buffer.
    bands = np.array(_colour_bands(image))
    if image.mode == "L":
        return _with_colour_bands(image, filters.median(bands, footprint, behavior="rank"))

    filtered = np.stack(
        [filters.median(bands[..., i], footprint, behavior="rank") for i in range(3)], axis=-1
    )
    return _with_colour_bands(image, filtered)


def sobel_edges(image: Image.Image) -> tuple[Image.Image, float]:
    """Sobel gradient magnitude as an L image, plus the raw maximum response.

    Edges are measured on the grayscale (luminance) image, since an edge map
    has one value per pixel. ``skimage.filters.sobel`` returns small floats, so
    the result is scaled so that the strongest edge becomes 255; a flat image
    gives an all-black map.
    """
    _check_image(image, "Sobel edge detection")
    gray = np.asarray(image.convert("L")).astype(np.float64) / 255.0
    magnitude = filters.sobel(gray)

    peak = float(magnitude.max())
    # A flat image still gives ~1e-17 of rounding noise; scaling that up to
    # 255 would invent edges, so anything far below one grey level is none.
    if peak < 1e-6:
        return Image.new("L", image.size, 0), 0.0
    magnitude = magnitude / peak
    return Image.fromarray(_float_to_uint8(magnitude), mode="L"), peak


def unsharp_mask(image: Image.Image, radius: float, amount: float) -> Image.Image:
    """Sharpen with ``skimage.filters.unsharp_mask``.

    The image is blurred with a Gaussian of ``radius`` (sigma), and the
    difference between the image and the blur -- the fine detail -- is added
    back ``amount`` times: ``out = image + amount * (image - blurred)``.
    ``amount=0`` changes nothing. Colour channels are sharpened separately;
    alpha is preserved.
    """
    _check_image(image, "Unsharp mask")
    if not 0.0 < radius <= MAX_UNSHARP_RADIUS:
        raise ValueError(
            f"Radius must be above 0 and at most {MAX_UNSHARP_RADIUS:g}; got {radius:g}."
        )
    if not 0.0 <= amount <= MAX_UNSHARP_AMOUNT:
        raise ValueError(
            f"Amount must be between 0 and {MAX_UNSHARP_AMOUNT:g}; got {amount:g}."
        )

    bands = _colour_bands(image).astype(np.float64) / 255.0
    # Must be 2, not -1: skimage's unsharp_mask slices channels with
    # slice_at_axis(), which treats -1 as the first axis (image rows) and
    # scrambles colour images. gaussian() handles -1 correctly.
    channel_axis = None if image.mode == "L" else 2
    sharpened = filters.unsharp_mask(
        bands, radius=radius, amount=amount, channel_axis=channel_axis
    )
    return _with_colour_bands(image, _float_to_uint8(sharpened))


class _SkimageFilterTool(ForensicsTool):
    """Shared mode check for the tools in this module."""

    category = "Filtering"
    ACCEPTED_MODES = ACCEPTED_MODES

    def is_available(self, document: ImageDocument) -> bool:
        # No last_tool_id rule: filtering a filtered image is meaningful.
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


class GaussianFilterTool(_SkimageFilterTool):
    tool_id = "skimage_gaussian"
    title = "Gaussian blur (skimage)"
    description = "Blur with skimage.filters.gaussian; choose sigma."

    DEFAULT_SIGMA = 2.0

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        source = document.current
        assert source is not None
        assert source.mode in self.ACCEPTED_MODES, f"unsupported mode {source.mode!r}"

        sigma = self._ask_sigma(parent)
        if sigma is None:
            return None

        output = gaussian_blur(source, sigma)
        return ToolResult(
            image=output,
            message=f"Applied a Gaussian blur (sigma {sigma:g}).",
            details={
                "Operation": self.title,
                "Function": "skimage.filters.gaussian",
                "Sigma": sigma,
                "Output mode": output.mode,
            },
        )

    def _ask_sigma(self, parent: tk.Misc | None) -> float | None:
        return simpledialog.askfloat(
            self.title,
            "Sigma (standard deviation, in pixels).\n\n"
            "Larger values blur more; 1–3 is a mild blur.",
            parent=parent,
            minvalue=0.1,
            maxvalue=MAX_SIGMA,
            initialvalue=self.DEFAULT_SIGMA,
        )


class MedianFilterTool(_SkimageFilterTool):
    tool_id = "skimage_median"
    title = "Median filter (skimage)"
    description = "Remove salt-and-pepper noise with skimage.filters.median."

    DEFAULT_SIZE = 3

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        source = document.current
        assert source is not None
        assert source.mode in self.ACCEPTED_MODES, f"unsupported mode {source.mode!r}"

        size = self._ask_size(parent)
        if size is None:
            return None

        output = median_filter(source, size)
        return ToolResult(
            image=output,
            message=f"Applied a {size}×{size} median filter.",
            details={
                "Operation": self.title,
                "Function": "skimage.filters.median",
                "Window": f"{size}×{size}",
                "Output mode": output.mode,
            },
        )

    def _ask_size(self, parent: tk.Misc | None) -> int | None:
        """Re-ask until the size is odd, like the channel swap's order prompt."""
        size = self.DEFAULT_SIZE
        while True:
            size = simpledialog.askinteger(
                self.title,
                f"Window size (odd, 3–{MAX_MEDIAN_SIZE}).\n\n"
                "Each pixel becomes the median of the size×size square around it.",
                parent=parent,
                minvalue=3,
                maxvalue=MAX_MEDIAN_SIZE,
                initialvalue=size,
            )
            if size is None or size % 2 == 1:
                return size
            messagebox.showerror(
                self.title, f"The window size must be odd; got {size}.", parent=parent
            )


class SobelFilterTool(_SkimageFilterTool):
    tool_id = "skimage_sobel"
    title = "Sobel edges (skimage)"
    description = "Edge magnitude map with skimage.filters.sobel."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult:
        source = document.current
        assert source is not None
        assert source.mode in self.ACCEPTED_MODES, f"unsupported mode {source.mode!r}"

        output, peak = sobel_edges(source)
        return ToolResult(
            image=output,
            message="Computed the Sobel edge magnitude.",
            details={
                "Operation": self.title,
                "Function": "skimage.filters.sobel",
                "Input": "grayscale (luminance)" if source.mode != "L" else "grayscale",
                "Strongest edge (raw)": f"{peak:.4f}",
                "Scaling": "strongest edge → 255" if peak > 0 else "flat image, no edges",
                "Output mode": output.mode,
            },
        )


class UnsharpMaskTool(_SkimageFilterTool):
    tool_id = "skimage_unsharp_mask"
    title = "Unsharp mask (skimage)"
    description = "Sharpen detail with skimage.filters.unsharp_mask; choose radius and amount."

    DEFAULT_RADIUS = 2.0
    DEFAULT_AMOUNT = 1.0

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        source = document.current
        assert source is not None
        assert source.mode in self.ACCEPTED_MODES, f"unsupported mode {source.mode!r}"

        options = self._ask_options(parent)
        if options is None:
            return None
        radius, amount = options

        output = unsharp_mask(source, radius, amount)
        return ToolResult(
            image=output,
            message=f"Sharpened with an unsharp mask (radius {radius:g}, amount {amount:g}).",
            details={
                "Operation": self.title,
                "Function": "skimage.filters.unsharp_mask",
                "Radius (sigma)": radius,
                "Amount": amount,
                "Output mode": output.mode,
            },
        )

    def _ask_options(self, parent: tk.Misc | None) -> tuple[float, float] | None:
        """Ask for radius then amount; cancelling either dialog aborts."""
        radius = simpledialog.askfloat(
            self.title,
            "Radius (sigma of the blur, in pixels).\n\n"
            "Small values (1–2) sharpen fine texture; larger ones boost\n"
            "broader shapes and add halos around edges.",
            parent=parent,
            minvalue=0.1,
            maxvalue=MAX_UNSHARP_RADIUS,
            initialvalue=self.DEFAULT_RADIUS,
        )
        if radius is None:
            return None

        amount = simpledialog.askfloat(
            self.title,
            "Amount (how strongly the detail is added back).\n\n"
            "0 changes nothing, 1 is a normal sharpen, 2–3 is strong.",
            parent=parent,
            minvalue=0.0,
            maxvalue=MAX_UNSHARP_AMOUNT,
            initialvalue=self.DEFAULT_AMOUNT,
        )
        if amount is None:
            return None
        return float(radius), float(amount)
