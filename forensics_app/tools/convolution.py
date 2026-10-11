"""2-D convolution of the working image with a user-supplied kernel."""

from __future__ import annotations

from fractions import Fraction
import re
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


# Largest kernel side accepted. Each kernel entry is one full-image pass, so a
# much larger kernel would freeze the UI thread on a photo-sized image.
MAX_KERNEL_SIDE = 25

# Text shown in the dialog for each preset. Entries may be fractions, so the
# Gaussian reads the way it is written on the slides.
PRESETS: dict[str, str] = {
    "Identity": "0 0 0\n0 1 0\n0 0 0",
    "Box blur 3×3": "1 1 1\n1 1 1\n1 1 1",
    "Gaussian 3×3": "1 2 1\n2 4 2\n1 2 1",
    "Gaussian 5×5": (
        "1 4 6 4 1\n4 16 24 16 4\n6 24 36 24 6\n4 16 24 16 4\n1 4 6 4 1"
    ),
    "Sharpen": "0 -1 0\n-1 5 -1\n0 -1 0",
    "Laplacian (edges)": "0 1 0\n1 -4 1\n0 1 0",
    "Sobel X": "-1 0 1\n-2 0 2\n-1 0 1",
    "Sobel Y": "-1 -2 -1\n0 0 0\n1 2 1",
    "Emboss": "-2 -1 0\n-1 1 1\n0 1 2",
}

# Dialog label -> np.pad mode. "reflect" mirrors without repeating the edge
# pixel, the same as OpenCV's default BORDER_REFLECT_101.
BORDER_MODES = {
    "Reflect": "reflect",
    "Replicate edge": "edge",
    "Zero": "constant",
}

# How values outside 0..255 are brought back into range.
NEGATIVE_MODES = ("clip", "abs")


def parse_kernel(text: str) -> np.ndarray:
    """Turn typed text into a 2-D float kernel.

    Rows are separated by newlines or ``;``; entries by spaces or commas.
    Entries may be integers, decimals or fractions such as ``1/16``.

    Raises:
        ValueError: with a message written for the user, for an empty kernel,
            an entry that is not a number, rows of unequal length, an even
            side (no centre pixel), or a side above :data:`MAX_KERNEL_SIDE`.
    """
    rows: list[list[Fraction]] = []
    for line in re.split(r"[\n;]", text):
        tokens = [token for token in re.split(r"[\s,]+", line.strip()) if token]
        if not tokens:
            continue  # blank lines between rows are harmless
        row: list[Fraction] = []
        for token in tokens:
            try:
                row.append(Fraction(token))
            except (ValueError, ZeroDivisionError):
                raise ValueError(f"{token!r} is not a number.") from None
        rows.append(row)

    if not rows:
        raise ValueError("Enter at least one number for the kernel.")

    widths = {len(row) for row in rows}
    if len(widths) != 1:
        raise ValueError(
            "Every row must have the same number of entries; "
            f"got rows of lengths {', '.join(str(len(row)) for row in rows)}."
        )

    height, width = len(rows), len(rows[0])
    if height % 2 == 0 or width % 2 == 0:
        raise ValueError(
            f"The kernel is {height}×{width}; both sides must be odd "
            "(1, 3, 5, …) so it has a centre pixel."
        )
    if height > MAX_KERNEL_SIDE or width > MAX_KERNEL_SIDE:
        raise ValueError(
            f"The kernel is {height}×{width}; the largest side allowed is "
            f"{MAX_KERNEL_SIDE}."
        )

    return np.array([[float(value) for value in row] for row in rows], dtype=np.float64)


def normalization_factor(kernel: np.ndarray) -> float | None:
    """Return ``1 / sum(kernel)``, or ``None`` when the sum is zero.

    Zero-sum kernels (Laplacian, Sobel) measure change rather than average
    brightness, so they are left as they are instead of dividing by zero.
    """
    total = float(kernel.sum())
    if abs(total) < 1e-12:
        return None
    return 1.0 / total


def convolve2d(channel: np.ndarray, kernel: np.ndarray, border: str = "reflect") -> np.ndarray:
    """Convolve one 2-D channel with ``kernel`` and return a float64 array.

    This is true convolution, so the kernel is flipped in both directions
    before sliding it -- the result differs from correlation (and from
    ``cv2.filter2D``) for asymmetric kernels such as Sobel or emboss. The
    output has the same shape as ``channel``; ``border`` is an ``np.pad``
    mode that decides what lies beyond the image edge.
    """
    if border not in BORDER_MODES.values():
        raise ValueError(f"Unknown border mode {border!r}.")
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(channel.astype(np.float64), ((ph, ph), (pw, pw)), mode=border)
    flipped = kernel[::-1, ::-1]

    height, width = channel.shape
    out = np.zeros((height, width), dtype=np.float64)
    # One shifted copy of the image per kernel entry: vectorised over pixels,
    # and fast for the small kernels this tool allows.
    for i in range(kh):
        for j in range(kw):
            weight = flipped[i, j]
            if weight != 0.0:
                out += weight * padded[i : i + height, j : j + width]
    return out


def convolve_image(
    image: Image.Image,
    kernel: np.ndarray,
    *,
    normalize: bool = True,
    border: str = "reflect",
    negatives: str = "clip",
) -> Image.Image:
    """Return ``image`` convolved with ``kernel``, each colour band on its own.

    ``normalize`` divides by the kernel sum when it is non-zero, so blur
    kernels keep the overall brightness. ``negatives`` decides how results
    outside 0..255 are handled: ``"clip"`` cuts them off, ``"abs"`` takes the
    magnitude first so both sides of an edge show up. Alpha is preserved.
    """
    if image.mode not in ("L", "RGB", "RGBA"):
        raise ValueError(
            f"Convolution needs an L, RGB or RGBA image; got mode {image.mode!r}."
        )
    if image.width < 1 or image.height < 1:
        raise ValueError("Image is empty; nothing to convolve.")
    if negatives not in NEGATIVE_MODES:
        raise ValueError(f"Unknown handling for negative values {negatives!r}.")

    weights = kernel.astype(np.float64)
    if normalize:
        factor = normalization_factor(weights)
        if factor is not None:
            weights = weights * factor

    def to_uint8(values: np.ndarray) -> np.ndarray:
        if negatives == "abs":
            values = np.abs(values)
        # Round rather than truncate: truncation biases every blur darker.
        return np.rint(np.clip(values, 0, 255)).astype(np.uint8)

    arr = np.asarray(image)
    if image.mode == "L":
        return Image.fromarray(to_uint8(convolve2d(arr, weights, border)), mode="L")

    out = arr.copy()
    for band in range(3):
        out[..., band] = to_uint8(convolve2d(arr[..., band], weights, border))
    return Image.fromarray(out, mode=image.mode)


def format_kernel(kernel: np.ndarray) -> str:
    """Compact one-line form for the Results panel, e.g. ``[1 2 1; 2 4 2; …]``."""
    rows = ("  ".join(f"{value:g}" for value in row) for row in kernel)
    return "[" + "; ".join(rows) + "]"


class ConvolutionTool(ForensicsTool):
    tool_id = "convolution"
    title = "Convolution (custom kernel)"
    category = "Filtering"
    description = "Convolve the image with any odd-sized kernel you type, or a preset."

    ACCEPTED_MODES = ("L", "RGB", "RGBA")
    DEFAULT_PRESET = "Sharpen"

    def is_available(self, document: ImageDocument) -> bool:
        # No last_tool_id rule: convolving a convolved image is meaningful
        # (e.g. blur twice), the same reasoning as the channel swap tool.
        if document.current is None:
            return True
        return document.current.mode in self.ACCEPTED_MODES

    def unavailable_message(self, document: ImageDocument) -> str:
        current = document.current
        mode = current.mode if current is not None else "unknown"
        return (
            f"This image mode is not supported for convolution (mode {mode!r}).\n\n"
            "Open an L, RGB or RGBA image."
        )

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        source = document.current
        assert source is not None
        assert source.mode in self.ACCEPTED_MODES, f"unsupported mode {source.mode!r}"

        options = self._ask_kernel_options(parent)
        if options is None:
            return None
        kernel, normalize, border, negatives = options

        output = convolve_image(
            source, kernel, normalize=normalize, border=border, negatives=negatives
        )

        factor = normalization_factor(kernel)
        if not normalize:
            normalized = "off"
        elif factor is None:
            normalized = "not applied (kernel sum is 0)"
        else:
            normalized = f"× {factor:.6g} (1 / {1.0 / factor:g})"
        border_label = next(label for label, mode in BORDER_MODES.items() if mode == border)
        sample = np.asarray(output.convert("L"))

        size = f"{kernel.shape[0]}×{kernel.shape[1]}"
        return ToolResult(
            image=output,
            message=f"Convolved the image with a {size} kernel.",
            details={
                "Operation": self.title,
                "Kernel size": size,
                "Kernel": format_kernel(kernel),
                "Normalization": normalized,
                "Border": border_label,
                "Negative values": "absolute value" if negatives == "abs" else "clipped to 0",
                "Output min / max": f"{int(sample.min())} / {int(sample.max())}",
                "Output mode": output.mode,
            },
        )

    def _ask_kernel_options(
        self, parent: tk.Misc | None
    ) -> tuple[np.ndarray, bool, str, str] | None:
        """One dialog: kernel text, preset, normalize, border, negatives.

        Returns ``(kernel, normalize, border, negatives)`` or ``None`` on
        cancel. A badly typed kernel shows an error and keeps the dialog open
        so it can be corrected.
        """
        dialog = tk.Toplevel(parent)
        dialog.title(self.title)
        dialog.transient(parent)
        dialog.resizable(False, False)

        result: dict[str, tuple[np.ndarray, bool, str, str] | None] = {"value": None}

        frame = ttk.Frame(dialog, padding=14)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="Preset:").grid(row=0, column=0, sticky="w")
        preset_var = tk.StringVar(value=self.DEFAULT_PRESET)
        preset_box = ttk.Combobox(
            frame,
            textvariable=preset_var,
            values=list(PRESETS),
            state="readonly",
            width=22,
        )
        preset_box.grid(row=0, column=1, sticky="w", padx=(8, 0))

        ttk.Label(frame, text="Kernel:").grid(row=1, column=0, sticky="nw", pady=(12, 0))
        kernel_text = tk.Text(frame, width=34, height=7, font=("TkFixedFont", 11), undo=True)
        kernel_text.grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(12, 0))
        kernel_text.insert("1.0", PRESETS[self.DEFAULT_PRESET])

        def load_preset(_event: object = None) -> None:
            kernel_text.delete("1.0", "end")
            kernel_text.insert("1.0", PRESETS[preset_var.get()])

        preset_box.bind("<<ComboboxSelected>>", load_preset)

        ttk.Label(
            frame,
            text="One row per line (or separate rows with ;). Entries may be\n"
            "integers, decimals or fractions like 1/16. Both sides must be\n"
            f"odd, at most {MAX_KERNEL_SIDE}.",
            justify="left",
        ).grid(row=2, column=1, sticky="w", padx=(8, 0), pady=(4, 0))

        normalize_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            frame,
            text="Normalize (divide by the kernel sum, if not 0)",
            variable=normalize_var,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(12, 0))

        ttk.Label(frame, text="Border:").grid(row=4, column=0, sticky="w", pady=(12, 0))
        border_var = tk.StringVar(value="Reflect")
        ttk.Combobox(
            frame,
            textvariable=border_var,
            values=list(BORDER_MODES),
            state="readonly",
            width=22,
        ).grid(row=4, column=1, sticky="w", padx=(8, 0), pady=(12, 0))

        ttk.Label(frame, text="Negative values:").grid(row=5, column=0, sticky="nw", pady=(12, 0))
        negatives_var = tk.StringVar(value="clip")
        negatives_frame = ttk.Frame(frame)
        negatives_frame.grid(row=5, column=1, sticky="w", padx=(8, 0), pady=(12, 0))
        ttk.Radiobutton(
            negatives_frame, text="Clip to 0", variable=negatives_var, value="clip"
        ).pack(anchor="w")
        ttk.Radiobutton(
            negatives_frame,
            text="Absolute value (shows both sides of edges)",
            variable=negatives_var,
            value="abs",
        ).pack(anchor="w")

        def accept() -> None:
            try:
                kernel = parse_kernel(kernel_text.get("1.0", "end"))
            except ValueError as error:
                messagebox.showerror(self.title, str(error), parent=dialog)
                return
            result["value"] = (
                kernel,
                bool(normalize_var.get()),
                BORDER_MODES[border_var.get()],
                negatives_var.get(),
            )
            dialog.destroy()

        def cancel() -> None:
            result["value"] = None
            dialog.destroy()

        buttons = ttk.Frame(frame)
        buttons.grid(row=6, column=0, columnspan=2, sticky="e", pady=(16, 0))
        ttk.Button(buttons, text="Cancel", command=cancel).pack(side="right")
        ttk.Button(buttons, text="Apply", command=accept).pack(side="right", padx=(0, 8))

        dialog.protocol("WM_DELETE_WINDOW", cancel)
        dialog.grab_set()
        kernel_text.focus_set()
        dialog.wait_window()
        return result["value"]
