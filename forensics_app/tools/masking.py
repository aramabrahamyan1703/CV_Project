"""Threshold masking: keep bright or dark pixels and zero the rest."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult


def _luminance(image: Image.Image) -> np.ndarray:
    """Intensity used for the threshold decision (L as-is, RGB via BT.601)."""
    arr = np.asarray(image)
    if image.mode == "L":
        return arr.astype(np.float64)
    rgb = arr[..., :3].astype(np.float64)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def build_threshold_mask(
    image: Image.Image,
    threshold: int,
    *,
    keep_above: bool = True,
) -> np.ndarray:
    """Boolean mask: True where the pixel is kept (Set 2.3 binary mask)."""
    if image.mode not in ("L", "RGB", "RGBA"):
        raise ValueError(
            f"Masking needs an L, RGB or RGBA image; got mode {image.mode!r}."
        )
    if not 0 <= threshold <= 255:
        raise ValueError(f"Threshold must be between 0 and 255; got {threshold}.")
    if image.width < 1 or image.height < 1:
        raise ValueError("Image is empty; nothing to mask.")

    intensity = _luminance(image)
    return intensity > threshold if keep_above else intensity < threshold


def apply_threshold_mask(
    image: Image.Image,
    threshold: int,
    *,
    keep_above: bool = True,
) -> Image.Image:
    """Return ``image`` with pixels on the wrong side of ``threshold`` set to 0.

    Matches Set 2.3: ``masked = image * (image > T)`` or the inverse
    ``image * (image < T)``. Alpha is left untouched when present.

    Raises:
        ValueError: if no pixel falls on the chosen side of the threshold.
    """
    mask = build_threshold_mask(image, threshold, keep_above=keep_above)
    if not mask.any():
        side = "above" if keep_above else "under"
        raise ValueError(
            f"No pixels {side} the threshold ({threshold}).\n\n"
            "Try another threshold, or switch above/under."
        )

    arr = np.asarray(image)
    if image.mode == "L":
        return Image.fromarray((arr * mask).astype(np.uint8), mode="L")

    out = arr.copy()
    out[..., :3] = out[..., :3] * mask[..., None]
    return Image.fromarray(out, mode=image.mode)


class MaskingTool(ForensicsTool):
    tool_id = "masking"
    title = "Masking"
    category = "Intensity"
    description = (
        "Keep pixels above or under a threshold (Set 2.3); the rest become black."
    )

    ACCEPTED_MODES = ("L", "RGB", "RGBA")
    DEFAULT_THRESHOLD = 135

    def is_available(self, document: ImageDocument) -> bool:
        if document.current is None:
            return True
        return document.current.mode in self.ACCEPTED_MODES

    def unavailable_message(self, document: ImageDocument) -> str:
        current = document.current
        mode = current.mode if current is not None else "unknown"
        return (
            f"This image mode is not supported for masking (mode {mode!r}).\n\n"
            "Open an L, RGB or RGBA image, or convert to grayscale first."
        )

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        source = document.current
        assert source is not None
        assert source.mode in self.ACCEPTED_MODES, f"unsupported mode {source.mode!r}"

        choice = self._ask_mask_options(parent)
        if choice is None:
            return None
        threshold, keep_above = choice

        try:
            output = apply_threshold_mask(source, threshold, keep_above=keep_above)
        except ValueError as error:
            # Empty side of the threshold is a nudge, not a crashed tool.
            messagebox.showinfo(self.title, str(error), parent=parent)
            return None

        mask = build_threshold_mask(source, threshold, keep_above=keep_above)
        kept = int(mask.sum())
        total = mask.size
        side = "above threshold" if keep_above else "under threshold"
        return ToolResult(
            image=output,
            message=f"Kept pixels {side} ({threshold}).",
            details={
                "Operation": self.title,
                "Threshold": threshold,
                "Keep": side,
                "Kept pixels": f"{kept} / {total}",
                "Output mode": output.mode,
            },
        )

    def _ask_mask_options(
        self, parent: tk.Misc | None
    ) -> tuple[int, bool] | None:
        """One dialog: threshold + above/under. Returns ``None`` on cancel."""
        dialog = tk.Toplevel(parent)
        dialog.title(self.title)
        dialog.transient(parent)
        dialog.resizable(False, False)

        result: dict[str, tuple[int, bool] | None] = {"value": None}

        frame = ttk.Frame(dialog, padding=14)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="Threshold (0–255):").grid(row=0, column=0, sticky="w")
        threshold_var = tk.IntVar(value=self.DEFAULT_THRESHOLD)
        ttk.Spinbox(
            frame,
            from_=0,
            to=255,
            textvariable=threshold_var,
            width=8,
        ).grid(row=0, column=1, sticky="w", padx=(8, 0))

        ttk.Label(frame, text="Keep pixels:").grid(row=1, column=0, sticky="nw", pady=(12, 0))
        side_var = tk.StringVar(value="above")
        sides = ttk.Frame(frame)
        sides.grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(12, 0))
        ttk.Radiobutton(
            sides, text="Above the threshold", variable=side_var, value="above"
        ).pack(anchor="w")
        ttk.Radiobutton(
            sides, text="Under the threshold", variable=side_var, value="under"
        ).pack(anchor="w")

        ttk.Label(
            frame,
            text="Above keeps brighter pixels; under keeps darker ones.\n"
            "Pixels on the other side become black.",
            justify="left",
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(12, 0))

        def accept() -> None:
            try:
                threshold = int(threshold_var.get())
            except (tk.TclError, ValueError, TypeError):
                messagebox.showerror(
                    self.title, "Enter an integer threshold between 0 and 255.", parent=dialog
                )
                return
            if not 0 <= threshold <= 255:
                messagebox.showerror(
                    self.title, "Threshold must be between 0 and 255.", parent=dialog
                )
                return
            result["value"] = (threshold, side_var.get() == "above")
            dialog.destroy()

        def cancel() -> None:
            result["value"] = None
            dialog.destroy()

        buttons = ttk.Frame(frame)
        buttons.grid(row=3, column=0, columnspan=2, sticky="e", pady=(16, 0))
        ttk.Button(buttons, text="Cancel", command=cancel).pack(side="right")
        ttk.Button(buttons, text="Apply", command=accept).pack(side="right", padx=(0, 8))

        dialog.protocol("WM_DELETE_WINDOW", cancel)
        dialog.grab_set()
        dialog.wait_window()
        return result["value"]
