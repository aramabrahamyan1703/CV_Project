"""Composite an overlay (e.g. coat) onto the working image using a black-background mask."""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult

OPEN_TYPES = [
    ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
    ("All files", "*.*"),
]


def build_foreground_mask(overlay: Image.Image, threshold: int = 10) -> np.ndarray:
    """True where the overlay is not a near-black background (Set 2 coat mask)."""
    if overlay.mode not in ("L", "RGB", "RGBA", "P"):
        raise ValueError(
            f"Overlay needs an L, RGB or RGBA image; got mode {overlay.mode!r}."
        )
    if not 0 <= threshold <= 255:
        raise ValueError(f"Threshold must be between 0 and 255; got {threshold}.")
    if overlay.width < 1 or overlay.height < 1:
        raise ValueError("Overlay image is empty.")

    rgb = np.asarray(overlay.convert("RGB"))
    return np.any(rgb > threshold, axis=-1)


def compose_with_mask(
    base: Image.Image,
    overlay: Image.Image,
    texture: Image.Image | None = None,
    *,
    threshold: int = 10,
) -> tuple[Image.Image, dict[str, object]]:
    """Paste ``overlay`` (or ``texture``) onto ``base`` where the overlay is not black.

    Matches the Set 2 coat / model / textured-coat demo: black in the overlay is
    transparent; non-black pixels replace the base (or are filled from a texture).
    Overlay and texture are resized to the base size when needed.
    """
    if base.mode not in ("L", "RGB", "RGBA", "P"):
        raise ValueError(
            f"Base image needs an L, RGB or RGBA mode; got mode {base.mode!r}."
        )

    base_rgb = base.convert("RGB")
    overlay_rgb = overlay.convert("RGB")
    resized_overlay = overlay_rgb.size != base_rgb.size
    if resized_overlay:
        # Nearest-neighbour keeps the hard black/subject cut (bilinear bleeds grey).
        overlay_rgb = overlay_rgb.resize(base_rgb.size, Image.Resampling.NEAREST)

    mask = build_foreground_mask(overlay_rgb, threshold)
    if not mask.any():
        raise ValueError(
            f"No mask pixels found (everything is at or under threshold {threshold}).\n\n"
            "Use an overlay with a dark background and a brighter subject, "
            "or lower the black threshold."
        )

    out = np.asarray(base_rgb).copy()
    resized_texture = False
    if texture is None:
        out[mask] = np.asarray(overlay_rgb)[mask]
        fill = "overlay colours"
    else:
        texture_rgb = texture.convert("RGB")
        resized_texture = texture_rgb.size != base_rgb.size
        if resized_texture:
            texture_rgb = texture_rgb.resize(base_rgb.size, Image.Resampling.BILINEAR)
        out[mask] = np.asarray(texture_rgb)[mask]
        fill = "texture"

    details: dict[str, object] = {
        "Mask pixels": f"{int(mask.sum())} / {mask.size}",
        "Black threshold": threshold,
        "Fill": fill,
        "Overlay resized": "yes" if resized_overlay else "no",
        "Texture resized": ("yes" if resized_texture else "no") if texture is not None else "n/a",
        "Output mode": "RGB",
    }
    return Image.fromarray(out, mode="RGB"), details


class MaskingTool(ForensicsTool):
    tool_id = "masking"
    title = "Masking"
    category = "Color"
    description = (
        "Composite an overlay (coat) onto the working image; "
        "optionally fill the mask with a texture."
    )

    ACCEPTED_MODES = ("L", "RGB", "RGBA", "P")
    DEFAULT_THRESHOLD = 10

    def is_available(self, document: ImageDocument) -> bool:
        if document.current is None:
            return True
        return document.current.mode in self.ACCEPTED_MODES

    def unavailable_message(self, document: ImageDocument) -> str:
        current = document.current
        mode = current.mode if current is not None else "unknown"
        return (
            f"This image mode is not supported for masking (mode {mode!r}).\n\n"
            "Open an L, RGB or RGBA image as the base (e.g. the model)."
        )

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        source = document.current
        assert source is not None
        assert source.mode in self.ACCEPTED_MODES, f"unsupported mode {source.mode!r}"

        paths = self._ask_images(parent)
        if paths is None:
            return None
        overlay_path, texture_path = paths

        try:
            overlay = Image.open(overlay_path)
            overlay.load()
            texture = None
            if texture_path is not None:
                texture = Image.open(texture_path)
                texture.load()
            output, info = compose_with_mask(
                source,
                overlay,
                texture,
                threshold=self.DEFAULT_THRESHOLD,
            )
        except (OSError, ValueError) as error:
            messagebox.showerror(self.title, str(error), parent=parent)
            return None

        details = {"Operation": self.title, "Overlay": overlay_path.name, **info}
        if texture_path is not None:
            details["Texture"] = texture_path.name
        fill = info["Fill"]
        return ToolResult(
            image=output,
            message=f"Composited overlay onto the image ({fill}).",
            details=details,
        )

    def _ask_images(
        self, parent: tk.Misc | None
    ) -> tuple[Path, Path | None] | None:
        """Ask for the overlay (required) and an optional texture."""
        overlay_name = filedialog.askopenfilename(
            parent=parent,
            title="Choose overlay / mask image (e.g. coat.png)",
            filetypes=OPEN_TYPES,
        )
        if not overlay_name:
            return None

        use_texture = messagebox.askyesno(
            self.title,
            "Fill the masked region with a texture image?\n\n"
            "Yes → pick a texture (textured coat).\n"
            "No → keep the overlay's own colours.",
            parent=parent,
        )
        texture_path: Path | None = None
        if use_texture:
            texture_name = filedialog.askopenfilename(
                parent=parent,
                title="Choose texture image",
                filetypes=OPEN_TYPES,
            )
            if not texture_name:
                return None
            texture_path = Path(texture_name)

        return Path(overlay_name), texture_path
