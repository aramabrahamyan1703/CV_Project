"""Corner detection with FAST / Harris and a peak threshold (Set 5.1).

Slide: show the raw response (``imshow(fast)``) and peaks as hollow red circles
via ``corner_peaks`` with a modifiable threshold.
"""

from __future__ import annotations

import math
import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image
from skimage import feature

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .param_dialog import IntField, ParamDialog, parse_int_fields


ACCEPTED_MODES = ("L", "RGB", "RGBA", "P")
METHODS = ("harris", "fast")

CORNER_FIELDS = (
    IntField("Peak threshold", 10, 100, "% of the strongest corner response", minimum=1),
    IntField("Min. distance", 5, 50, "pixels between accepted peaks", minimum=1),
)


def parse_corner_params(
    method_text: str, threshold_text: str, distance_text: str
) -> tuple[str, float, int]:
    """``(method, threshold_rel, min_distance)`` for ``corner_peaks``."""
    method = method_text.strip().lower()
    if method not in METHODS:
        raise ValueError(f"Method must be one of {', '.join(METHODS)}.")
    percent, min_distance = parse_int_fields(
        CORNER_FIELDS, (threshold_text, distance_text)
    )
    return method, percent / 100.0, min_distance


def detect_corners(
    image: Image.Image,
    *,
    method: str = "harris",
    threshold_rel: float = 0.1,
    min_distance: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(corner_coords[N,2], response_map)`` from FAST or Harris + peaks."""
    method = method.strip().lower()
    if method not in METHODS:
        raise ValueError(f"Method must be one of {', '.join(METHODS)}.")
    if image.mode not in ACCEPTED_MODES:
        raise ValueError(
            f"Corner detection needs an L, RGB, RGBA or P image; got mode {image.mode!r}."
        )
    if not 0.0 < threshold_rel <= 1.0:
        raise ValueError(f"Peak threshold must be in (0, 1]; got {threshold_rel}.")
    if min_distance < 1:
        raise ValueError(f"Min. distance must be at least 1; got {min_distance}.")

    gray = np.asarray(image.convert("L"), dtype=np.float64)
    if method == "harris":
        response = feature.corner_harris(gray)
    else:
        response = feature.corner_fast(gray)

    coords = feature.corner_peaks(
        response,
        min_distance=min_distance,
        threshold_rel=threshold_rel,
    )
    return coords, response


def render_corners(
    image: Image.Image, corners: np.ndarray, response: np.ndarray, *, method: str
) -> Image.Image:
    """Two-panel sheet like the slide: response map | grayscale + hollow red peaks."""
    width, height = image.size
    panel_in, title_in = 4.5, 0.8
    fig_w = 2 * panel_in
    fig_h = min(panel_in * height / max(width, 1) + title_in, 2 * panel_in)
    dpi = min(300, max(100, width / (panel_in * 0.85)))
    dpi = min(dpi, math.sqrt(12_000_000 / (fig_w * fig_h)))

    figure = Figure(figsize=(fig_w, fig_h), dpi=int(dpi), layout="constrained")
    axes = figure.subplots(1, 2)

    axes[0].imshow(response)
    axes[0].set_title(f"{method.upper()} response")
    axes[0].set_axis_off()

    gray = np.asarray(image.convert("L"))
    axes[1].imshow(gray, cmap="gray", vmin=0, vmax=255)
    if len(corners):
        # Slide: scatter(..., facecolors='none', edgecolors='r')
        axes[1].scatter(
            corners[:, 1],
            corners[:, 0],
            s=36,
            facecolors="none",
            edgecolors="r",
            linewidths=1.2,
        )
    axes[1].set_title("Corner peaks")
    axes[1].set_axis_off()

    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    out = Image.fromarray(np.asarray(canvas.buffer_rgba())).convert("RGB")
    plt.close(figure)
    return out


class CornerDetectionTool(ForensicsTool):
    tool_id = "corners"
    title = "Corner detection"
    category = "Features"
    description = (
        "Detect corners with skimage FAST or Harris; peak threshold is configurable (Set 5.1)."
    )

    ACCEPTED_MODES = ACCEPTED_MODES

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
                f"This image mode is not supported for corner detection (mode {current.mode!r}).\n\n"
                "Open an L or RGB image."
            )
        return (
            "The corner visualization is already shown.\n\n"
            "Undo or reset the image to detect corners again."
        )

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        source = document.current
        assert source is not None

        params = self._ask_params(parent, source)
        if params is None:
            return None
        method, threshold_rel, min_distance = params

        try:
            corners, response = detect_corners(
                source,
                method=method,
                threshold_rel=threshold_rel,
                min_distance=min_distance,
            )
        except ValueError as error:
            messagebox.showinfo(self.title, str(error), parent=parent)
            return None

        if len(corners) == 0:
            messagebox.showinfo(
                self.title,
                f"No corners found with {method} at peak threshold {threshold_rel:.0%}.\n\n"
                "Try a lower peak threshold.",
                parent=parent,
            )
            return None

        function = (
            "skimage.feature.corner_harris + corner_peaks"
            if method == "harris"
            else "skimage.feature.corner_fast + corner_peaks"
        )
        return ToolResult(
            image=render_corners(source, corners, response, method=method),
            message=f"Detected {len(corners)} corner(s) with {method}.",
            details={
                "Operation": self.title,
                "Function": function,
                "Method": method,
                "Peak threshold": f"{threshold_rel:.0%}",
                "Min. distance": min_distance,
                "Corners": len(corners),
            },
        )

    def _ask_params(
        self, parent: tk.Misc | None, _source: Image.Image
    ) -> tuple[str, float, int] | None:
        dialog: ParamDialog[tuple[str, float, int]] = ParamDialog(
            parent,
            title=self.title,
            intro="Find corners with FAST or Harris. Lower the peak threshold to keep more points.",
            action="Detect",
        )
        method_frame = dialog.section("Detector")
        method_var = tk.StringVar(value="harris")
        ttk.Radiobutton(method_frame, text="Harris", variable=method_var, value="harris").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Radiobutton(method_frame, text="FAST", variable=method_var, value="fast").grid(
            row=0, column=1, sticky="w", padx=(16, 0)
        )
        method_var.trace_add("write", lambda *_args: dialog.refresh())

        variables = dialog.int_fields(dialog.section("Peaks"), CORNER_FIELDS)

        def validate() -> tuple[tuple[str, float, int], str]:
            params = parse_corner_params(method_var.get(), *(v.get() for v in variables))
            method, threshold_rel, min_distance = params
            return params, f"{method} · peak ≥ {threshold_rel:.0%} · min dist {min_distance}"

        return dialog.show(validate)
