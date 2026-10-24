"""ORB / SIFT descriptors with keypoint visualization (Set 5.2).

Slide: ORB ``n_keypoints`` and SIFT ``n_octaves`` / ``n_scales`` / ``sigma_min``
must be configurable; keypoints drawn as hollow red circles.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
from PIL import Image, ImageDraw
from skimage import feature

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .param_dialog import IntField, ParamDialog, parse_int_fields


ACCEPTED_MODES = ("L", "RGB", "RGBA", "P")
METHODS = ("orb", "sift")

ORB_FIELDS = (
    IntField("Keypoints", 200, 500, "ORB n_keypoints", minimum=1),
)
SIFT_FIELDS = (
    IntField("Octaves", 8, 12, "SIFT n_octaves", minimum=1),
    IntField("Scales", 3, 8, "SIFT n_scales per octave", minimum=1),
    IntField("Sigma min ×10", 16, 50, "SIFT sigma_min = value / 10", minimum=5),
)


@dataclass(frozen=True)
class DescriptorParams:
    method: str
    n_keypoints: int = 200
    n_octaves: int = 8
    n_scales: int = 3
    sigma_min: float = 1.6


def parse_descriptor_params(
    method_text: str,
    *field_texts: str,
) -> DescriptorParams:
    """Parse ORB (keypoints) or SIFT (octaves, scales, sigma×10) dialog fields."""
    method = method_text.strip().lower()
    if method not in METHODS:
        raise ValueError(f"Method must be one of {', '.join(METHODS)}.")
    if method == "orb":
        if len(field_texts) != 1:
            raise ValueError("ORB needs one Keypoints value.")
        (n_keypoints,) = parse_int_fields(ORB_FIELDS, field_texts)
        return DescriptorParams(method="orb", n_keypoints=n_keypoints)
    if len(field_texts) != 3:
        raise ValueError("SIFT needs Octaves, Scales and Sigma min.")
    n_octaves, n_scales, sigma_x10 = parse_int_fields(SIFT_FIELDS, field_texts)
    return DescriptorParams(
        method="sift",
        n_octaves=n_octaves,
        n_scales=n_scales,
        sigma_min=sigma_x10 / 10.0,
    )


def extract_descriptors(
    image: Image.Image,
    *,
    method: str = "orb",
    n_keypoints: int = 200,
    n_octaves: int = 8,
    n_scales: int = 3,
    sigma_min: float = 1.6,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(keypoints[N,2] as (row, col), descriptors)`` from ORB or SIFT."""
    method = method.strip().lower()
    if method not in METHODS:
        raise ValueError(f"Method must be one of {', '.join(METHODS)}.")
    if image.mode not in ACCEPTED_MODES:
        raise ValueError(
            f"Descriptors need an L, RGB, RGBA or P image; got mode {image.mode!r}."
        )

    gray = np.asarray(image.convert("L"), dtype=np.float64)
    if method == "orb":
        if n_keypoints < 1:
            raise ValueError(f"Keypoints must be at least 1; got {n_keypoints}.")
        detector = feature.ORB(n_keypoints=n_keypoints)
    else:
        if n_octaves < 1 or n_scales < 1:
            raise ValueError("SIFT octaves and scales must be at least 1.")
        if sigma_min <= 0:
            raise ValueError(f"SIFT sigma_min must be positive; got {sigma_min}.")
        detector = feature.SIFT(
            n_octaves=n_octaves, n_scales=n_scales, sigma_min=sigma_min
        )

    try:
        detector.detect_and_extract(gray)
    except RuntimeError:
        # skimage raises when it finds nothing at all (flat or low-contrast images).
        raise ValueError(
            f"No keypoints found with {method.upper()}.\n\n"
            "Try a more textured image, or a different detector."
        ) from None
    keypoints = np.asarray(detector.keypoints)
    descriptors = np.asarray(detector.descriptors)
    if keypoints.size == 0:
        raise ValueError(
            f"No keypoints found with {method.upper()}.\n\n"
            "Try a more textured image, or a different detector."
        )
    return keypoints, descriptors


def metric_for(method: str) -> str:
    """``match_descriptors`` metric: ORB descriptors are bits, SIFT ones are numbers."""
    return "hamming" if method.strip().lower() == "orb" else "euclidean"


def descriptor_section(dialog: ParamDialog) -> Callable[[], tuple[DescriptorParams, str]]:
    """Add the ORB/SIFT choice and both field groups to ``dialog``.

    Returns a reader for use inside the dialog's ``validate``: it parses the
    active method's fields and gives ``(params, readout)``, or raises
    ``ValueError`` naming the bad field.
    """
    method_frame = dialog.section("Detector")
    method_var = tk.StringVar(value="orb")
    ttk.Radiobutton(method_frame, text="ORB", variable=method_var, value="orb").grid(
        row=0, column=0, sticky="w"
    )
    ttk.Radiobutton(method_frame, text="SIFT", variable=method_var, value="sift").grid(
        row=0, column=1, sticky="w", padx=(16, 0)
    )
    # Both groups stay visible; only the chosen method's fields are read.
    orb_vars = dialog.int_fields(dialog.section("ORB (n_keypoints)"), ORB_FIELDS)
    sift_vars = dialog.int_fields(
        dialog.section("SIFT (n_octaves, n_scales, sigma_min)"), SIFT_FIELDS
    )
    method_var.trace_add("write", lambda *_args: dialog.refresh())

    def read() -> tuple[DescriptorParams, str]:
        if method_var.get() == "orb":
            params = parse_descriptor_params("orb", orb_vars[0].get())
            return params, f"ORB · n_keypoints={params.n_keypoints}"
        params = parse_descriptor_params("sift", *(v.get() for v in sift_vars))
        return (
            params,
            f"SIFT · octaves={params.n_octaves} · scales={params.n_scales} · "
            f"sigma_min={params.sigma_min:g}",
        )

    return read


def render_keypoints(image: Image.Image, keypoints: np.ndarray) -> Image.Image:
    """Draw hollow red circles at ``(row, col)`` keypoints (slide figure style)."""
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    for row, col in keypoints:
        x, y = int(round(col)), int(round(row))
        r = 4
        draw.ellipse((x - r, y - r, x + r, y + r), outline=(255, 0, 0), width=2)
    return out


class DescriptorTool(ForensicsTool):
    tool_id = "descriptors"
    title = "Descriptors"
    category = "Features"
    description = "Extract ORB or SIFT descriptors and visualize keypoints (Set 5.2)."

    ACCEPTED_MODES = ACCEPTED_MODES

    def is_available(self, document: ImageDocument) -> bool:
        if document.current is None:
            return True
        return document.current.mode in self.ACCEPTED_MODES

    def unavailable_message(self, document: ImageDocument) -> str:
        current = document.current
        mode = current.mode if current is not None else "unknown"
        return (
            f"This image mode is not supported for descriptors (mode {mode!r}).\n\n"
            "Open an L or RGB image."
        )

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        source = document.current
        assert source is not None

        params = self._ask_params(parent, source)
        if params is None:
            return None

        try:
            keypoints, descriptors = extract_descriptors(
                source,
                method=params.method,
                n_keypoints=params.n_keypoints,
                n_octaves=params.n_octaves,
                n_scales=params.n_scales,
                sigma_min=params.sigma_min,
            )
        except ValueError as error:
            messagebox.showinfo(self.title, str(error), parent=parent)
            return None

        details: dict[str, object] = {
            "Operation": self.title,
            "Function": f"skimage.feature.{params.method.upper()}",
            "Method": params.method,
            "Keypoints": len(keypoints),
            "Descriptor length": int(descriptors.shape[1]) if descriptors.ndim == 2 else 0,
        }
        if params.method == "orb":
            details["n_keypoints"] = params.n_keypoints
        else:
            details["n_octaves"] = params.n_octaves
            details["n_scales"] = params.n_scales
            details["sigma_min"] = params.sigma_min

        return ToolResult(
            image=render_keypoints(source, keypoints),
            message=f"Extracted {len(keypoints)} {params.method.upper()} keypoint(s).",
            details=details,
        )

    def _ask_params(
        self, parent: tk.Misc | None, _source: Image.Image
    ) -> DescriptorParams | None:
        dialog: ParamDialog[DescriptorParams] = ParamDialog(
            parent,
            title=self.title,
            intro="Detect keypoints and extract ORB or SIFT descriptors. "
            "Parameters highlighted on the slide are configurable.",
            action="Extract",
        )
        read_detector = descriptor_section(dialog)
        return dialog.show(read_detector)
