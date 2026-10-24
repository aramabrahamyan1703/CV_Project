"""Template matching with configurable angle and threshold (Set 4.2).

Slide: ``transform.rotate(template, a)``, configurable threshold, and a
thresholded view with a red marker (Set 3-style red superimposition on a
dimmed original).
"""

from __future__ import annotations

from dataclasses import dataclass
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
from PIL import Image, ImageDraw
from skimage import feature, transform

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .masking import OPEN_TYPES
from .param_dialog import IntField, ParamDialog, parse_int_fields


ACCEPTED_MODES = ("L", "RGB", "RGBA", "P")

MATCH_FIELDS = (
    IntField("Angle", 0, 180, "degrees to rotate the template", minimum=-180),
    IntField("Threshold", 80, 100, "% NCC score to accept a match", minimum=0),
)


@dataclass(frozen=True)
class Match:
    box: tuple[int, int, int, int]  # x, y, width, height (template top-left)
    score: float

    @property
    def centre(self) -> tuple[int, int]:
        x, y, w, h = self.box
        return x + w // 2, y + h // 2


def parse_match_params(angle_text: str, threshold_text: str) -> tuple[int, float]:
    """Angle in degrees and NCC threshold in ``[0, 1]`` from percent text."""
    angle, percent = parse_int_fields(MATCH_FIELDS, (angle_text, threshold_text))
    return angle, percent / 100.0


def rotate_template(template: Image.Image, angle: float) -> Image.Image:
    """Rotate ``template`` counterclockwise; expand canvas to fit (slide ``transform.rotate``)."""
    if template.mode not in ACCEPTED_MODES:
        raise ValueError(
            f"Template needs an L, RGB, RGBA or P image; got mode {template.mode!r}."
        )
    if angle == 0:
        return template.convert("L")
    gray = np.asarray(template.convert("L"), dtype=np.float64) / 255.0
    rotated = transform.rotate(gray, angle, resize=True, mode="constant", cval=0.0)
    pixels = (np.clip(rotated, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(pixels, mode="L")


def find_template_matches(
    scene: Image.Image,
    template: Image.Image,
    *,
    angle: float = 0.0,
    threshold: float = 0.8,
) -> tuple[Match, ...]:
    """Rotate the template, run NCC, and keep local peaks at or above ``threshold``."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Threshold must be between 0 and 1; got {threshold}.")
    if scene.mode not in ACCEPTED_MODES:
        raise ValueError(
            f"Scene needs an L, RGB, RGBA or P image; got mode {scene.mode!r}."
        )

    rotated = rotate_template(template, angle)
    scene_g = np.asarray(scene.convert("L"), dtype=np.float64) / 255.0
    template_g = np.asarray(rotated, dtype=np.float64) / 255.0
    th, tw = template_g.shape
    sh, sw = scene_g.shape
    if th > sh or tw > sw:
        raise ValueError(
            f"The rotated template ({tw}×{th}) is larger than the image ({sw}×{sh})."
        )
    if float(template_g.std()) < 1e-12:
        raise ValueError(
            "The template is constant, so normalized cross-correlation is undefined.\n\n"
            "Use a template with some texture or contrast."
        )

    # No pad: result[i, j] is the score when the template's top-left is at (i, j).
    response = feature.match_template(scene_g, template_g, pad_input=False)
    peaks = feature.peak_local_max(
        response,
        min_distance=max(1, min(th, tw) // 2),
        threshold_abs=threshold,
    )
    matches = [
        Match(box=(int(col), int(row), tw, th), score=float(response[row, col]))
        for row, col in peaks
    ]
    return tuple(sorted(matches, key=lambda m: m.score, reverse=True))


def superimpose_matches(scene: Image.Image, matches: tuple[Match, ...]) -> Image.Image:
    """Thresholded view from the slide: dimmed original + hollow red markers.

    Uses the Set 3 dimming idea (``rgb * 0.4``) and paints red circles at each
    match centre, like the Einstein eye marker on the Set 4.2 figure.
    """
    rgb = np.asarray(scene.convert("RGB"), dtype=np.float64) / 255.0
    dimmed = rgb * 0.4
    out = Image.fromarray((np.clip(dimmed, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(out)
    for match in matches:
        cx, cy = match.centre
        r = 4
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(255, 0, 0), width=2)
    return out


class TemplateMatchingTool(ForensicsTool):
    tool_id = "template_matching"
    title = "Template matching"
    category = "Matching"
    description = (
        "Match a rotated template and mark peaks above a threshold (Set 4.2)."
    )

    ACCEPTED_MODES = ACCEPTED_MODES

    def is_available(self, document: ImageDocument) -> bool:
        if document.current is None:
            return True
        return document.current.mode in self.ACCEPTED_MODES

    def unavailable_message(self, document: ImageDocument) -> str:
        current = document.current
        mode = current.mode if current is not None else "unknown"
        return (
            f"This image mode is not supported for template matching (mode {mode!r}).\n\n"
            "Open an L or RGB image."
        )

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        source = document.current
        assert source is not None

        template = self._ask_template(parent)
        if template is None:
            return None

        params = self._ask_params(parent)
        if params is None:
            return None
        angle, threshold = params

        try:
            matches = find_template_matches(
                source, template, angle=angle, threshold=threshold
            )
        except ValueError as error:
            messagebox.showinfo(self.title, str(error), parent=parent)
            return None

        if not matches:
            messagebox.showinfo(
                self.title,
                f"No matches at or above threshold {threshold:.0%}.\n\n"
                "Try a lower threshold or a different angle.",
                parent=parent,
            )
            return None

        best = matches[0].score
        return ToolResult(
            image=superimpose_matches(source, matches),
            message=f"Found {len(matches)} match(es); best score {best:.3f}.",
            details={
                "Operation": self.title,
                "Function": "skimage.feature.match_template",
                "Angle (deg)": angle,
                "Threshold": f"{threshold:.0%}",
                "Matches": len(matches),
                "Best score": f"{best:.4f}",
                "Template size": f"{template.width}×{template.height}",
            },
        )

    def _ask_template(self, parent: tk.Misc | None) -> Image.Image | None:
        path = filedialog.askopenfilename(
            parent=parent,
            title="Choose a template to match",
            filetypes=OPEN_TYPES,
        )
        if not path:
            return None
        try:
            with Image.open(path) as opened:
                return opened.copy()
        except OSError as error:
            messagebox.showinfo(self.title, f"Could not open template:\n{error}", parent=parent)
            return None

    def _ask_params(self, parent: tk.Misc | None) -> tuple[int, float] | None:
        dialog: ParamDialog[tuple[int, float]] = ParamDialog(
            parent,
            title=self.title,
            intro="Rotate the template (configurable angle) and keep NCC peaks at or "
            "above the threshold, then mark them on a dimmed original.",
            action="Match",
        )
        variables = dialog.int_fields(dialog.section("Match settings"), MATCH_FIELDS)

        def validate() -> tuple[tuple[int, float], str]:
            params = parse_match_params(*(v.get() for v in variables))
            angle, threshold = params
            return params, f"angle {angle}° · threshold {threshold:.0%}"

        return dialog.show(validate)
