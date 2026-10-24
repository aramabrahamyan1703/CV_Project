"""Match ORB / SIFT descriptors between the working image and a second image (Set 5.3).

Slide: ``match_descriptors(desc1, desc2, cross_check=..., max_ratio=...,
metric='hamming')``, drawn as the two images side by side with a coloured line
per match. The metric follows the detector: Hamming for ORB's binary
descriptors, Euclidean for SIFT's numeric ones.
"""

from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image
from skimage import feature

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .descriptors import (
    ACCEPTED_MODES,
    DescriptorParams,
    descriptor_section,
    extract_descriptors,
    metric_for,
)
from .param_dialog import ImageChooser, IntField, ParamDialog, parse_int_fields


MATCH_FIELDS = (
    IntField("Max ratio ×100", 80, 100, "Lowe's ratio test; 100 = off"),
    IntField("Lines drawn", 50, 1000, "closest matches drawn; 0 = all", minimum=0),
)

# Rendered sheet width in pixels: small pairs are enlarged, big ones reduced.
MIN_SHEET_W = 1000
MAX_SHEET_W = 2400


@dataclass(frozen=True)
class MatchParams:
    detector: DescriptorParams
    cross_check: bool = True
    max_ratio: float = 0.8
    lines: int = 50  # 0 draws every match


@dataclass(frozen=True)
class MatchResult:
    keypoints1: np.ndarray  # (N, 2) as (row, col)
    keypoints2: np.ndarray
    matches: np.ndarray  # (M, 2) indices into keypoints1 / keypoints2
    distances: np.ndarray  # (M,) descriptor distance of each match
    metric: str


def parse_matching_params(
    detector: DescriptorParams, cross_check: bool, ratio_text: str, lines_text: str
) -> MatchParams:
    """Validate the matching fields; the ratio is entered as a percentage."""
    ratio, lines = parse_int_fields(MATCH_FIELDS, (ratio_text, lines_text))
    return MatchParams(detector, cross_check, ratio / 100, lines)


def matching_summary(params: MatchParams) -> str:
    """Footer readout, e.g. ``"hamming · cross-check · ratio 0.80"``."""
    cross = "cross-check" if params.cross_check else "no cross-check"
    ratio = "no ratio test" if params.max_ratio >= 1 else f"ratio {params.max_ratio:.2f}"
    return f"{metric_for(params.detector.method)} · {cross} · {ratio}"


def extract(image: Image.Image, detector: DescriptorParams) -> tuple[np.ndarray, np.ndarray]:
    """Keypoints and descriptors for ``image`` with the chosen detector."""
    return extract_descriptors(
        image,
        method=detector.method,
        n_keypoints=detector.n_keypoints,
        n_octaves=detector.n_octaves,
        n_scales=detector.n_scales,
        sigma_min=detector.sigma_min,
    )


def match_features(
    keypoints1: np.ndarray,
    descriptors1: np.ndarray,
    keypoints2: np.ndarray,
    descriptors2: np.ndarray,
    params: MatchParams,
) -> MatchResult:
    """``skimage.feature.match_descriptors`` plus each match's distance."""
    metric = metric_for(params.detector.method)
    matches = feature.match_descriptors(
        descriptors1,
        descriptors2,
        cross_check=params.cross_check,
        max_ratio=params.max_ratio,
        metric=metric,
    )
    first, second = descriptors1[matches[:, 0]], descriptors2[matches[:, 1]]
    if metric == "hamming":
        distances = np.mean(first != second, axis=1)  # share of differing bits
    else:
        distances = np.linalg.norm(first.astype(np.float64) - second.astype(np.float64), axis=1)
    return MatchResult(keypoints1, keypoints2, matches, distances, metric)


def match_images(image1: Image.Image, image2: Image.Image, params: MatchParams) -> MatchResult:
    """Extract descriptors from both images and match them."""
    keypoints1, descriptors1 = extract(image1, params.detector)
    keypoints2, descriptors2 = extract(image2, params.detector)
    return match_features(keypoints1, descriptors1, keypoints2, descriptors2, params)


def closest(matches: np.ndarray, distances: np.ndarray, count: int) -> np.ndarray:
    """The ``count`` matches with the smallest distance (all of them for 0)."""
    if count == 0 or count >= len(matches):
        return matches
    return matches[np.argsort(distances, kind="stable")[:count]]


def render_matches(
    image1: Image.Image,
    image2: Image.Image,
    keypoints1: np.ndarray,
    keypoints2: np.ndarray,
    matches: np.ndarray,
    *,
    title: str,
    all_keypoints: bool = False,
    keypoints_color: str = "k",
) -> Image.Image:
    """The two images side by side in grayscale, a coloured line per match.

    By default only the matched keypoints are circled (slide 5.3); with
    ``all_keypoints`` every keypoint is (slide 5.4).
    """
    gray1 = np.asarray(image1.convert("L"))
    gray2 = np.asarray(image2.convert("L"))
    if not all_keypoints:
        # Circle only the line ends: keep the matched keypoints and renumber.
        keypoints1, keypoints2 = keypoints1[matches[:, 0]], keypoints2[matches[:, 1]]
        matches = np.column_stack([np.arange(len(matches))] * 2)

    width = gray1.shape[1] + gray2.shape[1]
    height = max(gray1.shape[0], gray2.shape[0])
    scale = min(max(width, MIN_SHEET_W), MAX_SHEET_W) / width
    figure = Figure(
        figsize=(width * scale / 100, height * scale / 100 + 0.5), dpi=100, layout="constrained"
    )
    axis = figure.add_subplot(1, 1, 1)
    # Evenly spread hues, fixed so the same input always draws the same picture.
    colours = [plt.get_cmap("hsv")((k * 0.618) % 1.0) for k in range(len(matches))]
    feature.plot_matched_features(
        gray1,
        gray2,
        keypoints0=keypoints1,
        keypoints1=keypoints2,
        matches=matches,
        ax=axis,
        keypoints_color=keypoints_color,
        matches_color=colours if len(matches) else None,
    )
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(title)

    FigureCanvasAgg(figure)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0.15)
    plt.close(figure)
    buffer.seek(0)
    with Image.open(buffer) as rendered:
        return rendered.convert("RGB")


@dataclass(frozen=True)
class MatchingRequest:
    second_path: Path
    second: Image.Image
    params: MatchParams


class FeatureMatchingTool(ForensicsTool):
    tool_id = "feature_matching"
    title = "Feature matching"
    category = "Features"
    description = "Match ORB or SIFT keypoints with a second image (Set 5.3)."

    ACCEPTED_MODES = ACCEPTED_MODES

    def __init__(self) -> None:
        self._last_second: Path | None = None  # pre-fills the dialog on the next run

    def is_available(self, document: ImageDocument) -> bool:
        """Refuse unsupported modes and the tool's own side-by-side sheet."""
        if document.current is None:
            return True
        if document.current.mode not in self.ACCEPTED_MODES:
            return False
        return document.last_tool_id != self.tool_id

    def unavailable_message(self, document: ImageDocument) -> str:
        current = document.current
        if current is not None and current.mode not in self.ACCEPTED_MODES:
            return (
                f"This image mode is not supported for matching (mode {current.mode!r}).\n\n"
                "Open an L or RGB image."
            )
        return (
            "The matches are already shown.\n\n"
            "Undo or reset the image to match the original again."
        )

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        source = document.current
        assert source is not None
        assert source.mode in self.ACCEPTED_MODES, f"unsupported mode {source.mode!r}"

        request = self._ask_params(parent, source)
        if request is None:
            return None
        params = request.params

        try:
            result = match_images(source, request.second, params)
        except ValueError as error:
            messagebox.showinfo(self.title, str(error), parent=parent)
            return None

        count = len(result.matches)
        drawn = closest(result.matches, result.distances, params.lines)
        shown = "all drawn" if len(drawn) == count else f"{len(drawn)} closest drawn"
        sheet = render_matches(
            source,
            request.second,
            result.keypoints1,
            result.keypoints2,
            drawn,
            title=f"{count} matches ({shown})" if count else "No matches",
        )

        detector = params.detector
        details: dict[str, object] = {
            "Operation": self.title,
            "Function": "skimage.feature.match_descriptors",
            "Second image": (
                f"{request.second_path.name} ({request.second.width}×{request.second.height})"
            ),
            "Detector": detector.method.upper(),
        }
        if detector.method == "orb":
            details["n_keypoints"] = detector.n_keypoints
        else:
            details["n_octaves"] = detector.n_octaves
            details["n_scales"] = detector.n_scales
            details["sigma_min"] = detector.sigma_min
        details.update(
            {
                "Metric": result.metric,
                "Cross-check": "yes" if params.cross_check else "no",
                "Max ratio": f"{params.max_ratio:.2f}",
                "Keypoints (this / second)": (
                    f"{len(result.keypoints1)} / {len(result.keypoints2)}"
                ),
                "Matches": count,
                "Lines drawn": len(drawn),
            }
        )
        if count:
            details["Mean match distance"] = f"{float(result.distances.mean()):.4f}"

        return ToolResult(
            image=sheet,
            message=(
                f"Found {count} {detector.method.upper()} matches with "
                f"{request.second_path.name}."
            ),
            details=details,
        )

    def _ask_params(self, parent: tk.Misc | None, source: Image.Image) -> MatchingRequest | None:
        """One dialog: second image, detector and matching settings; ``None`` on cancel."""
        dialog: ParamDialog[MatchingRequest] = ParamDialog(
            parent,
            title=self.title,
            intro="Find keypoints in both images and pair each with its most "
            "similar descriptor in the other image.",
            action="Match",
        )
        chooser = ImageChooser(
            dialog,
            dialog.section("Second image"),
            prompt="Choose a second image to match against.",
            empty_hint="The same scene or object,\nseen differently.",
            initial=self._last_second,
        )
        read_detector = descriptor_section(dialog)
        frame = dialog.section("Matching")
        cross_check = dialog.checkbox(
            frame, "Cross-check", default=True, hint="keep only mutual best matches"
        )
        ratio_var, lines_var = dialog.int_fields(frame, MATCH_FIELDS)

        def validate() -> tuple[MatchingRequest, str]:
            path, second = chooser.require()
            if second.mode not in self.ACCEPTED_MODES:
                second = second.convert("RGB")
            detector, detector_readout = read_detector()
            params = parse_matching_params(
                detector, cross_check.get(), ratio_var.get(), lines_var.get()
            )
            readout = f"{detector_readout}\n{matching_summary(params)}"
            return MatchingRequest(path, second, params), readout

        request = dialog.show(validate)
        if request is not None:
            self._last_second = request.second_path
        return request
