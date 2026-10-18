"""Find a person in the working image by sliding-window HOG template matching (Set 4.4).

The slide's method compares a template-sized window with the template at one
size and always reports the best window. Two additions make it usable on real
scenes:

- **Scales.** The scene is searched at several sizes (an image pyramid, ×1.25
  apart), so the person need not be exactly as many pixels tall as the template.
- **An optional threshold.** Each window's cosine similarity to the template
  (the dot product of the unit-length descriptors, 0..1 for HOG) is reported.
  With no minimum set the best window is always marked, as on the slide; with
  one, a best window below it is reported as "no person found". Similarity to
  a single template does not separate people from background reliably, so
  any minimum has to be chosen for the images at hand.
"""

from __future__ import annotations

from dataclasses import dataclass
import io
import math
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PIL import Image, ImageTk
from skimage import exposure, feature

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .masking import OPEN_TYPES
from .param_dialog import IntField, ParamDialog, parse_int_fields


ACCEPTED_MODES = ("L", "RGB", "RGBA", "P")

# HOG time grows with the pixels it reads: windows × template area. About
# 7.5 million pixels take a second, so this keeps one run under ~20 s.
MAX_WINDOW_PIXELS = 150_000_000
PIXELS_PER_SECOND = 7_500_000

# Neighbouring pyramid levels differ by about this factor.
SCALE_FACTOR = 1.25

# By default the search looks for people at least this share of the image's height,
# and never smaller than this % of the template (the scene is enlarged at most 2×).
MIN_PERSON_FRACTION = 0.25
MIN_SMALLEST_PERSON = 50

HOG_FIELDS = (
    IntField("Orientations", 16, 36, "gradient direction bins per cell"),
    IntField("Pixels per cell", 8, 64, "square cell side, in pixels"),
    IntField("Cells per block", 2, 8, "square block side, in cells"),
)
# Person sizes are percentages of the template's size. Choosing a template
# replaces the two size defaults with default_size_range() for that template
# and image. Min. similarity 0 means no threshold:
# a different person than the template's scored 0.58 in a real scene, below
# the 0.59 best window of a scene with no person, so no default is safe.
SEARCH_FIELDS = (
    IntField("Step size", 8, 256, "window shift, in pixels"),
    IntField("Smallest person", 100, 1000, "% of the template's size"),
    IntField("Largest person", 300, 1000, "% of the template's size"),
    IntField("Min. similarity", 0, 99, "%; 0 always marks the best window", minimum=0),
)


@dataclass(frozen=True)
class SearchParams:
    orientations: int
    pixels_per_cell: int
    cells_per_block: int
    step: int
    scales: tuple[float, ...]
    min_similarity: float


@dataclass(frozen=True)
class ScaleLevel:
    scale: float  # person size relative to the template
    size: tuple[int, int]  # the scene resized by 1 / scale
    windows: int


@dataclass(frozen=True)
class Detection:
    """One method's best window, in the original scene's pixels."""

    box: tuple[int, int, int, int]  # x, y, width, height
    scale: float
    score: float  # Euclidean distance or dot product, per the method
    similarity: float  # cosine similarity to the template
    found: bool  # similarity reached the threshold


@dataclass(frozen=True)
class HOGMatch:
    euclidean: Detection
    dot: Detection
    levels: tuple[ScaleLevel, ...]
    windows: int
    descriptor_length: int


def parse_search_params(*texts: str) -> SearchParams:
    """Validate the seven numeric dialog fields, HOG fields first."""
    values = parse_int_fields(HOG_FIELDS + SEARCH_FIELDS, texts)
    orientations, ppc, cpb, step, smallest, largest, similarity = values
    if smallest > largest:
        raise ValueError("Smallest person must not be larger than largest person.")
    return SearchParams(
        orientations, ppc, cpb, step, pyramid_scales(smallest, largest), similarity / 100
    )


def default_size_range(
    scene_size: tuple[int, int],
    template_size: tuple[int, int],
    *,
    step: int = SEARCH_FIELDS[0].default,
) -> tuple[int, int]:
    """Smallest and largest person, in % of the template, worth searching by default.

    The largest is the biggest person whose window still fits in the image.
    The smallest is a person MIN_PERSON_FRACTION of the image's height, at
    least MIN_SMALLEST_PERSON % of the template (or the largest, if smaller), raised
    a pyramid step at a time while the search would exceed the time limit:
    people smaller than the template are the slow case, as finding them means
    enlarging the scene.
    """
    (scene_w, scene_h), (tmpl_w, tmpl_h) = scene_size, template_size
    largest_field = SEARCH_FIELDS[2]
    fits = math.floor(min(scene_w / tmpl_w, scene_h / tmpl_h) * 100)
    largest = min(max(fits, largest_field.minimum), largest_field.limit)
    smallest = round(MIN_PERSON_FRACTION * scene_h / tmpl_h * 100)
    smallest = min(max(smallest, MIN_SMALLEST_PERSON), largest)
    limit = MAX_WINDOW_PIXELS / PIXELS_PER_SECOND
    while smallest < largest:
        levels = _levels(scene_size, template_size, pyramid_scales(smallest, largest), step)
        if _seconds(levels, template_size) <= limit:
            break
        smallest = min(math.ceil(smallest * SCALE_FACTOR), largest)
    return smallest, largest


def pyramid_scales(smallest_pct: int, largest_pct: int) -> tuple[float, ...]:
    """Scales from smallest to largest, evenly spaced in ratio, about ×1.25 apart."""
    low, high = smallest_pct / 100, largest_pct / 100
    count = max(1, round(math.log(high / low) / math.log(SCALE_FACTOR)) + 1)
    if count == 1:
        return ((low * high) ** 0.5,)
    return tuple(float(s) for s in np.geomspace(low, high, count))


def plan_search(
    scene_size: tuple[int, int],
    template_size: tuple[int, int],
    *,
    pixels_per_cell: int,
    cells_per_block: int,
    step: int,
    scales: tuple[float, ...] = (1.0,),
) -> tuple[ScaleLevel, ...]:
    """The pyramid levels the template fits in; raises ``ValueError`` if none can run.

    Sizes are PIL ``(width, height)``. At scale ``s`` the scene is resized by
    ``1 / s``, so a person ``s`` times the template's size becomes template-sized.
    Unlike the slide's ``range(0, size - window, step)``, the last position that
    still fits is included, so a template as large as the scene gives one window.
    """
    tmpl_w, tmpl_h = template_size
    block_px = pixels_per_cell * cells_per_block
    if tmpl_w < block_px or tmpl_h < block_px:
        raise ValueError(
            f"The template ({tmpl_w}×{tmpl_h}) is smaller than one HOG block "
            f"({block_px}×{block_px} px). Use smaller cells or fewer cells per block."
        )
    levels = _levels(scene_size, template_size, scales, step)
    if not levels:
        scene_w, scene_h = scene_size
        fits = min(scene_w / tmpl_w, scene_h / tmpl_h)
        raise ValueError(
            f"The template ({tmpl_w}×{tmpl_h}) is too large for this image "
            f"({scene_w}×{scene_h}) at every size searched: people here can be at most "
            f"{fits:.0%} of the template's size. Lower the smallest person size."
        )
    seconds = _seconds(levels, template_size)
    if seconds > MAX_WINDOW_PIXELS / PIXELS_PER_SECOND:
        advice = "Narrow the size range or use a smaller template."
        for faster in range(step + 1, 257):
            quicker = _seconds(_levels(scene_size, template_size, scales, faster), template_size)
            if quicker <= MAX_WINDOW_PIXELS / PIXELS_PER_SECOND:
                advice = (
                    f"Step size {faster} would take about {max(quicker, 1):.0f} s, "
                    "or narrow the size range."
                )
                break
        raise ValueError(f"This search would take about {seconds:.0f} s. {advice}")
    return levels


def _levels(
    scene_size: tuple[int, int],
    template_size: tuple[int, int],
    scales: tuple[float, ...],
    step: int,
) -> tuple[ScaleLevel, ...]:
    """Every pyramid level the template fits in, with its window count."""
    (scene_w, scene_h), (tmpl_w, tmpl_h) = scene_size, template_size
    levels = []
    for scale in scales:
        width, height = round(scene_w / scale), round(scene_h / scale)
        if width < tmpl_w or height < tmpl_h:
            continue
        windows = len(range(0, height - tmpl_h + 1, step)) * len(
            range(0, width - tmpl_w + 1, step)
        )
        levels.append(ScaleLevel(scale, (width, height), windows))
    return tuple(levels)


def _seconds(levels: tuple[ScaleLevel, ...], template_size: tuple[int, int]) -> float:
    """Rough HOG time for a search: windows × template area at the measured rate."""
    windows = sum(level.windows for level in levels)
    return windows * template_size[0] * template_size[1] / PIXELS_PER_SECOND


def search_summary(
    levels: tuple[ScaleLevel, ...], scales: int, template_size: tuple[int, int]
) -> str:
    """Footer readout, e.g. ``"7 scales · 12,345 windows · about 9 s"``."""
    windows = sum(level.windows for level in levels)
    seconds = _seconds(levels, template_size)
    fitted = f"{len(levels)} scale" + ("" if len(levels) == 1 else "s")
    if len(levels) < scales:
        fitted = f"{len(levels)} of {scales} scales"
    timing = "under 1 s" if seconds < 1 else f"about {seconds:.0f} s"
    return f"{fitted} · {windows:,} windows · {timing}"


def _hog_pixels(image: Image.Image, grayscale: bool) -> tuple[np.ndarray, int | None]:
    """Pixels and ``channel_axis`` for ``feature.hog``; scene and template must agree."""
    if grayscale:
        return np.asarray(image.convert("L")), None
    return np.asarray(image.convert("RGB")), -1


def match_hog_template(
    scene: Image.Image,
    template: Image.Image,
    *,
    orientations: int,
    pixels_per_cell: int,
    cells_per_block: int,
    step: int,
    scales: tuple[float, ...] = (1.0,),
    min_similarity: float = 0.0,
) -> HOGMatch:
    """Slide a template-sized window over each pyramid level and compare HOGs.

    As on the slide, one best window is the smallest Euclidean distance to the
    template's descriptor and the other the largest dot product. A grayscale
    scene compares grayscale HOGs; otherwise both are RGB.
    """
    if scene.mode not in ACCEPTED_MODES:
        raise ValueError(
            f"Person detection needs an L, RGB, RGBA or P image; got mode {scene.mode!r}."
        )
    levels = plan_search(
        scene.size,
        template.size,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        step=step,
        scales=scales,
    )
    grayscale = scene.mode == "L"
    tmpl_px, channel_axis = _hog_pixels(template, grayscale)
    hog_options = dict(
        orientations=orientations,
        pixels_per_cell=(pixels_per_cell, pixels_per_cell),
        cells_per_block=(cells_per_block, cells_per_block),
        channel_axis=channel_axis,
    )
    tmpl_fd = feature.hog(tmpl_px, **hog_options)
    tmpl_norm = float(np.linalg.norm(tmpl_fd))
    tmpl_h, tmpl_w = tmpl_px.shape[:2]

    # (score, similarity, scale, row, col) of each method's best window so far.
    best_euclid = (math.inf, 0.0, 1.0, 0, 0)
    best_dot = (-math.inf, 0.0, 1.0, 0, 0)
    for level in levels:
        resized = (
            scene
            if level.size == scene.size
            else scene.resize(level.size, Image.Resampling.LANCZOS)
        )
        scene_px, _ = _hog_pixels(resized, grayscale)
        height, width = scene_px.shape[:2]
        # Strict comparisons keep the first best window, like np.argmin/argmax.
        for i in range(0, height - tmpl_h + 1, step):
            for j in range(0, width - tmpl_w + 1, step):
                img_fd = feature.hog(scene_px[i : i + tmpl_h, j : j + tmpl_w], **hog_options)
                distance = float(np.linalg.norm(tmpl_fd - img_fd))
                dot = float(np.dot(img_fd, tmpl_fd))
                norms = tmpl_norm * float(np.linalg.norm(img_fd))
                similarity = dot / norms if norms > 0 else 0.0
                if distance < best_euclid[0]:
                    best_euclid = (distance, similarity, level.scale, i, j)
                if dot > best_dot[0]:
                    best_dot = (dot, similarity, level.scale, i, j)

    def detection(best: tuple[float, float, float, int, int]) -> Detection:
        score, similarity, scale, row, col = best
        box = (
            round(col * scale),
            round(row * scale),
            round(tmpl_w * scale),
            round(tmpl_h * scale),
        )
        return Detection(box, scale, score, similarity, similarity >= min_similarity)

    return HOGMatch(
        euclidean=detection(best_euclid),
        dot=detection(best_dot),
        levels=levels,
        windows=sum(level.windows for level in levels),
        descriptor_length=int(tmpl_fd.size),
    )


@dataclass(frozen=True)
class DetectionParams:
    template_path: Path
    template: Image.Image
    search: SearchParams


class HOGPersonDetectionTool(ForensicsTool):
    tool_id = "hog_person_detection"
    title = "Person detection (HOG)"
    category = "Features"
    description = "Locate a person template by multi-scale HOG matching (Set 4.4)."

    ACCEPTED_MODES = ACCEPTED_MODES

    # Sheet geometry: every panel is PANEL_H inches tall and as wide as its
    # image's aspect ratio asks for.
    PANEL_H = 4.0
    TITLE_IN = 1.1
    MIN_DPI = 100
    MAX_DPI = 300
    MAX_PIXELS = 16_000_000

    # The dialog's template thumbnail box, in pixels.
    THUMB_W = 76
    THUMB_H = 120

    def __init__(self) -> None:
        self._last_template: Path | None = None  # pre-fills the dialog on the next run

    def is_available(self, document: ImageDocument) -> bool:
        """Refuse unsupported modes and the tool's own output sheet."""
        if document.current is None:
            return True
        if document.current.mode not in self.ACCEPTED_MODES:
            return False
        return document.last_tool_id != self.tool_id

    def unavailable_message(self, document: ImageDocument) -> str:
        current = document.current
        if current is not None and current.mode not in self.ACCEPTED_MODES:
            return (
                f"This image mode is not supported for person detection "
                f"(mode {current.mode!r}).\n\nOpen an L or RGB image."
            )
        return (
            "The detection result is already shown.\n\n"
            "Undo or reset the image to search the original scene again."
        )

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        source = document.current
        assert source is not None
        assert source.mode in self.ACCEPTED_MODES, f"unsupported mode {source.mode!r}"

        params = self._ask_params(parent, source)
        if params is None:
            return None
        search = params.search

        try:
            match = match_hog_template(
                source,
                params.template,
                orientations=search.orientations,
                pixels_per_cell=search.pixels_per_cell,
                cells_per_block=search.cells_per_block,
                step=search.step,
                scales=search.scales,
                min_similarity=search.min_similarity,
            )
        except ValueError as error:
            messagebox.showinfo(self.title, str(error), parent=parent)
            return None

        def where(found: Detection) -> str:
            x, y, w, h = found.box
            return f"x={x}, y={y}, {w}×{h} ({found.scale:.2f}×)"

        threshold = search.min_similarity > 0

        def verdict(found: Detection) -> str:
            if not threshold:
                return f"{found.similarity:.2f} (no minimum set)"
            relation = "≥" if found.found else "<"
            outcome = "person found" if found.found else "no person"
            return f"{found.similarity:.2f} {relation} {search.min_similarity:.2f}: {outcome}"

        tmpl_w, tmpl_h = params.template.size
        euclid, dot = match.euclidean, match.dot
        same = euclid.box == dot.box
        if not threshold:
            message = (
                "Euclidean distance and dot product picked the same window."
                if same
                else "Euclidean distance and dot product picked different windows."
            )
        elif euclid.found and dot.found:
            message = (
                "Both methods found a person in the same window."
                if same
                else "Both methods found a person, in different windows."
            )
        elif euclid.found or dot.found:
            message = "Only one method found a person above the minimum similarity."
        else:
            message = "No person found: no window reached the minimum similarity."

        return ToolResult(
            image=self._render_sheet(source, params, match),
            message=message,
            details={
                "Operation": self.title,
                "Template": f"{params.template_path.name} ({tmpl_w}×{tmpl_h})",
                "Orientations": search.orientations,
                "Pixels per cell": f"({search.pixels_per_cell}, {search.pixels_per_cell})",
                "Cells per block": f"({search.cells_per_block}, {search.cells_per_block})",
                "Step size": search.step,
                "Scales searched": ", ".join(f"{level.scale:.2f}×" for level in match.levels),
                "Windows compared": match.windows,
                "Descriptor length": match.descriptor_length,
                "Euclidean best": where(euclid),
                "Euclidean distance": f"{euclid.score:.4f}",
                "Euclidean similarity": verdict(euclid),
                "Dot product best": where(dot),
                "Dot product": f"{dot.score:.4f}",
                "Dot product similarity": verdict(dot),
            },
        )

    def _render_sheet(
        self, source: Image.Image, params: DetectionParams, match: HOGMatch
    ) -> Image.Image:
        """Template and its HOG, then the original and both detections, as on the slide."""
        search = params.search
        grayscale = source.mode == "L"
        tmpl_px, channel_axis = _hog_pixels(params.template, grayscale)
        _, tmpl_hog = feature.hog(
            tmpl_px,
            orientations=search.orientations,
            pixels_per_cell=(search.pixels_per_cell, search.pixels_per_cell),
            cells_per_block=(search.cells_per_block, search.cells_per_block),
            visualize=True,
            channel_axis=channel_axis,
        )
        # Rescaled as in the skimage HOG example, so faint strokes are visible.
        tmpl_hog = exposure.rescale_intensity(tmpl_hog, in_range=(0, 10))
        scene_px, _ = _hog_pixels(source, grayscale)

        tmpl_w, tmpl_h = params.template.size
        scene_w, scene_h = source.size
        ratios = [tmpl_w / tmpl_h] * 2 + [scene_w / scene_h] * 3
        fig_w = self.PANEL_H * sum(ratios) * 1.15
        fig_h = self.PANEL_H + self.TITLE_IN

        native_dpi = scene_h / (self.PANEL_H * 0.85)
        dpi = min(self.MAX_DPI, max(self.MIN_DPI, native_dpi))
        dpi = min(dpi, math.sqrt(self.MAX_PIXELS / (fig_w * fig_h)))

        figure = Figure(figsize=(fig_w, fig_h), dpi=int(dpi), layout="constrained")
        axes = figure.subplots(1, 5, gridspec_kw={"width_ratios": ratios})
        gray = {"cmap": "gray", "vmin": 0, "vmax": 255} if grayscale else {}

        axes[0].imshow(tmpl_px, **gray)
        axes[0].set_title("Input image", fontsize=9)
        axes[1].imshow(tmpl_hog, cmap="gray")
        axes[1].set_title("Histogram of Oriented Gradients\n(Rescaled intensity)", fontsize=9)
        for axis in axes[:2]:
            axis.tick_params(labelsize=7)

        axes[2].imshow(scene_px, **gray)
        axes[2].set_title("Original image")
        axes[2].set_axis_off()
        for axis, title, found in (
            (axes[3], "Euclidean distance", match.euclidean),
            (axes[4], "Dot product", match.dot),
        ):
            axis.imshow(scene_px, **gray)
            axis.set_axis_off()
            similarity = f"similarity {found.similarity:.2f}"
            if found.found:
                x, y, w, h = found.box
                axis.add_patch(
                    Rectangle((x, y), w, h, facecolor="none", linewidth=2, edgecolor="red")
                )
                axis.set_title(f"{title}")
            else:
                axis.set_title(f"{title}\n{similarity} < {search.min_similarity:.2f}")
                axis.text(
                    0.5, 0.5, "No person found",
                    transform=axis.transAxes, ha="center", va="center",
                    color="white", fontsize=13, fontweight="bold",
                    bbox={"facecolor": "black", "alpha": 0.6, "edgecolor": "none", "pad": 8},
                )

        # Saved with a tight bounding box: the two-line detection titles would
        # otherwise run off the top of the fixed-size figure.
        FigureCanvasAgg(figure)
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0.15)
        plt.close(figure)
        buffer.seek(0)
        with Image.open(buffer) as rendered:
            return rendered.convert("RGB")

    def _ask_params(self, parent: tk.Misc | None, scene: Image.Image) -> DetectionParams | None:
        """One dialog: template file, HOG and search parameters; ``None`` on cancel.

        The footer shows how many scales and windows the search covers and
        roughly how long it takes, or why it cannot run, as values change.
        """
        dialog: ParamDialog[DetectionParams] = ParamDialog(
            parent,
            title=self.title,
            intro="Slide a template-sized window over the image at several sizes "
            "and mark the window whose HOG looks most like the template's.",
            action="Detect",
        )
        state: dict[str, object] = {"path": None, "template": None, "error": None, "thumb": None}

        frame = dialog.section("Person template")
        preview = ttk.Frame(frame, style="Preview.TFrame", width=self.THUMB_W, height=self.THUMB_H)
        preview.grid(row=0, column=0, rowspan=3, sticky="nw")
        preview.grid_propagate(False)
        preview.columnconfigure(0, weight=1)
        preview.rowconfigure(0, weight=1)
        thumb = ttk.Label(preview, style="Preview.TLabel", text="No\ntemplate")
        thumb.grid(row=0, column=0)

        name = ttk.Label(frame, style="Strong.TLabel", text="No image chosen")
        name.grid(row=0, column=1, sticky="sw", padx=(14, 0))
        info = ttk.Label(
            frame,
            style="Hint.TLabel",
            text="A cropped photo of one standing person.\nThe search window takes its size.",
            justify="left",
        )
        info.grid(row=1, column=1, sticky="nw", padx=(14, 0), pady=(2, 0))
        frame.rowconfigure(1, weight=1)

        def load(path: Path) -> None:
            try:
                with Image.open(path) as opened:
                    template = opened.convert("RGB")
            except OSError:
                state.update(
                    path=path, template=None, error=f"Cannot open {path.name} as an image."
                )
                name.configure(text=path.name)
                info.configure(text="Not a readable image file.")
                thumb.configure(image="", text="No\ntemplate")
            else:
                state.update(path=path, template=template, error=None)
                shown = template.copy()
                shown.thumbnail((self.THUMB_W - 8, self.THUMB_H - 8), Image.Resampling.LANCZOS)
                state["thumb"] = ImageTk.PhotoImage(shown, master=dialog.top)
                thumb.configure(image=state["thumb"], text="")
                name.configure(text=path.name)
                info.configure(
                    text=f"{template.width} × {template.height} px\n"
                    "The search window takes its size."
                )
                try:
                    step = int(search_vars[0].get())
                except ValueError:
                    step = SEARCH_FIELDS[0].default
                smallest, largest = default_size_range(
                    scene.size, template.size, step=max(step, 1)
                )
                search_vars[1].set(str(smallest))
                search_vars[2].set(str(largest))
            dialog.refresh()

        def choose() -> None:
            chosen = filedialog.askopenfilename(
                parent=dialog.top,
                title="Choose a person template",
                filetypes=OPEN_TYPES,
                initialdir=str(self._last_template.parent) if self._last_template else None,
            )
            if chosen:
                load(Path(chosen))

        ttk.Button(frame, text="Choose image…", command=choose).grid(
            row=2, column=1, sticky="sw", padx=(14, 0)
        )

        hog_vars = dialog.int_fields(dialog.section("HOG descriptor"), HOG_FIELDS)
        search_vars = dialog.int_fields(dialog.section("Search"), SEARCH_FIELDS)

        def validate() -> tuple[DetectionParams, str]:
            if state["error"]:
                raise ValueError(str(state["error"]))
            template = state["template"]
            if template is None:
                raise ValueError("Choose a person template to search for.")
            assert isinstance(template, Image.Image)
            search = parse_search_params(*(v.get() for v in hog_vars + search_vars))
            levels = plan_search(
                scene.size,
                template.size,
                pixels_per_cell=search.pixels_per_cell,
                cells_per_block=search.cells_per_block,
                step=search.step,
                scales=search.scales,
            )
            path = state["path"]
            assert isinstance(path, Path)
            params = DetectionParams(path, template, search)
            return params, search_summary(levels, len(search.scales), template.size)

        if self._last_template is not None and self._last_template.exists():
            load(self._last_template)
        result = dialog.show(validate)
        if result is not None:
            self._last_template = result.template_path
        return result
