"""Find the image in a folder that best matches the working image (Set 5.4).

The slide's algorithm, "return the image with the maximum number of matches",
is the default: with SIFT and a 0.8 ratio test it found the right image for
7 of 7 references among the course's Set 5 images. A RANSAC ranking is
offered as an alternative: it counts only the matches that agree on one
affine transform (shift, rotation, scale, shear) between the two images. It
did not beat plain counting there (6/7), because photos of 3D buildings from
different spots do not follow one transform; it can help with flat subjects
such as logos and signs on busy, repetitive backgrounds.

The folder is searched with its subfolders (the course images come grouped
in them). The reference never counts as its own best match: its file is left
out, and so is any other file showing the same picture, such as a copy saved
under another name or format.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk
import warnings

import numpy as np
from PIL import Image
from skimage import measure, transform

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .descriptors import ACCEPTED_MODES
from .matching import (
    MatchParams,
    closest,
    extract,
    match_features,
    matching_sections,
    matching_summary,
    parse_matching_params,
    render_matches,
    to_work_size,
)
from .param_dialog import FolderChooser, ParamDialog, ProgressWindow


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# Every image and the reference are resized to matching.WORK_SIDE first.
# Measured seconds per folder image at that size, including matching.
SECONDS_PER_IMAGE = {"orb": 0.4, "sift": 1.4}  # SIFT: 1.15–1.6 s on Set 5
MAX_SECONDS = 120

RANKINGS = {
    "matches": "Most matches",
    "ransac": "Consistent matches (RANSAC)",
}
# An affine transform needs 3 point pairs; RANSAC is skipped below that.
MIN_RANSAC_MATCHES = 3

# Two files show the same picture when their proportions agree within this
# share and their 32×32 grayscale thumbnails differ by at most this much on
# average (0..255). Re-saving as JPEG or resizing stays well under it; the
# closest two different course images differ by 46.
SAME_PICTURE_ASPECT = 0.02
SAME_PICTURE_DIFFERENCE = 3.0

# The best image is called a clear match when its score is at least this many
# times the folder's median score; below that it is reported as weak. With
# every Set 5 image as the reference in turn, every clear answer was right
# (SIFT 11/11, ORB 10/10); the weak ones were a mix, including every wrong
# answer (e.g. eiffel_tower2, whose other photo shares almost no features
# with it: the "best" was a near tie between unrelated images).
CLEAR_STAND_OUT = 1.7
# Ratios of tiny counts mean nothing (2 matches is "2× a median of 1"), so a
# clear match also needs this score; right answers on Set 5 had at least 17.
MIN_CLEAR_SCORE = 10
# A median of fewer images says nothing about the folder's typical score.
MIN_IMAGES_FOR_CONFIDENCE = 3

# Every keypoint is circled, as on the slide, only up to this many per image:
# SIFT finds ~3,000 in a 640 px photo and the circles would hide it. Above
# that, only the matched keypoints are circled.
MAX_KEYPOINTS_DRAWN = 600


@dataclass(frozen=True)
class Candidate:
    path: Path
    image: Image.Image  # at the working size, as compared
    keypoints: np.ndarray
    matches: np.ndarray
    distances: np.ndarray
    inliers: np.ndarray  # bool per match: consistent with the RANSAC transform

    @property
    def match_count(self) -> int:
        return len(self.matches)

    @property
    def consistent_count(self) -> int:
        return int(self.inliers.sum())


@dataclass(frozen=True)
class LookupResult:
    reference: Image.Image  # at the working size, as compared
    reference_keypoints: np.ndarray
    ranked: tuple[Candidate, ...]  # best first
    skipped: tuple[tuple[Path, str], ...]  # (file, reason)
    ranking: str = "matches"

    def score(self, candidate: Candidate) -> int:
        """The count the ranking sorts by."""
        if self.ranking == "ransac":
            return candidate.consistent_count
        return candidate.match_count

    @property
    def stand_out(self) -> float | None:
        """Best score over the median score of all compared images (None if too few)."""
        return stand_out([self.score(candidate) for candidate in self.ranked])

    @property
    def clear(self) -> bool:
        """Whether the best image stands out enough to trust; see CLEAR_STAND_OUT."""
        value = self.stand_out
        if value is None or not self.ranked:
            return False
        return value >= CLEAR_STAND_OUT and self.score(self.ranked[0]) >= MIN_CLEAR_SCORE


def stand_out(scores: list[int]) -> float | None:
    """How many times the median the top score is; None below MIN_IMAGES_FOR_CONFIDENCE."""
    if len(scores) < MIN_IMAGES_FOR_CONFIDENCE:
        return None
    return max(scores) / max(float(np.median(scores)), 1.0)


class SearchCancelled(Exception):
    """The user pressed Cancel during the search."""


def list_images(folder: Path, exclude: Path | None = None) -> list[Path]:
    """Image files in ``folder`` and its subfolders, sorted, minus the file ``exclude``.

    Hidden files and folders (``.DS_Store``, ``._photo.jpg`` and the like) are
    skipped. ``exclude`` is compared as a file, not as a path string, so a
    different spelling, letter case or link to the same file is left out too.
    """
    found = []
    for path in folder.rglob("*"):
        if any(part.startswith(".") for part in path.relative_to(folder).parts):
            continue
        if path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file():
            if exclude is None or not _same_file(path, exclude):
                found.append(path)
    return sorted(found)


def _same_file(first: Path, second: Path) -> bool:
    try:
        return first.samefile(second)
    except OSError:
        return False


@dataclass(frozen=True)
class Fingerprint:
    aspect: float  # width / height
    thumbnail: np.ndarray  # 32×32 grayscale, as float


def fingerprint(image: Image.Image) -> Fingerprint:
    thumbnail = image.convert("L").resize((32, 32), Image.Resampling.BILINEAR)
    return Fingerprint(image.width / image.height, np.asarray(thumbnail, dtype=np.float64))


def same_picture(first: Fingerprint, second: Fingerprint) -> bool:
    """Whether two images show the same picture, regardless of size or file format."""
    if abs(first.aspect / second.aspect - 1) > SAME_PICTURE_ASPECT:
        return False
    return float(np.abs(first.thumbnail - second.thumbnail).mean()) <= SAME_PICTURE_DIFFERENCE


def estimate_seconds(count: int, method: str) -> float:
    return count * SECONDS_PER_IMAGE[method]


def consistent_matches(
    keypoints1: np.ndarray,
    keypoints2: np.ndarray,
    matches: np.ndarray,
    size2: tuple[int, int],
) -> np.ndarray:
    """Which matches fit one affine transform, by RANSAC; all False if none can.

    The tolerance is 1% of the second image's diagonal (at least 3 px), and
    the random generator is seeded so the same input always ranks the same.
    """
    inliers = np.zeros(len(matches), dtype=bool)
    if len(matches) < MIN_RANSAC_MATCHES:
        return inliers
    # Keypoints are (row, col); transforms work in (x, y).
    source = keypoints1[matches[:, 0]][:, ::-1]
    target = keypoints2[matches[:, 1]][:, ::-1]
    tolerance = max(3.0, 0.01 * math.hypot(*size2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # degenerate samples warn, then get skipped
        model, found = measure.ransac(
            (source, target),
            transform.AffineTransform,
            min_samples=MIN_RANSAC_MATCHES,
            residual_threshold=tolerance,
            max_trials=500,
            rng=0,
        )
    if model is None or found is None:
        return inliers
    return np.asarray(found, dtype=bool)


def find_closest_image(
    reference: Image.Image,
    paths: list[Path],
    params: MatchParams,
    ranking: str = "matches",
    *,
    also_same_as: tuple[Image.Image, ...] = (),
    progress: Callable[[int, Path], bool] | None = None,
) -> LookupResult:
    """Compare ``reference`` with every image in ``paths`` and rank them, best first.

    Files showing the same picture as ``reference`` (or as any image in
    ``also_same_as``, e.g. the unedited original) are skipped, so the
    reference cannot find itself; so are unreadable files and images without
    keypoints, each with a reason. Ties are broken by the other count, then by
    path. ``progress(done, next_path)`` is called before each file; returning
    False stops the search with :class:`SearchCancelled`.
    """
    if ranking not in RANKINGS:
        raise ValueError(f"Ranking must be one of {', '.join(RANKINGS)}.")
    if not paths:
        raise ValueError("The folder has no images to compare with.")
    small_reference = to_work_size(reference)
    ref_keypoints, ref_descriptors = extract(small_reference, params.detector)
    references = [fingerprint(small_reference)]
    references += [fingerprint(to_work_size(image)) for image in also_same_as]

    candidates: list[Candidate] = []
    skipped: list[tuple[Path, str]] = []
    for done, path in enumerate(paths):
        if progress is not None and not progress(done, path):
            raise SearchCancelled
        try:
            with Image.open(path) as opened:
                image = to_work_size(opened)
        except OSError:
            skipped.append((path, "not a readable image"))
            continue
        if any(same_picture(fingerprint(image), known) for known in references):
            skipped.append((path, "same picture as the reference"))
            continue
        try:
            keypoints, descriptors = extract(image, params.detector)
        except ValueError:
            skipped.append((path, "no keypoints found"))
            continue
        result = match_features(ref_keypoints, ref_descriptors, keypoints, descriptors, params)
        inliers = consistent_matches(ref_keypoints, keypoints, result.matches, image.size)
        candidates.append(
            Candidate(path, image, keypoints, result.matches, result.distances, inliers)
        )

    if not candidates:
        reasons = sorted({why for _, why in skipped})
        raise ValueError(
            f"None of the {len(paths)} images could be compared "
            f"({'; '.join(reasons)})."
        )

    def key(candidate: Candidate) -> tuple[int, int, str]:
        counts = (candidate.consistent_count, candidate.match_count)
        if ranking == "matches":
            counts = counts[::-1]
        return (-counts[0], -counts[1], str(candidate.path))

    return LookupResult(
        small_reference,
        ref_keypoints,
        tuple(sorted(candidates, key=key)),
        tuple(skipped),
        ranking,
    )


@dataclass(frozen=True)
class LookupRequest:
    folder: Path
    paths: tuple[Path, ...]
    params: MatchParams
    ranking: str


class ImageLookupTool(ForensicsTool):
    tool_id = "image_lookup"
    title = "Closest image lookup"
    category = "Features"
    description = "Find the most similar image in a folder by keypoint matching (Set 5.4)."

    ACCEPTED_MODES = ACCEPTED_MODES
    TOP_RANKS = 10  # rows of the ranking shown in the Results panel

    def __init__(self) -> None:
        self._last_folder: Path | None = None  # pre-fills the dialog on the next run

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
                f"This image mode is not supported for lookup (mode {current.mode!r}).\n\n"
                "Open an L or RGB image."
            )
        return (
            "The closest image is already shown.\n\n"
            "Undo or reset the image to search with the original again."
        )

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        source = document.current
        assert source is not None
        assert source.mode in self.ACCEPTED_MODES, f"unsupported mode {source.mode!r}"

        request = self._ask_params(parent, document.path)
        if request is None:
            return None
        params = request.params

        def name(path: Path) -> str:
            """The file's path inside the searched folder, e.g. ``buildings/a.jpg``."""
            return path.relative_to(request.folder).as_posix()

        # The unedited original counts as "the reference" too, so a copy of the
        # opened file is left out even after the working image was edited.
        same_as = (document.original,) if document.original is not None else ()
        window = (
            ProgressWindow(parent, title="Searching the folder…", total=len(request.paths))
            if parent is not None
            else None
        )

        def progress(done: int, path: Path) -> bool:
            if window is None:
                return True
            return window.step(done, f"{done + 1} of {len(request.paths)}: {name(path)}")

        try:
            lookup = find_closest_image(
                source,
                list(request.paths),
                params,
                request.ranking,
                also_same_as=same_as,
                progress=progress,
            )
        except SearchCancelled:
            return None
        except ValueError as error:
            if window is not None:
                window.close()
                window = None
            messagebox.showinfo(self.title, str(error), parent=parent)
            return None
        finally:
            if window is not None:
                window.close()

        best = lookup.ranked[0]
        reference_name = document.path.name if document.path is not None else "working image"
        if request.ranking == "ransac":
            # Draw only the matches RANSAC kept: the ones that decided the ranking.
            shown, distances = best.matches[best.inliers], best.distances[best.inliers]
            counts = f"matches={best.match_count}, consistent={best.consistent_count}"
        else:
            shown, distances = best.matches, best.distances
            counts = f"matches={best.match_count}"
        drawn = closest(shown, distances, params.lines)

        factor = lookup.stand_out
        if factor is None:
            confidence = (
                f"unknown: only {len(lookup.ranked)} image(s) compared, "
                f"{MIN_IMAGES_FOR_CONFIDENCE} are needed to judge"
            )
        else:
            level = "clear" if lookup.clear else "weak, may be wrong"
            confidence = f"{level} ({factor:.1f}× the folder's median score)"
            if not lookup.clear and lookup.score(best) < MIN_CLEAR_SCORE:
                confidence = f"weak, may be wrong (only {lookup.score(best)} matches)"
        title = f"Reference: {reference_name}  <--->  Best: {name(best.path)}  ({counts})"
        if not lookup.clear:
            title += "\nNo clear match: the best image barely stands out from the rest"
        sheet = render_matches(
            lookup.reference,
            best.image,
            lookup.reference_keypoints,
            best.keypoints,
            drawn,
            title=title,
            all_keypoints=max(len(lookup.reference_keypoints), len(best.keypoints))
            <= MAX_KEYPOINTS_DRAWN,
            keypoints_color="r",
        )

        details: dict[str, object] = {
            "Operation": self.title,
            "Reference": reference_name,
            "Folder": str(request.folder),
            "Ranking": RANKINGS[request.ranking],
            "Detector": params.detector.method.upper(),
            "Matching": matching_summary(params),
            "Images compared": len(lookup.ranked),
            "Best": name(best.path),
            "Confidence": confidence,
        }
        for rank, candidate in enumerate(lookup.ranked[: self.TOP_RANKS], start=1):
            noun = "match" if candidate.match_count == 1 else "matches"
            details[f"Rank {rank}"] = (
                f"{name(candidate.path)}: {candidate.match_count} {noun} · "
                f"{candidate.consistent_count} consistent"
            )
        if lookup.skipped:
            details["Skipped"] = "; ".join(f"{name(path)} ({why})" for path, why in lookup.skipped)

        return ToolResult(
            image=sheet,
            message=(
                f"Closest image: {name(best.path)} ({counts.replace('=', ' ')})."
                if lookup.clear
                else f"No clear match. Best guess: {name(best.path)} "
                f"({counts.replace('=', ' ')}), barely above the other images."
            ),
            details=details,
        )

    def _ask_params(self, parent: tk.Misc | None, reference_path: Path | None) -> LookupRequest | None:
        """One dialog: folder, detector, matching and ranking; ``None`` on cancel."""
        dialog: ParamDialog[LookupRequest] = ParamDialog(
            parent,
            title=self.title,
            intro="Match the working image against every image in a folder "
            "and show the one that matches best.",
            action="Search",
        )

        # Listing a folder walks all its subfolders, so it is done once per
        # folder chosen, not on every edit in the dialog.
        listings: dict[Path, list[Path] | None] = {}

        def listing(folder: Path) -> list[Path] | None:
            if folder not in listings:
                try:
                    listings[folder] = list_images(folder, reference_path)
                except OSError:
                    listings[folder] = None
            return listings[folder]

        def describe(folder: Path) -> str:
            paths = listing(folder)
            if paths is None:
                return "Cannot read this folder."
            noun = "image" if len(paths) == 1 else "images"
            return f"{len(paths)} {noun} found, subfolders included"

        chooser = FolderChooser(
            dialog,
            dialog.section("Folder"),
            prompt="Choose a folder of images to search.",
            describe=describe,
            initial=self._last_folder,
        )
        # SIFT: right 7/7 times on the Set 5 images, ORB 5/7 (see matching.py).
        read_detector, cross_check, ratio_var, lines_var = matching_sections(
            dialog, method="sift"
        )

        ranking_frame = dialog.section("Ranking")
        ranking_var = tk.StringVar(value="matches")
        for row, (value, label) in enumerate(RANKINGS.items()):
            ttk.Radiobutton(ranking_frame, text=label, variable=ranking_var, value=value).grid(
                row=row, column=0, sticky="w", pady=(0, 0) if row == 0 else (6, 0)
            )
        ttk.Label(
            ranking_frame,
            text="RANSAC counts only matches that agree on one\n"
            "shift, rotation and scale: best for flat subjects.",
            style="Hint.TLabel",
            justify="left",
        ).grid(row=len(RANKINGS), column=0, sticky="w", pady=(6, 0))
        ranking_var.trace_add("write", lambda *_args: dialog.refresh())

        def validate() -> tuple[LookupRequest, str]:
            folder = chooser.require()
            paths = listing(folder)
            if paths is None:
                raise ValueError(f"Cannot read the folder {folder.name}.")
            if not paths:
                raise ValueError("This folder has no images to compare with.")
            detector, _ = read_detector()
            params = parse_matching_params(
                detector, cross_check.get(), ratio_var.get(), lines_var.get()
            )
            seconds = estimate_seconds(len(paths), detector.method)
            if seconds > MAX_SECONDS:
                raise ValueError(
                    f"{len(paths)} images would take about {seconds:.0f} s with "
                    f"{detector.method.upper()}. Use a folder with fewer images"
                    + (" or ORB." if detector.method == "sift" else ".")
                )
            timing = "under 1 s" if seconds < 1 else f"about {seconds:.0f} s"
            noun = "image" if len(paths) == 1 else "images"
            readout = f"{len(paths)} {noun} · {detector.method.upper()} · {timing}"
            return LookupRequest(folder, tuple(paths), params, ranking_var.get()), readout

        request = dialog.show(validate)
        if request is not None:
            self._last_folder = request.folder
        return request
