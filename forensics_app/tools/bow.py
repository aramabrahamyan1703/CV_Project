"""Bag-of-visual-words codebook, histograms, and classification (Set 7).

Slide pipeline:
1. Stack descriptors from a class-folder dataset and cluster with
   ``MiniBatchKMeans`` (``K = 10 * num_classes``); each centre is a visual word.
2. Assign each image's descriptors to the closest centre and build a
   normalized frequency histogram.
3. Train a probabilistic classifier on training histograms; for a new image
   show the histogram and ``p(class) = …`` for the top class.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image, ImageOps
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .descriptors import ACCEPTED_MODES, DescriptorParams, extract_descriptors
from .param_dialog import FolderChooser, ParamDialog


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
WORDS_PER_CLASS = 10
PATCH_SIZE = 32
# Cap training images per class so a full Caltech subset stays interactive.
MAX_TRAIN_PER_CLASS = 40
# Default detector: ORB is much faster than SIFT on hundreds of images.
DEFAULT_DETECTOR = DescriptorParams(method="orb", n_keypoints=200)

# In-memory model shared by the three Set 7 tools (rebuilt by CodebookTool).
_MODEL: "BowModel | None" = None


@dataclass(frozen=True)
class Codebook:
    centroids: np.ndarray  # (K, D) float
    class_names: tuple[str, ...]
    patches: tuple[Image.Image, ...]  # one illustrative patch per visual word
    detector: DescriptorParams

    @property
    def size(self) -> int:
        return int(self.centroids.shape[0])


@dataclass(frozen=True)
class BowModel:
    codebook: Codebook
    classifier: Pipeline  # StandardScaler + LogisticRegression
    train_accuracy: float


def get_model() -> BowModel | None:
    return _MODEL


def set_model(model: BowModel | None) -> None:
    global _MODEL
    _MODEL = model


def clear_model() -> None:
    set_model(None)


def list_class_folders(root: Path) -> list[tuple[str, list[Path]]]:
    """``[(class_name, image_paths), ...]`` for immediate subfolders of ``root``."""
    if not root.is_dir():
        raise ValueError(f"Not a folder: {root}")
    classes: list[tuple[str, list[Path]]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        images = sorted(
            p
            for p in child.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if images:
            classes.append((child.name, images))
    if len(classes) < 2:
        raise ValueError(
            "Need at least two class subfolders with images "
            "(e.g. butterfly/, kangaroo/, dalmatian/)."
        )
    return classes


def describe_dataset(folder: Path) -> str:
    """One-line summary for the folder chooser."""
    try:
        classes = list_class_folders(folder)
    except ValueError as error:
        return str(error)
    n_images = sum(len(paths) for _, paths in classes)
    k = WORDS_PER_CLASS * len(classes)
    return f"{len(classes)} classes · {n_images} images · K={k} visual words"


def _extract(image: Image.Image, detector: DescriptorParams) -> tuple[np.ndarray, np.ndarray]:
    return extract_descriptors(
        image,
        method=detector.method,
        n_keypoints=detector.n_keypoints,
        n_octaves=detector.n_octaves,
        n_scales=detector.n_scales,
        sigma_min=detector.sigma_min,
    )


def _crop_patch(image: Image.Image, row: float, col: float, size: int = PATCH_SIZE) -> Image.Image:
    """Square grayscale patch centred on ``(row, col)``, padded if near the border."""
    gray = image.convert("L")
    half = size // 2
    left = int(round(col)) - half
    top = int(round(row)) - half
    right = left + size
    bottom = top + size
    pad_left = max(0, -left)
    pad_top = max(0, -top)
    pad_right = max(0, right - gray.width)
    pad_bottom = max(0, bottom - gray.height)
    if pad_left or pad_top or pad_right or pad_bottom:
        gray = ImageOps.expand(gray, border=(pad_left, pad_top, pad_right, pad_bottom), fill=0)
        left += pad_left
        top += pad_top
        right = left + size
        bottom = top + size
    return gray.crop((left, top, right, bottom))


def build_codebook(
    dataset_root: Path,
    *,
    detector: DescriptorParams = DEFAULT_DETECTOR,
    max_per_class: int = MAX_TRAIN_PER_CLASS,
    random_state: int = 0,
) -> BowModel:
    """Cluster training descriptors, visualize patches, and train a classifier."""
    classes = list_class_folders(dataset_root)
    class_names = tuple(name for name, _ in classes)
    n_classes = len(class_names)
    n_clusters = WORDS_PER_CLASS * n_classes

    all_descriptors: list[np.ndarray] = []
    patch_sources: list[tuple[Image.Image, float, float]] = []
    # Per-image training histograms for the classifier.
    train_hists: list[np.ndarray] = []
    train_labels: list[int] = []

    # First pass: collect every descriptor (and a patch source) for clustering.
    per_image: list[tuple[int, np.ndarray]] = []  # (label_index, descriptors)
    for label, (name, paths) in enumerate(classes):
        for path in paths[:max_per_class]:
            with Image.open(path) as opened:
                image = opened.copy()
            try:
                keypoints, descriptors = _extract(image, detector)
            except ValueError:
                continue
            desc = np.asarray(descriptors, dtype=np.float64)
            if desc.ndim != 2 or desc.shape[0] == 0:
                continue
            all_descriptors.append(desc)
            for i in range(len(keypoints)):
                patch_sources.append((image, float(keypoints[i, 0]), float(keypoints[i, 1])))
            per_image.append((label, desc))

    if not all_descriptors:
        raise ValueError(
            "No descriptors could be extracted from the training folder.\n\n"
            "Check that the images are textured enough for ORB/SIFT."
        )

    stacked = np.vstack(all_descriptors)
    if stacked.shape[0] < n_clusters:
        raise ValueError(
            f"Need at least {n_clusters} descriptors for K={n_clusters}, "
            f"but only found {stacked.shape[0]}. Add more images or keypoints."
        )

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=3,
        batch_size=min(1024, stacked.shape[0]),
    )
    kmeans.fit(stacked)
    centroids = np.asarray(kmeans.cluster_centers_, dtype=np.float64)

    # Illustrative patch = training patch whose descriptor is closest to each centre.
    patches: list[Image.Image] = []
    # Approximate nearest training descriptor via kmeans labels on the fit data.
    labels = kmeans.labels_
    for word in range(n_clusters):
        members = np.where(labels == word)[0]
        if len(members) == 0:
            patches.append(Image.new("L", (PATCH_SIZE, PATCH_SIZE), 0))
            continue
        # Among members, pick the one closest to the centroid.
        member_desc = stacked[members]
        dist = np.linalg.norm(member_desc - centroids[word], axis=1)
        best = int(members[int(np.argmin(dist))])
        image, row, col = patch_sources[best]
        patches.append(_crop_patch(image, row, col))

    codebook = Codebook(
        centroids=centroids,
        class_names=class_names,
        patches=tuple(patches),
        detector=detector,
    )

    for label, desc in per_image:
        train_hists.append(histogram_from_descriptors(desc, codebook))
        train_labels.append(label)

    x = np.vstack(train_hists)
    y = np.asarray(train_labels, dtype=np.int64)
    classifier: Pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(max_iter=1000, random_state=random_state),
            ),
        ]
    )
    classifier.fit(x, y)
    accuracy = float(classifier.score(x, y))
    model = BowModel(codebook=codebook, classifier=classifier, train_accuracy=accuracy)
    set_model(model)
    return model


def histogram_from_descriptors(descriptors: np.ndarray, codebook: Codebook) -> np.ndarray:
    """Normalized frequency histogram over the codebook (sums to 1)."""
    desc = np.asarray(descriptors, dtype=np.float64)
    if desc.ndim != 2 or desc.shape[0] == 0:
        raise ValueError("No descriptors to assign to visual words.")
    if desc.shape[1] != codebook.centroids.shape[1]:
        raise ValueError(
            "Descriptor length does not match the codebook. "
            "Rebuild the codebook with the same detector."
        )
    # Closest centroid for each descriptor (Euclidean).
    # (N,1,D) - (1,K,D) → (N,K)
    diff = desc[:, None, :] - codebook.centroids[None, :, :]
    dist = np.einsum("nkd,nkd->nk", diff, diff)
    assignments = np.argmin(dist, axis=1)
    hist = np.bincount(assignments, minlength=codebook.size).astype(np.float64)
    total = hist.sum()
    if total <= 0:
        raise ValueError("Histogram is empty.")
    return hist / total


def histogram_for_image(image: Image.Image, codebook: Codebook) -> np.ndarray:
    """Extract descriptors from ``image`` and build a normalized BoW histogram."""
    _keypoints, descriptors = _extract(image, codebook.detector)
    return histogram_from_descriptors(descriptors, codebook)


def classify_histogram(histogram: np.ndarray, model: BowModel) -> tuple[str, float, np.ndarray]:
    """Return ``(top_class, probability, all_probabilities)``."""
    hist = np.asarray(histogram, dtype=np.float64).reshape(1, -1)
    if hist.shape[1] != model.codebook.size:
        raise ValueError("Histogram size does not match the codebook.")
    probs = model.classifier.predict_proba(hist)[0]
    index = int(np.argmax(probs))
    return model.codebook.class_names[index], float(probs[index]), probs


def classify_image(image: Image.Image, model: BowModel) -> tuple[str, float, np.ndarray, np.ndarray]:
    """``(label, probability, all_probs, histogram)`` for ``image``."""
    hist = histogram_for_image(image, model.codebook)
    label, prob, probs = classify_histogram(hist, model)
    return label, prob, probs, hist


def render_visual_words(patches: tuple[Image.Image, ...], *, columns: int = 10) -> Image.Image:
    """Grid of ``vw i`` patches, as on the Set 7.1 slide."""
    n = len(patches)
    columns = max(1, min(columns, n))
    rows = (n + columns - 1) // columns
    fig_w = columns * 0.7
    fig_h = rows * 0.9
    figure = Figure(figsize=(fig_w, fig_h), dpi=120, layout="constrained")
    axes = figure.subplots(rows, columns, squeeze=False)
    for i in range(rows * columns):
        ax = axes[i // columns][i % columns]
        ax.set_xticks([])
        ax.set_yticks([])
        if i >= n:
            ax.axis("off")
            continue
        ax.imshow(np.asarray(patches[i].convert("L")), cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"vw {i}", fontsize=7, pad=2)
        for spine in ax.spines.values():
            spine.set_visible(False)
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    image = Image.fromarray(np.asarray(canvas.buffer_rgba())).convert("RGB")
    plt.close(figure)
    return image


def render_histogram_sheet(image: Image.Image, histogram: np.ndarray, *, title: str = "") -> Image.Image:
    """Image beside a normalized visual-word histogram (Set 7.2)."""
    figure = Figure(figsize=(9, 3.8), dpi=120, layout="constrained")
    ax_img, ax_hist = figure.subplots(1, 2, width_ratios=[1, 1.4])
    show = image.convert("RGB") if image.mode != "L" else image
    if show.mode == "L":
        ax_img.imshow(np.asarray(show), cmap="gray", vmin=0, vmax=255)
    else:
        ax_img.imshow(np.asarray(show))
    ax_img.set_title(title or "Image")
    ax_img.axis("off")

    indices = np.arange(len(histogram))
    ax_hist.bar(indices, histogram, width=0.9, color="steelblue")
    ax_hist.set_xlabel("Visual word index")
    ax_hist.set_ylabel("Normalized frequency")
    ax_hist.set_title("Histogram of visual words")
    ax_hist.set_xlim(-0.5, len(histogram) - 0.5)

    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    out = Image.fromarray(np.asarray(canvas.buffer_rgba())).convert("RGB")
    plt.close(figure)
    return out


def render_classification(
    image: Image.Image, label: str, probability: float, histogram: np.ndarray
) -> Image.Image:
    """Histogram sheet plus ``p(label) = …`` overlay text (Set 7.3)."""
    sheet = render_histogram_sheet(image, histogram, title=f"p({label}) = {probability:.2f}")
    return sheet


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class CodebookTool(ForensicsTool):
    tool_id = "bow_codebook"
    title = "BoW codebook"
    category = "Recognition"
    description = (
        "Build a visual-word codebook with MiniBatchKMeans "
        "(K = 10 × classes) and train a classifier (Set 7.1)."
    )
    requires_image = False

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        folder = self._ask_folder(parent)
        if folder is None:
            return None
        try:
            model = build_codebook(folder)
        except ValueError as error:
            messagebox.showinfo(self.title, str(error), parent=parent)
            return None

        sheet = render_visual_words(model.codebook.patches)
        names = ", ".join(model.codebook.class_names)
        return ToolResult(
            image=sheet,
            message=(
                f"Built codebook with {model.codebook.size} visual words "
                f"({names}); train accuracy {model.train_accuracy:.0%}."
            ),
            details={
                "Operation": self.title,
                "Function": "sklearn.cluster.MiniBatchKMeans",
                "Classes": names,
                "K": model.codebook.size,
                "Detector": model.codebook.detector.method.upper(),
                "Train accuracy": f"{model.train_accuracy:.3f}",
            },
        )

    def _ask_folder(self, parent: tk.Misc | None) -> Path | None:
        dialog: ParamDialog[Path] = ParamDialog(
            parent,
            title=self.title,
            intro="Choose a folder whose subfolders are class names "
            "(e.g. Caltech_101_subset with butterfly, dalmatian, kangaroo). "
            f"K = {WORDS_PER_CLASS} × number of classes.",
            action="Build",
        )
        chooser = FolderChooser(
            dialog,
            dialog.section("Training dataset"),
            prompt="Choose a dataset folder with class subfolders",
            describe=describe_dataset,
        )

        def validate() -> tuple[Path, str]:
            folder = chooser.require()
            classes = list_class_folders(folder)
            return folder, describe_dataset(folder)

        return dialog.show(validate)


class BowHistogramTool(ForensicsTool):
    tool_id = "bow_histogram"
    title = "BoW histogram"
    category = "Recognition"
    description = (
        "Assign the working image's features to visual words and plot the "
        "normalized histogram (Set 7.2)."
    )

    ACCEPTED_MODES = ACCEPTED_MODES

    def is_available(self, document: ImageDocument) -> bool:
        if get_model() is None:
            return False
        if document.current is None:
            return True
        if document.current.mode not in self.ACCEPTED_MODES:
            return False
        return document.last_tool_id != self.tool_id

    def unavailable_message(self, document: ImageDocument) -> str:
        if get_model() is None:
            return (
                "No codebook yet.\n\n"
                "Run BoW codebook first on a class-folder dataset."
            )
        current = document.current
        if current is not None and current.mode not in self.ACCEPTED_MODES:
            return (
                f"This image mode is not supported (mode {current.mode!r}).\n\n"
                "Open an L or RGB image."
            )
        return (
            "The BoW histogram is already shown.\n\n"
            "Undo or reset to build it again."
        )

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        model = get_model()
        assert model is not None
        source = document.current
        assert source is not None
        try:
            hist = histogram_for_image(source, model.codebook)
        except ValueError as error:
            messagebox.showinfo(self.title, str(error), parent=parent)
            return None
        return ToolResult(
            image=render_histogram_sheet(source, hist),
            message=f"Built a {model.codebook.size}-bin visual-word histogram.",
            details={
                "Operation": self.title,
                "Codebook K": model.codebook.size,
                "Non-zero bins": int(np.count_nonzero(hist)),
                "Peak word": int(np.argmax(hist)),
                "Peak frequency": f"{float(hist.max()):.3f}",
            },
        )


class BowClassifyTool(ForensicsTool):
    tool_id = "bow_classify"
    title = "BoW classify"
    category = "Recognition"
    description = (
        "Classify the working image with the trained BoW model and show "
        "p(class) with its histogram (Set 7.3)."
    )

    ACCEPTED_MODES = ACCEPTED_MODES

    def is_available(self, document: ImageDocument) -> bool:
        if get_model() is None:
            return False
        if document.current is None:
            return True
        if document.current.mode not in self.ACCEPTED_MODES:
            return False
        return document.last_tool_id != self.tool_id

    def unavailable_message(self, document: ImageDocument) -> str:
        if get_model() is None:
            return (
                "No trained classifier yet.\n\n"
                "Run BoW codebook first on a class-folder dataset."
            )
        current = document.current
        if current is not None and current.mode not in self.ACCEPTED_MODES:
            return (
                f"This image mode is not supported (mode {current.mode!r}).\n\n"
                "Open an L or RGB image."
            )
        return (
            "The classification result is already shown.\n\n"
            "Undo or reset to classify again."
        )

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        model = get_model()
        assert model is not None
        source = document.current
        assert source is not None
        try:
            label, prob, probs, hist = classify_image(source, model)
        except ValueError as error:
            messagebox.showinfo(self.title, str(error), parent=parent)
            return None

        details: dict[str, object] = {
            "Operation": self.title,
            "Predicted": label,
            "Probability": f"{prob:.3f}",
            "Train accuracy": f"{model.train_accuracy:.3f}",
        }
        for name, p in zip(model.codebook.class_names, probs, strict=True):
            details[f"p({name})"] = f"{float(p):.3f}"

        return ToolResult(
            image=render_classification(source, label, prob, hist),
            message=f"p({label}) = {prob:.2f}",
            details=details,
        )
