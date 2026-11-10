"""Bag-of-visual-words codebook, histograms, and classification (Set 7).

Slide pipeline:
1. Stack descriptors from a class-folder dataset and cluster with
   ``MiniBatchKMeans`` (``K = 10 * num_classes``); each centre is a visual word.
2. Assign each image's descriptors to the closest centre and build a
   normalized frequency histogram.
3. Train a probabilistic classifier on training histograms; for a new image
   show the histogram and ``p(class) = …`` for the top class.

Each class folder is split once, with a fixed seed, into training images
(codebook and classifier) and held-out test images, and the reported
accuracy is measured on the test images: an accuracy on the training images
says little (on the course's Caltech subset, 84% on training images was 55%
on unseen ones).

Defaults were measured on that subset (butterfly, dalmatian, kangaroo; 124
held-out images; K = 30 as the slide asks). The classifier is logistic
regression on square-rooted histograms (the Hellinger mapping, standard for
histograms): 69% held-out with SIFT, against 61% for the standardised
histograms used before, whose probabilities were also overconfident (0.78 on
average for a 61% hit rate). SIFT beat ORB in every setting tried (ORB: 58%);
K-Means also suits SIFT's numeric descriptors better than ORB's bit strings.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import math
from pathlib import Path
import time
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
from sklearn.preprocessing import FunctionTransformer

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .descriptors import ACCEPTED_MODES, DescriptorParams, descriptor_section, extract_descriptors
from .param_dialog import FolderChooser, IntField, ParamDialog, ProgressWindow, parse_int_fields


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
WORDS_PER_CLASS = 10
PATCH_SIZE = 32
# Cap training images per class so a large class folder stays interactive.
MAX_TRAIN_PER_CLASS = 100
DEFAULT_DETECTOR = DescriptorParams(method="sift")
TEST_SHARE = 0.3
# Measured seconds per image (extraction plus word assignment), for the estimate.
SECONDS_PER_IMAGE = {"orb": 0.1, "sift": 0.25}

SPLIT_FIELDS = (IntField("Test share", 30, 80, "% of each class held out", minimum=10),)

# In-memory model shared by the three Set 7 tools (rebuilt by CodebookTool).
_MODEL: "BowModel | None" = None


class BuildCancelled(Exception):
    """The user pressed Cancel while the codebook was being built."""


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
    classifier: Pipeline  # sqrt (Hellinger) + LogisticRegression
    train_accuracy: float
    test_accuracy: float | None = None  # on held-out images; None without any
    # Per class: (correct, total) over the held-out images.
    test_results: dict[str, tuple[int, int]] = field(default_factory=dict)
    train_files: frozenset[Path] = frozenset()
    test_files: frozenset[Path] = frozenset()
    skipped: tuple[tuple[Path, str], ...] = ()

    def was_trained_on(self, path: Path | None) -> bool:
        return path is not None and path.resolve() in self.train_files


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
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.startswith(".")
        )
        if images:
            classes.append((child.name, images))
    if len(classes) < 2:
        raise ValueError(
            "Need at least two class subfolders with images "
            "(e.g. butterfly/, kangaroo/, dalmatian/)."
        )
    return classes


def split_class(
    paths: list[Path], test_share: float, max_train: int, random_state: int = 0
) -> tuple[list[Path], list[Path]]:
    """``(train, test)`` paths of one class: a seeded shuffle, then ``test_share`` held out.

    At least one image always stays for training; a class with a single image
    has no test images. Training images beyond ``max_train`` are left unused.
    """
    order = np.random.default_rng(random_state).permutation(len(paths))
    shuffled = [paths[i] for i in order]
    n_test = min(len(paths) - 1, math.ceil(len(paths) * test_share)) if test_share > 0 else 0
    test, train = shuffled[:n_test], shuffled[n_test:]
    return sorted(train[:max_train]), sorted(test)


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


def _load(path: Path) -> Image.Image:
    with Image.open(path) as opened:
        image = opened.copy()
    return image if image.mode in ACCEPTED_MODES else image.convert("RGB")


def _new_classifier(random_state: int) -> Pipeline:
    return Pipeline(
        [
            ("hellinger", FunctionTransformer(np.sqrt)),
            ("clf", LogisticRegression(C=10.0, max_iter=5000, random_state=random_state)),
        ]
    )


def build_codebook(
    dataset_root: Path,
    *,
    detector: DescriptorParams = DEFAULT_DETECTOR,
    max_per_class: int = MAX_TRAIN_PER_CLASS,
    test_share: float = TEST_SHARE,
    random_state: int = 0,
    progress: Callable[[int, int, str], bool] | None = None,
) -> BowModel:
    """Cluster training descriptors, visualize patches, train and test a classifier.

    ``progress(done, total, text)`` is called before each image; returning
    False stops the build with :class:`BuildCancelled`. Unreadable images and
    images without keypoints are skipped and listed in ``model.skipped``.
    """
    classes = list_class_folders(dataset_root)
    class_names = tuple(name for name, _ in classes)
    n_clusters = WORDS_PER_CLASS * len(class_names)
    splits = [split_class(paths, test_share, max_per_class, random_state) for _, paths in classes]
    total = sum(len(train) + len(test) for train, test in splits)
    done = 0

    def step(text: str) -> None:
        nonlocal done
        if progress is not None and not progress(done, total, text):
            raise BuildCancelled
        done += 1

    skipped: list[tuple[Path, str]] = []

    def describe(path: Path) -> tuple[np.ndarray, np.ndarray, Image.Image] | None:
        try:
            image = _load(path)
        except OSError:
            skipped.append((path, "not a readable image"))
            return None
        try:
            keypoints, descriptors = _extract(image, detector)
        except ValueError:
            skipped.append((path, "no keypoints found"))
            return None
        return keypoints, np.asarray(descriptors, dtype=np.float64), image

    # First pass: training descriptors (and a patch source) for clustering.
    per_image: list[tuple[int, np.ndarray]] = []  # (label_index, descriptors)
    patch_sources: list[tuple[Image.Image, float, float]] = []
    for label, ((name, _), (train, _)) in enumerate(zip(classes, splits)):
        for path in train:
            step(f"Training image: {name}/{path.name}")
            found = describe(path)
            if found is None:
                continue
            keypoints, desc, image = found
            patch_sources.extend((image, float(r), float(c)) for r, c in keypoints)
            per_image.append((label, desc))

    if not per_image:
        raise ValueError(
            "No descriptors could be extracted from the training folder.\n\n"
            "Check that the images are textured enough for ORB/SIFT."
        )
    if len({label for label, _ in per_image}) < 2:
        raise ValueError("At least two classes need training images with keypoints.")

    stacked = np.vstack([desc for _, desc in per_image])
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

    # Illustrative patch = the training patch whose descriptor is closest to each centre.
    words = nearest_words(stacked, centroids)
    patches: list[Image.Image] = []
    for word in range(n_clusters):
        members = np.where(words == word)[0]
        if len(members) == 0:
            patches.append(Image.new("L", (PATCH_SIZE, PATCH_SIZE), 0))
            continue
        dist = np.linalg.norm(stacked[members] - centroids[word], axis=1)
        image, row, col = patch_sources[int(members[int(np.argmin(dist))])]
        patches.append(_crop_patch(image, row, col))

    codebook = Codebook(
        centroids=centroids,
        class_names=class_names,
        patches=tuple(patches),
        detector=detector,
    )

    x = np.vstack([histogram_from_descriptors(desc, codebook) for _, desc in per_image])
    y = np.asarray([label for label, _ in per_image], dtype=np.int64)
    classifier = _new_classifier(random_state)
    classifier.fit(x, y)
    train_accuracy = float(classifier.score(x, y))

    # Held-out test images: the accuracy worth reporting.
    results: dict[str, list[int]] = {name: [0, 0] for name in class_names}
    for label, ((name, _), (_, test)) in enumerate(zip(classes, splits)):
        for path in test:
            step(f"Test image: {name}/{path.name}")
            found = describe(path)
            if found is None:
                continue
            hist = histogram_from_descriptors(found[1], codebook)
            predicted = int(classifier.predict(hist.reshape(1, -1))[0])
            results[name][0] += predicted == label
            results[name][1] += 1
    tested = sum(t for _, t in results.values())
    test_accuracy = sum(c for c, _ in results.values()) / tested if tested else None

    model = BowModel(
        codebook=codebook,
        classifier=classifier,
        train_accuracy=train_accuracy,
        test_accuracy=test_accuracy,
        test_results={name: (c, t) for name, (c, t) in results.items() if t},
        train_files=frozenset(p.resolve() for train, _ in splits for p in train),
        test_files=frozenset(p.resolve() for _, test in splits for p in test),
        skipped=tuple(skipped),
    )
    set_model(model)
    return model


def nearest_words(descriptors: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Index of the closest centroid (Euclidean) for each descriptor row.

    Uses ||a - c||² = ||a||² - 2 a·c + ||c||², which needs an (N, K) array
    instead of the (N, K, D) differences (about 90 MB for one SIFT image).
    """
    squared = (
        np.einsum("nd,nd->n", descriptors, descriptors)[:, None]
        - 2.0 * descriptors @ centroids.T
        + np.einsum("kd,kd->k", centroids, centroids)[None, :]
    )
    return np.argmin(squared, axis=1)


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
    assignments = nearest_words(desc, codebook.centroids)
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


def true_class(path: Path | None, model: BowModel) -> str | None:
    """The class an image belongs to, when it sits in one of the model's class folders."""
    if path is None:
        return None
    name = path.parent.name
    return name if name in model.codebook.class_names else None


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


def _accuracy_text(model: BowModel) -> str:
    if model.test_accuracy is None:
        return f"no held-out images (train {model.train_accuracy:.0%})"
    tested = sum(t for _, t in model.test_results.values())
    return f"{model.test_accuracy:.0%} on {tested} held-out images"


class _NeedsModel(ForensicsTool):
    """Shared availability rules of the histogram and classify tools."""

    ACCEPTED_MODES = ACCEPTED_MODES

    def is_available(self, document: ImageDocument) -> bool:
        if get_model() is None:
            return False
        if document.current is None:
            return True
        # Figures (a BoW sheet, eigenfaces…) are refused by the app-wide rule in
        # ToolRegistry.blocked_by_figure, so only the image mode is checked here.
        return document.current.mode in self.ACCEPTED_MODES

    def unavailable_message(self, document: ImageDocument) -> str:
        if get_model() is None:
            return (
                "No codebook yet.\n\n"
                "Run BoW codebook first on a class-folder dataset."
            )
        current = document.current
        mode = current.mode if current is not None else "unknown"
        return (
            f"This image mode is not supported (mode {mode!r}).\n\n"
            "Open an L or RGB image."
        )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CodebookRequest:
    folder: Path
    detector: DescriptorParams
    test_share: float


class CodebookTool(ForensicsTool):
    tool_id = "bow_codebook"
    produces_figure = True
    title = "BoW codebook"
    category = "Recognition"
    description = (
        "Build a visual-word codebook with MiniBatchKMeans "
        "(K = 10 × classes) and train a classifier (Set 7.1)."
    )
    requires_image = False

    def __init__(self) -> None:
        self._folder: Path | None = None

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        request = self._ask_folder(parent)
        if request is None:
            return None
        window = ProgressWindow(parent, title="Building the codebook…", total=1) if parent else None

        def progress(done: int, total: int, text: str) -> bool:
            if window is None:
                return True
            return window.step(done, f"{done + 1} of {total} · {text}", total)

        started = time.perf_counter()
        try:
            model = build_codebook(
                request.folder,
                detector=request.detector,
                test_share=request.test_share,
                progress=progress,
            )
        except BuildCancelled:
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
        elapsed = time.perf_counter() - started

        sheet = render_visual_words(model.codebook.patches)
        names = ", ".join(model.codebook.class_names)
        details: dict[str, object] = {
            "Operation": self.title,
            "Function": "sklearn.cluster.MiniBatchKMeans",
            "Classes": names,
            "K": model.codebook.size,
            "Detector": model.codebook.detector.method.upper(),
            "Images": f"{len(model.train_files)} train · {len(model.test_files)} held out",
            "Classifier": "logistic regression on √histograms",
            "Test accuracy": _accuracy_text(model),
            "Train accuracy": f"{model.train_accuracy:.0%} (on the images it learned from)",
        }
        for name, (correct, total) in model.test_results.items():
            details[f"  {name}"] = f"{correct}/{total} ({correct / total:.0%})"
        if model.skipped:
            details["Skipped"] = "; ".join(f"{p.name} ({why})" for p, why in model.skipped)
        details["Time"] = f"{elapsed:.0f} s"
        return ToolResult(
            image=sheet,
            message=(
                f"Built codebook with {model.codebook.size} visual words "
                f"({names}); test accuracy {_accuracy_text(model)}."
            ),
            details=details,
        )

    def _ask_folder(self, parent: tk.Misc | None) -> CodebookRequest | None:
        dialog: ParamDialog[CodebookRequest] = ParamDialog(
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
            initial=self._folder,
        )
        read_detector = descriptor_section(dialog, method=DEFAULT_DETECTOR.method)
        (share_var,) = dialog.int_fields(dialog.section("Evaluation"), SPLIT_FIELDS)

        def validate() -> tuple[CodebookRequest, str]:
            folder = chooser.require()
            classes = list_class_folders(folder)
            detector, _ = read_detector()
            (share,) = parse_int_fields(SPLIT_FIELDS, (share_var.get(),))
            images = sum(len(paths) for _, paths in classes)
            seconds = images * SECONDS_PER_IMAGE[detector.method]
            readout = (
                f"{describe_dataset(folder)} · {detector.method.upper()} · "
                f"about {max(1, round(seconds))} s"
            )
            return CodebookRequest(folder, detector, share / 100), readout

        request = dialog.show(validate)
        if request is not None:
            self._folder = request.folder
        return request


class BowHistogramTool(_NeedsModel):
    tool_id = "bow_histogram"
    produces_figure = True
    title = "BoW histogram"
    category = "Recognition"
    description = (
        "Assign the working image's features to visual words and plot the "
        "normalized histogram (Set 7.2)."
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
        details: dict[str, object] = {
            "Operation": self.title,
            "Codebook K": model.codebook.size,
            "Non-zero bins": int(np.count_nonzero(hist)),
            "Peak word": int(np.argmax(hist)),
            "Peak frequency": f"{float(hist.max()):.3f}",
        }
        if model.was_trained_on(document.path):
            details["Note"] = "This image was used to build the codebook (a training image)."
        return ToolResult(
            image=render_histogram_sheet(source, hist),
            message=f"Built a {model.codebook.size}-bin visual-word histogram.",
            details=details,
        )


class BowClassifyTool(_NeedsModel):
    tool_id = "bow_classify"
    produces_figure = True
    title = "BoW classify"
    category = "Recognition"
    description = (
        "Classify the working image with the trained BoW model and show "
        "p(class) with its histogram (Set 7.3)."
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

        truth = true_class(document.path, model)
        details: dict[str, object] = {
            "Operation": self.title,
            "Predicted": label,
            "Probability": f"{prob:.3f}",
        }
        if truth is not None:
            details["True class"] = f"{truth} ({'correct' if truth == label else 'wrong'})"
        details["Model accuracy"] = _accuracy_text(model)
        for name, p in zip(model.codebook.class_names, probs, strict=True):
            details[f"p({name})"] = f"{float(p):.3f}"
        message = f"p({label}) = {prob:.2f}"
        if model.was_trained_on(document.path):
            details["Note"] = (
                "This image was used to train the classifier, so its probability "
                "is optimistic. Use a held-out image or a new one to judge the model."
            )
            message += " (a training image)"
        elif truth is not None:
            message += " — correct" if truth == label else f" — wrong, this is a {truth}"

        return ToolResult(
            image=render_classification(source, label, prob, hist),
            message=message,
            details=details,
        )
