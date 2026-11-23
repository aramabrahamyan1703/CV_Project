"""Extract CNN features of the test set, then project them with t-SNE and UMAP (Set 8.3).

*CNN feature extraction* runs CIFAR-10's test images through a model saved by
*CNN training* and keeps what the network computes just before its
classifier: 128 numbers per image for the small CNN, 2048 for InceptionV3.
They are saved with the true labels to ``data/cifar10/features/<model>_test.npz``
so the projections below load in a second instead of re-running the network
(InceptionV3 takes ~50 s for the 10,000 images on an M3 Pro, the small CNN ~1 s).

*Feature projection* loads such a file and maps the vectors to 2-D twice:

* t-SNE (scikit-learn) keeps each point's nearest neighbours close; distances
  between clusters and cluster sizes mean little.
* UMAP (umap-learn) does similarly from a neighbour graph and tends to keep
  more of the global layout.

Both first reduce the vectors to 50 dimensions with PCA, as scikit-learn
recommends for t-SNE: it removes noise and makes both much faster. Seeds are
fixed, so the same file and settings give the same picture. The plots colour
each point by its true class; as a number to go with the picture, the details
give each layout's leave-one-out 5-nearest-neighbour accuracy, i.e. how often
a point's closest neighbours on the plot share its class (in the original
space too, for comparison).

Measured on an M3 Pro for InceptionV3 features: 2,000 points take ~4 s
(t-SNE) and ~6 s (UMAP, its first run also compiles), 10,000 points ~17 s and
~22 s. Neither can be interrupted midway; Cancel discards the result.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import threading
import time
import tkinter as tk
from tkinter import messagebox
from typing import Any
import warnings

import numpy as np
from matplotlib.figure import Figure
from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .cnn_classify import MODEL_TYPES, default_weights, describe_weights, model_name
from .cnn_data import (
    FEATURE_LAYERS,
    FeatureSet,
    features_path,
    load_features,
    load_passport,
    save_features,
)
from .cnn_training import INSTALL_HINT, best_device, cifar_downloaded, torch_available
from .face_recognition import to_figure_image
from .param_dialog import FileChooser, IntField, ParamDialog, ProgressWindow, parse_int_fields


FEATURE_TYPES = [("Feature files", "*.npz"), ("All files", "*.*")]
TEST_IMAGES = 10_000
EXTRACT_FIELDS = (IntField("Test images", TEST_IMAGES, TEST_IMAGES, "first N of the 10,000", minimum=100),)
PROJECT_FIELDS = (
    IntField("Points", 5_000, TEST_IMAGES, "random subset to project", minimum=50),
    IntField("Perplexity", 30, 100, "t-SNE: neighbours per point", minimum=5),
    IntField("UMAP neighbours", 15, 200, "UMAP: local neighbourhood size", minimum=2),
)
PCA_DIMS = 50
SEED = 0
KNN = 5

Progress = Callable[[str, int, int], None]
Cancelled = Callable[[], bool]

# The feature file extracted last, offered first by the projection tool.
_last_features: Path | None = None


def umap_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("umap") is not None


# --------------------------------------------------------------------------
# Background work with a progress window
# --------------------------------------------------------------------------


class Worker:
    """Runs ``work(progress, cancelled)`` on a thread; the UI thread polls it."""

    def __init__(self, work: Callable[[Progress, Cancelled], Any]) -> None:
        self.result: Any = None
        self.error: BaseException | None = None
        self.stage = "Starting…"
        self.done = 0
        self.total = 0
        self._stop = False
        self._work = work
        self._thread = threading.Thread(target=self._run, name="cnn-features", daemon=True)

    def _run(self) -> None:
        try:
            self.result = self._work(self.progress, lambda: self._stop)
        except BaseException as error:  # reported on the UI thread
            self.error = error

    def progress(self, stage: str, done: int, total: int) -> None:
        self.stage, self.done, self.total = stage, done, total

    def status(self) -> tuple[int, int, str]:
        if self.total == 0:
            return 0, 1, self.stage
        return self.done, self.total, f"{self.stage} · {self.done:,}/{self.total:,}"

    def start(self) -> None:
        self._thread.start()

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    def request_stop(self) -> None:
        self._stop = True

    @property
    def stop_requested(self) -> bool:
        return self._stop


def run_with_progress(parent: tk.Misc, title: str, worker: Worker) -> bool:
    """Show a progress window until ``worker`` ends; False if the user cancelled."""
    window = ProgressWindow(parent, title=title, total=1)
    worker.start()
    try:
        while worker.is_alive():
            done, total, text = worker.status()
            if not window.cancelled:
                if not window.step(done, text, total):
                    worker.request_stop()
            else:
                worker.request_stop()
                break  # projections cannot stop midway: leave the thread, drop its result
            worker.join(0.1)
    finally:
        window.close()
    return not worker.stop_requested


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------


def extract_test_features(
    weights: Path,
    count: int,
    device: str,
    progress: Progress | None = None,
    cancelled: Cancelled | None = None,
    load_test: Callable[[], Any] | None = None,
    layer: str = "features",
) -> FeatureSet | None:
    """Features at ``layer`` of the first ``count`` test images under the model in ``weights``;
    None if cancelled."""
    from . import cnn

    if progress is not None:
        progress("Loading the model…", 0, 0)
    model, passport = cnn.load_model(weights, device)
    if load_test is not None:
        test = load_test()
    else:
        if progress is not None:
            progress("Loading CIFAR-10" + ("…" if cifar_downloaded() else " (downloading 170 MB once)…"), 0, 0)
        _, test = cnn.load_cifar10()
    test = test.subset(np.arange(min(count, len(test))))
    features = cnn.extract_features(
        model, passport.preprocessing, test.images, 64, device, progress, cancelled, "Extracting features", layer
    )
    if features is None:
        return None
    return FeatureSet(features, test.labels, passport.classes, passport.architecture, str(weights), layer=layer)


def layer_for(architecture: str, use_head: bool) -> str:
    """The trained head layer when asked for and the model has one, else the classifier's input."""
    return "head" if use_head and "head" in FEATURE_LAYERS.get(architecture, {}) else "features"


def layer_label(feature_set: FeatureSet) -> str:
    label, _ = FEATURE_LAYERS.get(feature_set.architecture, {}).get(feature_set.layer, (feature_set.layer, 0))
    return label


# --------------------------------------------------------------------------
# Projection
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ProjectParams:
    features: Path
    points: int
    perplexity: int
    neighbours: int


@dataclass(frozen=True)
class Projection:
    name: str  # "t-SNE" or "UMAP"
    points: np.ndarray  # (N, 2)
    seconds: float
    knn_accuracy: float  # leave-one-out k-NN accuracy of the true classes in 2-D


@dataclass(frozen=True)
class ProjectionResult:
    labels: np.ndarray  # (N,) of the projected points
    indices: np.ndarray  # (N,) their rows in the feature file
    reduced_dims: int  # after PCA
    explained: float  # share of the variance PCA kept
    knn_original: float  # the same k-NN accuracy in the full feature space
    projections: tuple[Projection, ...]


def choose_points(count: int, total: int, seed: int = SEED) -> np.ndarray:
    """Sorted indices of a seeded random subset of ``count`` of ``total`` rows (all if more)."""
    if count >= total:
        return np.arange(total)
    return np.sort(np.random.default_rng(seed).choice(total, size=count, replace=False))


def knn_accuracy(points: np.ndarray, labels: np.ndarray, k: int = KNN) -> float:
    """Leave-one-out: share of points whose k nearest other points vote for their class."""
    from sklearn.neighbors import NearestNeighbors

    k = min(k, len(points) - 1)
    if k < 1:
        return float("nan")
    _, neighbours = NearestNeighbors(n_neighbors=k + 1).fit(points).kneighbors(points)
    votes = labels[neighbours[:, 1:]]  # drop each point itself
    predicted = np.array([np.bincount(row).argmax() for row in votes])
    return float(np.mean(predicted == labels))


def pca_reduce(features: np.ndarray, dims: int = PCA_DIMS, seed: int = SEED) -> tuple[np.ndarray, float]:
    """The first ``dims`` principal components (all if there are fewer) and the variance share kept."""
    from sklearn.decomposition import PCA

    dims = min(dims, features.shape[1], len(features))
    if dims == features.shape[1]:
        return features.astype(np.float32), 1.0
    pca = PCA(dims, random_state=seed)
    reduced = pca.fit_transform(features)
    return reduced.astype(np.float32), float(pca.explained_variance_ratio_.sum())


def tsne(points: np.ndarray, perplexity: int, seed: int = SEED) -> np.ndarray:
    from sklearn.manifold import TSNE

    perplexity = min(perplexity, (len(points) - 1) // 3)  # scikit-learn needs perplexity < n
    return TSNE(2, perplexity=max(perplexity, 1), init="pca", random_state=seed).fit_transform(points)


def umap_2d(points: np.ndarray, neighbours: int, seed: int = SEED) -> np.ndarray:
    import umap

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # "n_jobs overridden by random_state": seeded runs are serial
        reducer = umap.UMAP(n_neighbors=min(neighbours, len(points) - 1), min_dist=0.1, random_state=seed)
        return reducer.fit_transform(points)


def project(
    feature_set: FeatureSet, params: ProjectParams, progress: Progress | None = None
) -> ProjectionResult:
    """PCA, then t-SNE and UMAP of a seeded subset of ``feature_set``."""
    def stage(text: str) -> None:
        if progress is not None:
            progress(text, 0, 0)

    indices = choose_points(params.points, len(feature_set.labels))
    features, labels = feature_set.features[indices], feature_set.labels[indices]
    stage(f"PCA of {len(indices):,} vectors to {PCA_DIMS} dimensions…")
    reduced, explained = pca_reduce(features)
    knn_original = knn_accuracy(features, labels)
    projections = []
    for name, run in (("t-SNE", lambda: tsne(reduced, params.perplexity)), ("UMAP", lambda: umap_2d(reduced, params.neighbours))):
        stage(f"{name} of {len(indices):,} points…")
        started = time.perf_counter()
        points = np.asarray(run(), dtype=np.float32)
        projections.append(Projection(name, points, time.perf_counter() - started, knn_accuracy(points, labels)))
    return ProjectionResult(labels, indices, reduced.shape[1], explained, knn_original, tuple(projections))


def render_projections(result: ProjectionResult, classes: tuple[str, ...], title: str = "") -> Image.Image:
    """The projections side by side, each point coloured by its true class."""
    count = len(result.projections)
    figure = Figure(figsize=(6 * count + 1.6, 6), dpi=110, layout="constrained")
    axes = np.atleast_1d(figure.subplots(1, count))
    colours = [f"C{i % 10}" for i in range(len(classes))]
    size = 6 if len(result.labels) <= 3000 else 3
    for axis, projection in zip(axes, result.projections):
        for index, name in enumerate(classes):
            mask = result.labels == index
            if mask.any():
                axis.scatter(*projection.points[mask].T, s=size, alpha=0.7, color=colours[index], label=name, linewidths=0)
        axis.set_title(f"{projection.name}  ·  {KNN}-NN accuracy {projection.knn_accuracy:.1%}", fontsize=11)
        axis.set_xlabel(f"{projection.name} 1")
        axis.set_ylabel(f"{projection.name} 2")
        axis.set_xticks([])
        axis.set_yticks([])
    handles, names = axes[0].get_legend_handles_labels()
    figure.legend(handles, names, loc="outside right center", markerscale=3, fontsize=9, frameon=False)
    if title:
        figure.suptitle(title, fontsize=12)
    return to_figure_image(figure)


def describe_features(path: Path) -> str:
    try:
        feature_set = load_features(path)
    except ValueError as error:
        return str(error)
    count, dims = feature_set.features.shape
    return f"{feature_set.architecture}, {layer_label(feature_set)} · {count:,} {feature_set.split} vectors · {len(feature_set.classes)} classes"


def default_features() -> Path | None:
    for candidate in (
        _last_features, features_path("inception_v3", layer="head"), features_path("inception_v3"), features_path("small_cnn")
    ):
        if candidate is not None and candidate.is_file():
            return candidate
    return None


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------


class CnnFeatureExtractionTool(ForensicsTool):
    tool_id = "cnn_features"
    requires_image = False
    title = "CNN feature extraction"
    category = "Classification"
    description = "Run the CIFAR-10 test set through a trained CNN and save its feature vectors (Set 8.3)."

    def __init__(self) -> None:
        self._weights: Path | None = None

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        global _last_features
        if not torch_available():
            messagebox.showinfo(self.title, INSTALL_HINT, parent=parent)
            return None
        asked = self._ask_params(parent)
        if asked is None:
            return None
        weights, count, layer = asked
        device = best_device()
        started = time.perf_counter()
        worker = Worker(
            lambda progress, cancelled: extract_test_features(weights, count, device, progress, cancelled, layer=layer)
        )
        finished = run_with_progress(parent, "Extracting features…", worker)
        if worker.error is not None:
            messagebox.showinfo(self.title, f"Extraction failed: {worker.error}", parent=parent)
            return None
        if not finished or worker.result is None:
            return None
        feature_set: FeatureSet = worker.result
        path = save_features(feature_set, features_path(feature_set.architecture, layer=feature_set.layer))
        _last_features = path
        count, dims = feature_set.features.shape
        per_class = np.bincount(feature_set.labels, minlength=len(feature_set.classes))
        return ToolResult(
            message=f"Saved {count:,} feature vectors of {dims} numbers to features/{path.name}.",
            details={
                "Operation": self.title,
                "Model": str(weights),
                "Images": f"the first {count:,} CIFAR-10 test images",
                "Feature vector": f"{dims} numbers: {layer_label(feature_set)}",
                "Per class": ", ".join(f"{name} {n}" for name, n in zip(feature_set.classes, per_class)),
                "Device": device,
                "Saved to": str(path),
                "Size": f"{path.stat().st_size / 1e6:.1f} MB",
                "Time": f"{time.perf_counter() - started:.0f} s",
            },
        )

    def _ask_params(self, parent: tk.Misc | None) -> tuple[Path, int, str] | None:
        dialog: ParamDialog[tuple[Path, int, str]] = ParamDialog(
            parent,
            title=self.title,
            intro="Compute the feature vector of each CIFAR-10 test image with a model saved by "
            "CNN training and save them with their labels, for Feature projection.",
            action="Extract",
        )
        chooser = FileChooser(
            dialog,
            dialog.section("Model"),
            prompt="Choose a model.pt saved by CNN training.",
            filetypes=MODEL_TYPES,
            describe=describe_weights,
            initial=self._weights or default_weights(),
        )
        frame = dialog.section("Layer")
        head_var = dialog.checkbox(
            frame, "Trained head layer", default=True,
            hint="InceptionV3: 1024 numbers learnt on CIFAR-10 (untick: 2048-d ImageNet backbone)",
        )
        variables = dialog.int_fields(dialog.section("Images"), EXTRACT_FIELDS)

        def validate() -> tuple[tuple[Path, int, str], str]:
            weights = chooser.require()
            passport = load_passport(weights)
            (count,) = parse_int_fields(EXTRACT_FIELDS, (variables[0].get(),))
            layer = layer_for(passport.architecture, bool(head_var.get()))
            label, dims = FEATURE_LAYERS.get(passport.architecture, {}).get(layer, (layer, passport.feature_dim))
            output = features_path(passport.architecture, layer=layer)
            replaces = " (replaces it)" if output.is_file() else ""
            readout = f"{model_name(passport)}: {label}, {dims} numbers per image · saves features/{output.name}{replaces}"
            if not cifar_downloaded():
                readout += " · downloads CIFAR-10 (170 MB) first"
            return (weights, count, layer), readout

        asked = dialog.show(validate)
        if asked is not None:
            self._weights = asked[0]
        return asked


class FeatureProjectionTool(ForensicsTool):
    tool_id = "feature_projection"
    produces_figure = True
    requires_image = False
    title = "Feature projection (t-SNE, UMAP)"
    category = "Classification"
    description = "Load saved CNN feature vectors and project them to 2-D with t-SNE and UMAP (Set 8.3)."

    def __init__(self) -> None:
        self._features: Path | None = None

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        if not umap_available():
            messagebox.showinfo(
                self.title, "UMAP needs umap-learn.\n\nInstall it with  python -m pip install -r requirements.txt", parent=parent
            )
            return None
        params = self._ask_params(parent)
        if params is None:
            return None
        try:
            feature_set = load_features(params.features)
        except ValueError as error:
            messagebox.showinfo(self.title, str(error), parent=parent)
            return None
        worker = Worker(lambda progress, _cancelled: project(feature_set, params, progress))
        if not run_with_progress(parent, "Projecting features…", worker):
            return None
        if worker.error is not None:
            messagebox.showinfo(self.title, f"Projection failed: {worker.error}", parent=parent)
            return None
        return self._result(params, feature_set, worker.result)

    def _result(self, params: ProjectParams, feature_set: FeatureSet, result: ProjectionResult) -> ToolResult:
        count, dims = len(result.labels), feature_set.features.shape[1]
        title = (
            f"{feature_set.architecture}, {layer_label(feature_set)}: {count:,} CIFAR-10 {feature_set.split} images "
            f"({dims}-d → PCA {result.reduced_dims}-d → 2-D)"
        )
        tsne_result, umap_result = result.projections
        details: dict[str, object] = {
            "Operation": self.title,
            "Features": str(params.features),
            "Points": f"{count:,} of {len(feature_set.labels):,} (seeded random subset)",
            "PCA": f"{dims} → {result.reduced_dims} dimensions, {result.explained:.1%} of the variance kept",
            "t-SNE": f"perplexity {min(params.perplexity, (count - 1) // 3)}, {tsne_result.seconds:.1f} s",
            "UMAP": f"{min(params.neighbours, count - 1)} neighbours, min_dist 0.1, {umap_result.seconds:.1f} s",
            f"{KNN}-NN accuracy, {dims}-d": f"{result.knn_original:.1%}",
            f"{KNN}-NN accuracy, t-SNE": f"{tsne_result.knn_accuracy:.1%}",
            f"{KNN}-NN accuracy, UMAP": f"{umap_result.knn_accuracy:.1%}",
            "Seed": str(SEED),
        }
        return ToolResult(
            image=render_projections(result, feature_set.classes, title),
            message=f"Projected {count:,} vectors: {KNN}-NN accuracy t-SNE {tsne_result.knn_accuracy:.1%}, "
            f"UMAP {umap_result.knn_accuracy:.1%} ({dims}-d: {result.knn_original:.1%}).",
            details=details,
        )

    def _ask_params(self, parent: tk.Misc | None) -> ProjectParams | None:
        dialog: ParamDialog[ProjectParams] = ParamDialog(
            parent,
            title=self.title,
            intro="Map saved feature vectors (from CNN feature extraction) to 2-D with t-SNE and "
            "UMAP, after PCA to 50 dimensions, and colour each point by its class.",
            action="Project",
        )
        chooser = FileChooser(
            dialog,
            dialog.section("Features"),
            prompt="Choose a feature file saved by CNN feature extraction.",
            filetypes=FEATURE_TYPES,
            describe=describe_features,
            initial=self._features or default_features(),
        )
        variables = dialog.int_fields(dialog.section("Projection"), PROJECT_FIELDS)
        totals: dict[Path, int] = {}  # vectors per file, read once (validate runs on every keystroke)

        def validate() -> tuple[ProjectParams, str]:
            path = chooser.require()
            if path not in totals:
                totals[path] = len(load_features(path).labels)
            total = totals[path]
            points, perplexity, neighbours = parse_int_fields(PROJECT_FIELDS, tuple(v.get() for v in variables))
            used = min(points, total)
            seconds = used / 10_000 * 40  # t-SNE + UMAP, measured on an M3 Pro
            readout = f"{used:,} of {total:,} vectors · " + ("under a minute" if seconds < 60 else f"about {seconds / 60:.0f} min")
            return ProjectParams(path, points, perplexity, neighbours), readout

        params = dialog.show(validate)
        if params is not None:
            self._features = params.features
        return params
