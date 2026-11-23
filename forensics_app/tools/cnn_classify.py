"""Classify the current image with a saved CNN and show its passport (Set 8.2).

*CNN classification* loads a model saved by *CNN training* (``model.pt`` with
its ``passport.json``), prepares the working image exactly as the passport
says (RGB, squashed to the dataset's 32×32, then the network's own resizing
and scaling, the same code that prepared the training batches) and shows the
most likely classes beside the image as the network saw it, above the
model's training curves. The details list the whole passport.

The network only knows CIFAR-10's ten classes and 32-pixel images: a photo of
anything else still gets one of the ten names, and a large photo loses most of
its detail on the way down to 32×32, as the middle panel shows.

The loaded model stays in memory until another file is chosen or the file
changes, so classifying several images does not reload InceptionV3's 96 MB.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tkinter as tk
from tkinter import messagebox

import numpy as np
from matplotlib.figure import Figure
from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .cnn_data import (
    ARCHITECTURES,
    INCEPTION_V3,
    MODELS_DIR,
    SMALL_CNN,
    WEIGHTS_FILE,
    Passport,
    load_passport,
    to_dataset_image,
)
from .cnn_training import INSTALL_HINT, MODEL_CHOICES, best_device, last_model, plot_history, torch_available
from .face_recognition import to_figure_image
from .param_dialog import FileChooser, IntField, ParamDialog, parse_int_fields


MODEL_TYPES = [("CNN weights", "*.pt"), ("All files", "*.*")]
TOP_FIELDS = (IntField("Top classes", 5, 10, "most likely classes to list"),)

_loaded: dict[str, Any] = {}  # key -> (model, passport) of the last model loaded


@dataclass(frozen=True)
class ClassifyParams:
    weights: Path
    top: int


@dataclass(frozen=True)
class Classification:
    ranking: list[tuple[str, float]]  # (class, probability), most likely first, all classes
    seen: np.ndarray  # the image as the network got it: (32, 32, 3) uint8

    def top(self, count: int) -> list[tuple[str, float]]:
        return self.ranking[:count]


def model_name(passport: Passport) -> str:
    for label, key in MODEL_CHOICES.items():
        if key == passport.architecture:
            return label
    return ARCHITECTURES.get(passport.architecture, passport.architecture)


def default_weights() -> Path | None:
    """The model saved last in this session, else InceptionV3's, else the small CNN's."""
    folder = last_model()
    candidates = ([folder] if folder is not None else []) + [MODELS_DIR / INCEPTION_V3, MODELS_DIR / SMALL_CNN]
    for candidate in candidates:
        if (candidate / WEIGHTS_FILE).is_file():
            return candidate / WEIGHTS_FILE
    return None


def describe_weights(path: Path) -> str:
    """One line about a ``model.pt`` from its passport, or why it cannot be used."""
    try:
        passport = load_passport(path)
    except ValueError as error:
        return str(error)
    parts = [model_name(passport), f"{len(passport.classes)} classes"]
    if "accuracy" in passport.test:
        parts.append(f"test accuracy {passport.test['accuracy']:.1%}")
    if passport.created:
        parts.append(f"trained {passport.created.replace('T', ' ')}")
    return " · ".join(parts)


def passport_details(passport: Passport) -> dict[str, str]:
    """The passport as label -> text rows (Set 8.2: "display model passport")."""
    dataset = passport.dataset
    rows = {
        "Model": f"{model_name(passport)} ({passport.architecture}, {passport.feature_dim}-d features)",
        "Dataset": str(dataset.get("name", "unknown")),
        "Samples": " · ".join(
            f"{dataset[key]:,} {label}"
            for key, label in (("train_samples", "train"), ("val_samples", "validation"), ("test_samples", "test"))
            if key in dataset
        ) or "not recorded",
        "Classes": ", ".join(passport.classes),
        "Preprocessing": passport.preprocessing.describe(),
    }
    for number, stage in enumerate(passport.training.get("stages", []), start=1):
        name = "head only, backbone frozen" if stage.get("frozen_backbone") else "all layers"
        optimizer = f"{stage.get('optimizer', '?').upper()} lr {stage.get('learning_rate', '?')}"
        if "momentum" in stage:
            optimizer += f", momentum {stage['momentum']}"
        rows[f"Training {number}"] = (
            f"{stage.get('epochs_done', '?')} of {stage.get('epochs', '?')} epochs, {name}; {optimizer}; "
            f"batch {stage.get('batch_size', '?')}; {stage.get('device', '?')}, {stage.get('seconds', 0) / 60:.1f} min"
        )
    history = passport.history
    if history.get("val_accuracy"):
        rows["History"] = (
            f"{len(history['val_accuracy'])} epochs; last: train {history['accuracy'][-1]:.1%}, "
            f"validation {history['val_accuracy'][-1]:.1%}"
        )
    if "accuracy" in passport.test:
        rows["Test accuracy"] = f"{passport.test['accuracy']:.1%} (loss {passport.test['loss']:.3f})"
    if "framework" in passport.training:
        rows["Framework"] = str(passport.training["framework"])
    if passport.created:
        rows["Created"] = passport.created.replace("T", " ")
    return rows


def load_cached(weights: Path, device: str):
    """``(model, passport)`` for ``weights``, loaded once per file version and device."""
    from .cnn import load_model

    stat = weights.stat()
    key = f"{weights.resolve()}|{stat.st_mtime_ns}|{stat.st_size}|{device}"
    if _loaded.get("key") != key:
        _loaded.clear()
        _loaded["value"] = load_model(weights, device)
        _loaded["key"] = key
    return _loaded["value"]


def classify(model, passport: Passport, image: Image.Image, device: str = "cpu") -> Classification:
    """Every class with its probability for ``image``, prepared as the passport says."""
    from .cnn import predict_proba

    seen = to_dataset_image(image, passport.preprocessing.image_size)
    probabilities = predict_proba(model, passport.preprocessing, seen[None], device)[0]
    order = np.argsort(-probabilities, kind="stable")
    return Classification([(passport.classes[i], float(probabilities[i])) for i in order], seen)


def render_classification(image: Image.Image, result: Classification, passport: Passport, top: int) -> Image.Image:
    """The image, what the network saw, the top classes; the training curves below."""
    figure = Figure(figsize=(12, 7.2), dpi=110, layout="constrained")
    grid = figure.add_gridspec(2, 6, height_ratios=[1.15, 1])
    ax_image, ax_seen, ax_bars = (figure.add_subplot(grid[0, 2 * i : 2 * i + 2]) for i in range(3))
    ax_image.imshow(np.asarray(image.convert("RGB")))
    ax_image.set_title(f"Image ({image.width}×{image.height})", fontsize=10)
    size = passport.preprocessing.image_size
    ax_seen.imshow(result.seen, interpolation="nearest")
    ax_seen.set_title(f"What the network gets: {size}×{size}", fontsize=10)
    for axis in (ax_image, ax_seen):
        axis.axis("off")

    shown = result.top(top)
    names = [name for name, _ in shown][::-1]
    values = [p for _, p in shown][::-1]
    colours = ["tab:gray"] * (len(shown) - 1) + ["tab:blue"]
    bars = ax_bars.barh(names, values, color=colours)
    ax_bars.bar_label(bars, labels=[f"{v:.1%}" for v in values], padding=3, fontsize=9)
    ax_bars.set_xlim(0, 1.15)
    ax_bars.set_xlabel("probability")
    ax_bars.set_title(f"Top {len(shown)} of {len(result.ranking)} classes", fontsize=10)
    ax_bars.spines[["top", "right"]].set_visible(False)

    plot_history([figure.add_subplot(grid[1, :3]), figure.add_subplot(grid[1, 3:])], passport)
    name, probability = result.ranking[0]
    figure.suptitle(f"{model_name(passport)}: {name} ({probability:.1%})", fontsize=12)
    return to_figure_image(figure)


class CnnClassifyTool(ForensicsTool):
    tool_id = "cnn_classify"
    produces_figure = True
    title = "CNN classification"
    category = "Classification"
    description = "Load a trained CNN, show its passport and classify the image into CIFAR-10's classes (Set 8.2)."

    def __init__(self) -> None:
        self._weights: Path | None = None

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        if not torch_available():
            messagebox.showinfo(self.title, INSTALL_HINT, parent=parent)
            return None
        params = self._ask_params(parent)
        if params is None:
            return None
        image = document.current
        assert image is not None
        device = best_device()
        try:
            model, passport = load_cached(params.weights, device)
        except (ValueError, OSError) as error:
            messagebox.showinfo(self.title, str(error), parent=parent)
            return None
        result = classify(model, passport, image, device)
        top = min(params.top, len(result.ranking))
        details: dict[str, object] = {"Operation": self.title, "Weights": str(params.weights)}
        for rank, (name, probability) in enumerate(result.top(top), start=1):
            details[f"{rank}. {name}"] = f"{probability:.1%}"
        details.update(passport_details(passport))
        details["Input"] = f"{image.width}×{image.height} {image.mode} → {passport.preprocessing.describe()}"
        name, probability = result.ranking[0]
        return ToolResult(
            image=render_classification(image, result, passport, top),
            message=f"{model_name(passport)}: {name} ({probability:.1%}); "
            f"next {result.ranking[1][0]} ({result.ranking[1][1]:.1%}).",
            details=details,
        )

    def _ask_params(self, parent: tk.Misc | None) -> ClassifyParams | None:
        dialog: ParamDialog[ClassifyParams] = ParamDialog(
            parent,
            title=self.title,
            intro="Classify the image with a model saved by CNN training (its model.pt; the "
            "passport.json beside it says how to prepare the image). The model knows "
            "CIFAR-10's ten classes and sees the image at 32×32.",
            action="Classify",
        )
        chooser = FileChooser(
            dialog,
            dialog.section("Model"),
            prompt="Choose a model.pt saved by CNN training.",
            filetypes=MODEL_TYPES,
            describe=describe_weights,
            initial=self._weights or default_weights(),
        )
        variables = dialog.int_fields(dialog.section("Result"), TOP_FIELDS)

        def validate() -> tuple[ClassifyParams, str]:
            weights = chooser.require()
            passport = load_passport(weights)  # ValueError: shown in the footer
            (top,) = parse_int_fields(TOP_FIELDS, (variables[0].get(),))
            readout = f"{model_name(passport)} · {passport.preprocessing.describe()}"
            return ClassifyParams(weights, min(top, len(passport.classes))), readout

        params = dialog.show(validate)
        if params is not None:
            self._weights = params.weights
        return params
