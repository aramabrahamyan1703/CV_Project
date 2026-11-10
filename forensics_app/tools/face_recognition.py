"""Eigenface builder and face recognition tools (Sets 6.2, 6.3).

The maths lives in :mod:`.eigenfaces`; these tools add the dialogs and the
figures from the slides: the grid of the first eigenfaces, a test face beside
its nearest training faces with their names, and a "predicted / true" gallery
with the accuracy over a whole test folder.

Default locations follow ``scripts/export_lfw.py``: faces in ``data/lfw/train``
and ``data/lfw/test``, the model in ``data/lfw/eigenfaces_model.npz``.
"""

from __future__ import annotations

from collections import Counter
import io
from pathlib import Path
import time
import tkinter as tk
from tkinter import messagebox

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .eigenfaces import (
    MODEL_FILE,
    EigenfaceModel,
    build_model,
    distances,
    list_faces,
    load_faces,
    load_model,
    nearest,
    person_name,
    project,
    same_faces,
    save_model,
    to_face_vector,
)
from .param_dialog import (
    FileChooser,
    FolderChooser,
    IntField,
    ParamDialog,
    parse_int_fields,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = ROOT / "data" / "lfw" / "train"
DEFAULT_TEST = ROOT / "data" / "lfw" / "test"
DEFAULT_MODEL = ROOT / "data" / "lfw" / MODEL_FILE
MODEL_TYPES = [("Eigenface model", "*.npz"), ("All files", "*.*")]
NO_DATA = "Run  python scripts/export_lfw.py  to download the LFW faces into data/lfw."

BUILD_FIELDS = (
    IntField("Eigenfaces", 100, 1000, "K, principal components kept"),
    IntField("Shown", 12, 48, "eigenfaces drawn in the grid"),
)
NEIGHBOUR_FIELDS = (IntField("Closest faces", 3, 10, "training faces shown"),)
WHITEN_HINT = "divide each coefficient by its spread"

# The model a run of the builder saved, offered first by the recognition tools.
_last_model: Path | None = None


def short_name(name: str) -> str:
    """The surname, as in the slide's titles ("George W Bush" -> "Bush")."""
    return name.rsplit(" ", 1)[-1]


def describe_faces(paths: list[Path], folder: Path) -> str:
    people = Counter(person_name(path, folder) for path in paths)
    if not paths:
        return "No face images found."
    with Image.open(paths[0]) as first:
        width, height = first.size
    return (
        f"{len(paths)} faces of {len(people)} people · {width}×{height} px "
        "(one subfolder per person)"
    )


def model_summary(model: EigenfaceModel) -> str:
    height, width = model.shape
    return (
        f"{model.components} eigenfaces · {len(model.names)} faces of "
        f"{len(model.people)} people · {width}×{height} px"
    )


def to_figure_image(figure: Figure) -> Image.Image:
    FigureCanvasAgg(figure)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0.15)
    plt.close(figure)
    buffer.seek(0)
    with Image.open(buffer) as rendered:
        return rendered.convert("RGB")


def gallery(
    faces: list[np.ndarray],
    titles: list[str],
    *,
    columns: int = 4,
    colours: list[str] | None = None,
    heading: str = "",
) -> Image.Image:
    """Faces in a grid with a title over each, like the notebook's plot_gallery."""
    rows = -(-len(faces) // columns)
    columns = min(columns, len(faces))
    figure = Figure(figsize=(1.9 * columns, 2.5 * rows + (0.4 if heading else 0)), dpi=110,
                    layout="constrained")
    axes = np.atleast_1d(figure.subplots(rows, columns)).ravel()
    for index, axis in enumerate(axes):
        axis.set_xticks([])
        axis.set_yticks([])
        if index >= len(faces):
            axis.axis("off")
            continue
        axis.imshow(faces[index], cmap="gray")
        axis.set_title(
            titles[index], fontsize=9, color=(colours[index] if colours else "black")
        )
    if heading:
        figure.suptitle(heading, fontsize=11)
    return to_figure_image(figure)


def eigenface_sheet(mean: np.ndarray, eigenfaces: list[np.ndarray], columns: int = 4) -> Image.Image:
    """The eigenfaces in the slide's grid (3×4 for 12), the mean face in a column to the left."""
    rows = -(-len(eigenfaces) // columns)
    columns = min(columns, len(eigenfaces))
    figure = Figure(figsize=(1.9 * (columns + 1) + 0.3, 2.5 * rows), dpi=110, layout="constrained")
    grid = figure.add_gridspec(rows, columns + 1, width_ratios=[1.15] + [1] * columns)
    middle = figure.add_subplot(grid[rows // 2, 0])
    middle.imshow(mean, cmap="gray")
    middle.set_title("mean face", fontsize=9, fontweight="bold")
    middle.set_xticks([])
    middle.set_yticks([])
    for index, face in enumerate(eigenfaces):
        axis = figure.add_subplot(grid[index // columns, 1 + index % columns])
        axis.imshow(face, cmap="gray")
        axis.set_title(f"eigenface {index}", fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
    return to_figure_image(figure)


# --------------------------------------------------------------------------
# 6.2 Eigenface builder
# --------------------------------------------------------------------------


class EigenfaceBuilderTool(ForensicsTool):
    tool_id = "eigenface_builder"
    produces_figure = True
    title = "Eigenface builder"
    category = "Faces"
    description = "PCA of a folder of faces: save the model and show the first eigenfaces (Set 6.2)."
    requires_image = False

    def __init__(self) -> None:
        self._folder = DEFAULT_TRAIN

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        global _last_model
        request = self._ask_params(parent)
        if request is None:
            return None
        folder, components, shown = request
        started = time.perf_counter()
        try:
            faces = load_faces(folder)
            model = build_model(faces, components, folder)
            target = folder.parent / MODEL_FILE
            save_model(model, target)
        except (ValueError, OSError) as error:
            messagebox.showinfo(self.title, str(error), parent=parent)
            return None
        elapsed = time.perf_counter() - started
        _last_model = target

        height, width = model.shape
        shown = min(shown, model.components)
        sheet = eigenface_sheet(
            model.mean.reshape(height, width),
            [face.reshape(height, width) for face in model.eigenfaces[:shown]],
        )
        people = Counter(model.names)
        details: dict[str, object] = {
            "Operation": self.title,
            "Training folder": str(folder),
            "Faces": len(model.names),
            "People": ", ".join(f"{name} ({count})" for name, count in sorted(people.items())),
            "Face size": f"{width}×{height} px, grayscale",
            "Eigenfaces (K)": model.components,
            "Variance explained": f"{model.explained.sum():.1%} (eigenface 0: {model.explained[0]:.1%})",
            "Model saved to": str(target),
            "Time": f"{elapsed:.2f} s",
        }
        if faces.skipped:
            details["Skipped"] = "; ".join(f"{path.name} ({why})" for path, why in faces.skipped)
        return ToolResult(
            image=sheet,
            message=(
                f"Built {model.components} eigenfaces from {len(model.names)} faces "
                f"({model.explained.sum():.1%} of the variance); saved {target.name}."
            ),
            details=details,
        )

    def _ask_params(self, parent: tk.Misc | None) -> tuple[Path, int, int] | None:
        dialog: ParamDialog[tuple[Path, int, int]] = ParamDialog(
            parent,
            title=self.title,
            intro="Compute the mean face and the eigenfaces of a folder of faces, "
            "and save them with every face's coefficients for recognition.",
            action="Build",
        )
        listings: dict[Path, list[Path]] = {}

        def listing(folder: Path) -> list[Path]:
            if folder not in listings:
                listings[folder] = list_faces(folder)
            return listings[folder]

        chooser = FolderChooser(
            dialog,
            dialog.section("Training faces"),
            prompt="Choose the training folder (one subfolder per person).",
            describe=lambda folder: describe_faces(listing(folder), folder),
            initial=self._folder,
        )
        component_var, shown_var = dialog.int_fields(dialog.section("Model"), BUILD_FIELDS)

        def validate() -> tuple[tuple[Path, int, int], str]:
            if chooser.folder is None and not DEFAULT_TRAIN.is_dir():
                raise ValueError(f"Choose a training folder. {NO_DATA}")
            folder = chooser.require()
            paths = listing(folder)
            if len(paths) < 2:
                raise ValueError("The folder needs at least 2 face images.")
            components, shown = parse_int_fields(
                BUILD_FIELDS, (component_var.get(), shown_var.get())
            )
            if components > len(paths):
                raise ValueError(f"Eigenfaces can be at most {len(paths)}, the number of faces.")
            target = folder.parent / MODEL_FILE
            return (folder, components, shown), f"saves {target.parent.name}/{target.name}"

        result = dialog.show(validate)
        if result is not None:
            self._folder = result[0]
        return result


# --------------------------------------------------------------------------
# 6.3 Face recognition
# --------------------------------------------------------------------------


class ModelSections:
    """Model file, eigenfaces used and whitening: shared by both recognition tools."""

    def __init__(self, dialog: ParamDialog) -> None:
        self._models: dict[Path, EigenfaceModel | str] = {}
        self.components: tk.StringVar | None = None  # created below the model row
        initial = _last_model if _last_model is not None else DEFAULT_MODEL
        self.chooser = FileChooser(
            dialog,
            dialog.section("Eigenface model"),
            prompt="Choose an eigenface model (.npz) made by the Eigenface builder.",
            filetypes=MODEL_TYPES,
            describe=self._describe,
            initial=initial,
        )
        frame = dialog.section("Matching")
        (self.components,) = dialog.int_fields(
            frame, (IntField("Eigenfaces used", 100, 1000, "the first K of the model"),)
        )
        self.whiten = dialog.checkbox(frame, "Whiten", default=True, hint=WHITEN_HINT)
        if self.chooser.path is not None:
            self._offer_all(self.chooser.path)

    def _load(self, path: Path) -> EigenfaceModel | str:
        if path not in self._models:
            try:
                self._models[path] = load_model(path)
            except ValueError as error:
                self._models[path] = str(error)
        return self._models[path]

    def _describe(self, path: Path) -> str:
        model = self._load(path)
        if isinstance(model, str):
            return model
        self._offer_all(path)
        return model_summary(model)

    def _offer_all(self, path: Path) -> None:
        """Start "Eigenfaces used" at every eigenface the chosen model has."""
        model = self._load(path)
        if self.components is not None and not isinstance(model, str):
            self.components.set(str(model.components))

    def read(self) -> tuple[EigenfaceModel, int, bool, str]:
        if self.chooser.path is None:
            raise ValueError(f"Choose an eigenface model, or build one first. {NO_DATA}")
        model = self._load(self.chooser.require())
        if isinstance(model, str):
            raise ValueError(model)
        (components,) = parse_int_fields(
            (IntField("Eigenfaces used", 1, model.components, ""),), (self.components.get(),)
        )
        whiten = bool(self.whiten.get())
        readout = f"{components} of {model.components} eigenfaces · {'whitened' if whiten else 'plain'}"
        return model, components, whiten, readout


class FaceRecognitionTool(ForensicsTool):
    tool_id = "face_recognition"
    produces_figure = True
    title = "Face recognition (eigenfaces)"
    category = "Faces"
    description = "Name the working face by its nearest training faces in eigenface space (Set 6.3)."

    ACCEPTED_MODES = ("L", "RGB", "RGBA", "P")

    def is_available(self, document: ImageDocument) -> bool:
        if document.current is None:
            return True
        if document.current.mode not in self.ACCEPTED_MODES:
            return False
        return document.last_tool_id != self.tool_id

    def unavailable_message(self, document: ImageDocument) -> str:
        current = document.current
        if current is not None and current.mode not in self.ACCEPTED_MODES:
            return f"This image mode is not supported (mode {current.mode!r}). Open an L or RGB image."
        return "The recognition result is already shown.\n\nUndo or reset to recognise the face again."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        source = document.current
        assert source is not None
        request = self._ask_params(parent)
        if request is None:
            return None
        model, components, whiten, shown = request

        weights = project(model, source, components)
        # A training face (or a copy of one) would only find itself at d = 0.
        itself = same_faces(model, to_face_vector(source, model.shape))
        found = nearest(model, weights, shown, whiten, exclude=itself)
        if not found:
            messagebox.showinfo(
                self.title, "Every training face is this same picture; nothing to compare with.",
                parent=parent,
            )
            return None
        predicted = model.names[found[0][0]]
        truth = None
        if document.path is not None:
            label = document.path.parent.name.replace("_", " ")
            truth = label if label in model.names else None

        height, width = model.shape
        test_face = to_face_vector(source, model.shape).reshape(height, width)
        pictures = [test_face] + [model.faces[index].reshape(height, width) for index, _ in found]
        titles = ["test face" + (f"\ntrue: {short_name(truth)}" if truth else "")] + [
            f"{rank}. {short_name(model.names[index])}\nd = {distance:.2f}"
            for rank, (index, distance) in enumerate(found, start=1)
        ]
        verdict = ""
        if truth is not None:
            verdict = "  ✓" if truth == predicted else "  ✗"
        heading = f"predicted: {short_name(predicted)}" + (
            f"   true: {short_name(truth)}{verdict}" if truth else ""
        )

        details: dict[str, object] = {
            "Operation": self.title,
            "Model": f"{model_summary(model)}",
            "Eigenfaces used": components,
            "Distance": "whitened Euclidean" if whiten else "Euclidean (slide's formula)",
            "Predicted": predicted,
        }
        if truth is not None:
            details["True"] = f"{truth} ({'correct' if truth == predicted else 'wrong'})"
        for rank, (index, distance) in enumerate(found, start=1):
            details[f"Closest {rank}"] = f"{model.names[index]} · d = {distance:.3f} · {model.files[index]}"
        if abs(source.width / source.height - width / height) > 0.15 * (width / height):
            details["Note"] = (
                f"The image ({source.width}×{source.height}) was squeezed to the model's "
                f"{width}×{height}; recognition expects a face crop like the training faces."
            )
        if len(itself):
            details["Training face"] = (
                f"this picture is in the training set ({model.files[int(itself[0])]}); "
                "it was left out of its own matches"
            )
        if truth is None:
            message = f"Predicted {predicted}."
        elif truth == predicted:
            message = f"Predicted {predicted}: correct."
        else:
            message = f"Predicted {predicted}, but this is {truth}."
        if len(itself):
            message += " (a training face, compared without itself)"
        return ToolResult(
            image=gallery(pictures, titles, columns=len(pictures), heading=heading),
            message=message,
            details=details,
        )

    def _ask_params(
        self, parent: tk.Misc | None
    ) -> tuple[EigenfaceModel, int, bool, int] | None:
        dialog: ParamDialog[tuple[EigenfaceModel, int, bool, int]] = ParamDialog(
            parent,
            title=self.title,
            intro="Project the working face onto the eigenfaces and show the "
            "training faces with the closest coefficients, with their names.",
            action="Recognise",
        )
        sections = ModelSections(dialog)
        (shown_var,) = dialog.int_fields(dialog.section("Result"), NEIGHBOUR_FIELDS)

        def validate() -> tuple[tuple[EigenfaceModel, int, bool, int], str]:
            model, components, whiten, readout = sections.read()
            (shown,) = parse_int_fields(NEIGHBOUR_FIELDS, (shown_var.get(),))
            return (model, components, whiten, shown), readout

        return dialog.show(validate)


class RecognitionAccuracyTool(ForensicsTool):
    tool_id = "recognition_accuracy"
    produces_figure = True
    title = "Recognition accuracy (test folder)"
    category = "Faces"
    description = "Recognise every face in a test folder: accuracy and a predicted/true gallery (Set 6.3)."
    requires_image = False

    GALLERY = 12  # the slide's 3×4 grid

    def __init__(self) -> None:
        self._folder = DEFAULT_TEST

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        request = self._ask_params(parent)
        if request is None:
            return None
        model, components, whiten, folder = request
        try:
            test = load_faces(folder, model.shape)
        except ValueError as error:
            messagebox.showinfo(self.title, str(error), parent=parent)
            return None

        weights = (test.pixels - model.mean) @ model.eigenfaces[:components].T
        predicted = []
        also_training = 0
        for w, pixels in zip(weights, test.pixels):
            # A test face that is also a training face would just find itself.
            itself = same_faces(model, pixels)
            also_training += bool(len(itself))
            found = nearest(model, w, 1, whiten, exclude=itself)
            predicted.append(model.names[found[0][0]] if found else "")
        truth = list(test.names)
        correct = [p == t for p, t in zip(predicted, truth)]
        accuracy = sum(correct) / len(correct)

        # Mix people in the gallery: order by file name (LFW sample numbers).
        order = sorted(range(len(test.files)), key=lambda i: (test.files[i].stem, str(test.files[i])))
        picks = order[: self.GALLERY]
        height, width = model.shape
        pictures = [test.pixels[i].reshape(height, width) for i in picks]
        titles = [
            f"predicted: {short_name(predicted[i])}\ntrue:      {short_name(truth[i])}" for i in picks
        ]
        colours = ["black" if correct[i] else "#c62828" for i in picks]
        heading = (
            f"Accuracy {accuracy:.1%} ({sum(correct)}/{len(correct)}) · {components} eigenfaces, "
            f"{'whitened' if whiten else 'plain'} · wrong ones in red"
        )

        majority, majority_count = Counter(truth).most_common(1)[0]
        details: dict[str, object] = {
            "Operation": self.title,
            "Model": model_summary(model),
            "Test folder": str(folder),
            "Eigenfaces used": components,
            "Distance": "whitened Euclidean" if whiten else "Euclidean (slide's formula)",
            "Accuracy": f"{accuracy:.1%} ({sum(correct)} of {len(correct)})",
            "Always guessing": f"{majority}: {majority_count / len(truth):.1%}",
        }
        if also_training:
            details["Also in training"] = (
                f"{also_training} of {len(truth)} test faces are training faces; each was "
                "compared without itself. Use a separate test folder for a fair accuracy."
            )
        for name in sorted(set(truth)):
            hits = sum(c for c, t in zip(correct, truth) if t == name)
            total = truth.count(name)
            known = "" if name in model.names else " (not in the model)"
            details[f"  {name}"] = f"{hits}/{total} ({hits / total:.0%}){known}"
        return ToolResult(
            image=gallery(pictures, titles, colours=colours, heading=heading),
            message=f"Recognised {sum(correct)} of {len(correct)} test faces ({accuracy:.1%}).",
            details=details,
        )

    def _ask_params(self, parent: tk.Misc | None) -> tuple[EigenfaceModel, int, bool, Path] | None:
        dialog: ParamDialog[tuple[EigenfaceModel, int, bool, Path]] = ParamDialog(
            parent,
            title=self.title,
            intro="Recognise every face in a test folder by its nearest training "
            "face, and compare with the person folder it is in.",
            action="Evaluate",
        )
        sections = ModelSections(dialog)
        listings: dict[Path, list[Path]] = {}

        def listing(folder: Path) -> list[Path]:
            if folder not in listings:
                listings[folder] = list_faces(folder)
            return listings[folder]

        chooser = FolderChooser(
            dialog,
            dialog.section("Test faces"),
            prompt="Choose the test folder (one subfolder per person).",
            describe=lambda folder: describe_faces(listing(folder), folder),
            initial=self._folder,
        )

        def validate() -> tuple[tuple[EigenfaceModel, int, bool, Path], str]:
            model, components, whiten, readout = sections.read()
            folder = chooser.require()
            if not listing(folder):
                raise ValueError("The test folder has no face images.")
            return (model, components, whiten, folder), f"{len(listing(folder))} faces · {readout}"

        result = dialog.show(validate)
        if result is not None:
            self._folder = result[3]
        return result
