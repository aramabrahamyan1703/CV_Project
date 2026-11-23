"""YOLO object detection with Ultralytics (Sets 9.1 and 9.3).

9.1 runs a pretrained YOLO11 model (80 COCO classes: person, bus, dog…) on
the working image and draws each box with its class and confidence.

9.3 runs a model trained with the *YOLO training* tool (e.g. on blood cells)
on an unseen test image and shows its boxes beside the image's ground-truth
labels, matched at an IoU threshold, with that image's precision and recall.

"Why are people blue?" (slide 9.1): Ultralytics follows OpenCV and treats a
NumPy image as BGR, and ``results[0].plot()`` returns BGR too, so showing it
with ``plt.imshow`` swaps red and blue. Here the image is converted to BGR
explicitly before prediction, and the boxes are drawn on the RGB image, so
nothing BGR ever reaches PIL or matplotlib.

Ultralytics (and PyTorch) are optional: they are imported only when a YOLO
tool runs, so the rest of the app works without them. The official
``yolo11n.pt`` (5 MB) is downloaded into ``models/`` the first time it is
used; trained weights are the ``best.pt`` files the training tool writes.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import time
import tkinter as tk
from tkinter import messagebox

import cv2
import numpy as np
from matplotlib import colormaps
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .face_recognition import to_figure_image
from .param_dialog import FileChooser, IntField, ParamDialog, parse_int_fields
from .yolo_data import Box, MatchResult, class_counts, label_path_for, match_boxes, read_yolo_labels


ROOT = Path(__file__).resolve().parents[2]
PRETRAINED_NAME = "yolo11n.pt"
DEFAULT_PRETRAINED = ROOT / "models" / PRETRAINED_NAME
# Official weights Ultralytics can download by name (a subset, for the hint).
DOWNLOADABLE = {f"yolo11{size}.pt" for size in "nsmlx"} | {f"yolov8{size}.pt" for size in "nsmlx"}
MODEL_TYPES = [("YOLO weights", "*.pt"), ("All files", "*.*")]
LABEL_TYPES = [("YOLO labels", "*.txt"), ("All files", "*.*")]
INSTALL_HINT = (
    "The YOLO tools need Ultralytics and PyTorch.\n\n"
    "Install them with  python -m pip install ultralytics  (see README)."
)
ACCEPTED_MODES = ("L", "RGB", "RGBA", "P")

DETECT_FIELDS = (
    IntField("Min. confidence", 25, 99, "% score to keep a box"),
    IntField("NMS overlap", 70, 99, "% IoU at which boxes merge"),
    IntField("Image size", 640, 1280, "px, multiple of 32", minimum=32),
)
MATCH_FIELDS = (IntField("Match IoU", 50, 95, "% overlap for a correct box", minimum=10),)

# The weights the training tool saved last, offered first by the test tool.
_last_weights: Path | None = None


def last_weights() -> Path | None:
    return _last_weights


def set_last_weights(path: Path | None) -> None:
    global _last_weights
    _last_weights = path


def ultralytics_available() -> bool:
    return importlib.util.find_spec("ultralytics") is not None


@dataclass(frozen=True)
class DetectParams:
    weights: Path
    conf: float  # 0..1
    iou: float  # 0..1, NMS
    imgsz: int


def parse_detect_params(weights: Path, conf: str, iou: str, size: str) -> DetectParams:
    conf_v, iou_v, size_v = parse_int_fields(DETECT_FIELDS, (conf, iou, size))
    if size_v % 32:
        raise ValueError(f"Image size must be a multiple of 32 (the network's stride); try {round(size_v / 32) * 32 or 32}.")
    return DetectParams(weights, conf_v / 100, iou_v / 100, size_v)


def check_weights(path: Path) -> None:
    """Raise ``ValueError`` unless ``path`` exists or can be downloaded by name."""
    if not ultralytics_available():
        raise ValueError(INSTALL_HINT)
    if not path.is_file() and path.name not in DOWNLOADABLE:
        raise ValueError(f"No weights at {path}.\n\nTrain a model first or choose a .pt file.")


_models: dict[tuple[Path, float], object] = {}


def load_yolo(path: Path):
    """An Ultralytics ``YOLO`` model, loaded once per file version and reused.

    Official weights missing from ``path`` are downloaded there (internet needed).
    """
    check_weights(path)
    if not path.is_file():  # official weights: let Ultralytics download them into place
        path.parent.mkdir(parents=True, exist_ok=True)
        _open_yolo(path)
        _models.clear()  # the download created the file; cache it under its real version
    key = (path.resolve(), path.stat().st_mtime)
    if key not in _models:
        _models[key] = _open_yolo(path)
    return _models[key]


def _open_yolo(path: Path):
    from ultralytics import YOLO

    try:
        model = YOLO(str(path))
    except Exception as error:  # download failures, corrupt or foreign files
        raise ValueError(f"Could not load {path.name} as YOLO weights: {error}") from error
    if getattr(model, "task", "detect") != "detect":
        raise ValueError(f"{path.name} is a {model.task} model, not a detection model.")
    return model


def class_names(model) -> list[str]:
    names = model.names
    return [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)


def detect(model, image: Image.Image, params: DetectParams) -> list[Box]:
    """Boxes found in a PIL image, most confident first, in that image's pixels."""
    # Ultralytics reads NumPy arrays as BGR (OpenCV order): convert explicitly.
    bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    result = model.predict(bgr, conf=params.conf, iou=params.iou, imgsz=params.imgsz, verbose=False)[0]
    boxes = result.boxes
    xyxy = boxes.xyxy.cpu().numpy()
    scores = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)
    found = [
        Box(int(c), float(x1), float(y1), float(x2), float(y2), float(s))
        for (x1, y1, x2, y2), s, c in zip(xyxy, scores, classes)
    ]
    return sorted(found, key=lambda box: -box.score)


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------


def class_colour(class_id: int) -> tuple[int, int, int]:
    """A distinct RGB colour per class (matplotlib's tab10, then tab20)."""
    cmap = colormaps["tab10"] if class_id < 10 else colormaps["tab20"]
    r, g, b, _ = cmap(class_id % cmap.N)
    return int(r * 255), int(g * 255), int(b * 255)


def _mpl_colour(class_id: int) -> tuple[float, float, float]:
    r, g, b = class_colour(class_id)
    return r / 255, g / 255, b / 255


def label_for(box: Box, names: list[str]) -> str:
    return names[box.class_id] if 0 <= box.class_id < len(names) else f"class {box.class_id}"


def draw_detections(image: Image.Image, boxes: list[Box], names: list[str]) -> Image.Image:
    """The image (RGB) with each box and a filled ``"person 0.87"`` tag, sized to the image."""
    out = np.ascontiguousarray(np.asarray(image.convert("RGB"))).copy()
    thick = max(2, round(max(out.shape[:2]) / 350))
    scale = 0.35 * thick
    for box in sorted(boxes, key=lambda b: b.score):  # most confident drawn last, on top
        colour = class_colour(box.class_id)
        p1 = (int(round(box.x1)), int(round(box.y1)))
        p2 = (int(round(box.x2)), int(round(box.y2)))
        cv2.rectangle(out, p1, p2, colour, thick, cv2.LINE_AA)
        text = f"{label_for(box, names)} {box.score:.2f}"
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, max(1, thick // 2))
        top = p1[1] - th - base - 2 if p1[1] - th - base - 2 >= 0 else p1[1]
        cv2.rectangle(out, (p1[0], top), (p1[0] + tw + 4, top + th + base + 2), colour, -1)
        bright = sum(colour) > 450
        cv2.putText(
            out, text, (p1[0] + 2, top + th + 1), cv2.FONT_HERSHEY_SIMPLEX, scale,
            (0, 0, 0) if bright else (255, 255, 255), max(1, thick // 2), cv2.LINE_AA,
        )
    return Image.fromarray(out)


def draw_boxes_on_axis(axis, boxes: list[Box], names: list[str], unmatched: set[int], *, scores: bool) -> None:
    """Class-coloured boxes on a matplotlib axis; indices in ``unmatched`` are dashed."""
    for index, box in enumerate(boxes):
        colour = _mpl_colour(box.class_id)
        axis.add_patch(
            Rectangle(
                (box.x1, box.y1), box.width, box.height, fill=False, edgecolor=colour,
                linewidth=1.6, linestyle="--" if index in unmatched else "-",
            )
        )
        text = label_for(box, names) + (f" {box.score:.2f}" if scores else "")
        axis.text(
            box.x1, box.y1 - 2, text, fontsize=6, color="white", va="bottom",
            bbox={"facecolor": colour, "edgecolor": "none", "pad": 0.8, "alpha": 0.85},
        )


def render_comparison(
    image: Image.Image,
    predictions: list[Box],
    truth: list[Box] | None,
    names: list[str],
    match: MatchResult | None,
    *,
    title: str = "",
    iou_threshold: float = 0.5,
) -> Image.Image:
    """Ground truth beside the prediction (Set 9.3); dashed boxes are unmatched.

    Without ground truth, only the prediction panel is drawn.
    """
    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    panels = 1 if truth is None else 2
    panel_w = 5.2
    figure = Figure(figsize=(panel_w * panels, panel_w * height / width + 1.1), dpi=130, layout="constrained")
    axes = np.atleast_1d(figure.subplots(1, panels))
    if truth is not None:
        missed = set(match.missed) if match else set()
        axes[0].imshow(rgb)
        draw_boxes_on_axis(axes[0], truth, names, missed, scores=False)
        axes[0].set_title(f"Ground truth ({len(truth)}): {class_counts(truth, names)}", fontsize=9)
    false_pos = set(match.false_positives) if match else set()
    axes[-1].imshow(rgb)
    draw_boxes_on_axis(axes[-1], predictions, names, false_pos, scores=True)
    heading = f"Prediction ({len(predictions)}): {class_counts(predictions, names)}"
    if match is not None:
        p, r = match.precision, match.recall
        heading += (
            f"\nIoU ≥ {iou_threshold:.2f}: {match.tp} correct · {len(match.false_positives)} false · "
            f"{len(match.missed)} missed · P {'-' if p is None else f'{p:.2f}'} · "
            f"R {'-' if r is None else f'{r:.2f}'}"
        )
    axes[-1].set_title(heading, fontsize=9)
    for axis in axes:
        axis.axis("off")
    used = sorted({b.class_id for b in predictions} | {b.class_id for b in (truth or [])})
    handles: list = [Patch(color=_mpl_colour(i), label=names[i] if i < len(names) else str(i)) for i in used]
    if match is not None:
        handles.append(Line2D([], [], color="gray", linestyle="--", label="unmatched (missed / false)"))
    if handles:
        figure.legend(handles=handles, loc="outside lower center", ncol=len(handles), fontsize=8, frameon=False)
    if title:
        figure.suptitle(title, fontsize=10)
    return to_figure_image(figure)


# --------------------------------------------------------------------------
# Dialog pieces shared by both tools
# --------------------------------------------------------------------------


def describe_weights(path: Path) -> str:
    if path.is_file():
        return f"{path.stat().st_size / 1e6:.1f} MB · {path.parent.parent.name}/{path.parent.name}" if path.parent.name == "weights" else f"{path.stat().st_size / 1e6:.1f} MB"
    if path.name in DOWNLOADABLE:
        return "not downloaded yet: fetched on first use (internet needed)"
    return "file not found"


def weights_section(dialog: ParamDialog, initial: Path | None, prompt: str) -> FileChooser:
    chooser = FileChooser(
        dialog,
        dialog.section("Model weights"),
        prompt=prompt,
        filetypes=MODEL_TYPES,
        describe=describe_weights,
        initial=initial,
    )
    return chooser


def chosen_weights(chooser: FileChooser, fallback: Path | None) -> Path:
    """The chosen file, else the fallback (which may still need downloading)."""
    path = chooser.path or fallback
    if path is None:
        raise ValueError("Choose a weights file (.pt), e.g. runs/…/weights/best.pt from YOLO training.")
    check_weights(path)
    return path


class _YoloTool(ForensicsTool):
    """Shared availability rules: an L/RGB photo, not this tool's own output."""

    ACCEPTED_MODES = ACCEPTED_MODES
    category = "Detection"
    produces_figure = True

    def is_available(self, document: ImageDocument) -> bool:
        if document.current is None:
            return True
        if document.current.mode not in self.ACCEPTED_MODES:
            return False
        return document.last_tool_id != self.tool_id

    def unavailable_message(self, document: ImageDocument) -> str:
        current = document.current
        if current is not None and current.mode not in self.ACCEPTED_MODES:
            return f"This image mode is not supported (mode {current.mode!r}).\n\nOpen an L or RGB image."
        return "The detections are already shown.\n\nUndo or reset the image to detect again."


# --------------------------------------------------------------------------
# 9.1 Pretrained detection
# --------------------------------------------------------------------------


class YoloDetectionTool(_YoloTool):
    tool_id = "yolo_detection"
    title = "YOLO object detection"
    description = "Detect objects with a pretrained Ultralytics YOLO11 model (80 COCO classes) (Set 9.1)."

    def __init__(self) -> None:
        self._weights = DEFAULT_PRETRAINED

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        source = document.current
        assert source is not None
        params = self._ask_params(parent)
        if params is None:
            return None
        try:
            started = time.perf_counter()
            model = load_yolo(params.weights)
            loaded = time.perf_counter()
            boxes = detect(model, source, params)
            elapsed = time.perf_counter() - loaded
        except ValueError as error:
            messagebox.showinfo(self.title, str(error), parent=parent)
            return None
        names = class_names(model)
        details: dict[str, object] = {
            "Operation": self.title,
            "Model": f"{params.weights.name} ({len(names)} classes)",
            "Min. confidence": f"{params.conf:.2f}",
            "NMS IoU": f"{params.iou:.2f}",
            "Network input": f"{params.imgsz} px (letterboxed; boxes in image pixels)",
            "Objects": f"{len(boxes)}: {class_counts(boxes, names)}",
        }
        for number, box in enumerate(boxes, start=1):
            details[f"{number}. {label_for(box, names)}"] = (
                f"{box.score:.2f} · x={box.x1:.0f}, y={box.y1:.0f}, {box.width:.0f}×{box.height:.0f}"
            )
        details["Time"] = f"{elapsed * 1000:.0f} ms (+ {(loaded - started) * 1000:.0f} ms loading)"
        details["Colours"] = "RGB: converted to BGR for YOLO, drawn on RGB (no blue people)"
        message = (
            f"Found {len(boxes)} object{'' if len(boxes) == 1 else 's'}: {class_counts(boxes, names)}."
            if boxes else "No objects found. Lower the minimum confidence to see weaker boxes."
        )
        return ToolResult(image=draw_detections(source, boxes, names), message=message, details=details)

    def _ask_params(self, parent: tk.Misc | None) -> DetectParams | None:
        dialog: ParamDialog[DetectParams] = ParamDialog(
            parent,
            title=self.title,
            intro="Run a YOLO11 detector pretrained on COCO (people, vehicles, animals…). "
            f"The default {PRETRAINED_NAME} is downloaded into models/ on first use; "
            "any detection .pt file works, including your trained best.pt.",
            action="Detect",
        )
        chooser = weights_section(dialog, self._weights, "Choose YOLO weights (.pt).")
        variables = dialog.int_fields(dialog.section("Detection"), DETECT_FIELDS)

        def validate() -> tuple[DetectParams, str]:
            weights = chosen_weights(chooser, self._weights)
            params = parse_detect_params(weights, *(v.get() for v in variables))
            return params, f"{weights.name} · conf ≥ {params.conf:.2f} · {params.imgsz} px"

        params = dialog.show(validate)
        if params is not None:
            self._weights = params.weights
        return params


# --------------------------------------------------------------------------
# 9.3 Trained model vs ground truth
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TestRequest:
    detect: DetectParams
    labels: Path | None
    match_iou: float


class YoloGroundTruthTool(_YoloTool):
    tool_id = "yolo_ground_truth"
    title = "YOLO test vs ground truth"
    description = (
        "Run trained YOLO weights on a test image and compare its boxes with the "
        "image's YOLO labels (Set 9.3)."
    )

    def __init__(self) -> None:
        self._weights: Path | None = None

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        source = document.current
        assert source is not None
        guess = label_path_for(document.path) if document.path is not None else None
        request = self._ask_params(parent, guess if guess is not None and guess.is_file() else None)
        if request is None:
            return None
        try:
            model = load_yolo(request.detect.weights)
            started = time.perf_counter()
            predictions = detect(model, source, request.detect)
            elapsed = time.perf_counter() - started
            truth = (
                read_yolo_labels(request.labels, source.width, source.height)
                if request.labels is not None else None
            )
        except (ValueError, OSError) as error:
            messagebox.showinfo(self.title, str(error), parent=parent)
            return None
        names = class_names(model)
        match = match_boxes(predictions, truth, request.match_iou) if truth is not None else None
        name = document.path.name if document.path is not None else "image"
        sheet = render_comparison(
            source, predictions, truth, names, match,
            title=f"{name} · {request.detect.weights.parent.parent.name}/{request.detect.weights.name}",
            iou_threshold=request.match_iou,
        )
        details: dict[str, object] = {
            "Operation": self.title,
            "Model": f"{request.detect.weights} ({', '.join(names)})",
            "Min. confidence": f"{request.detect.conf:.2f}",
            "Predicted": f"{len(predictions)}: {class_counts(predictions, names)}",
        }
        if truth is None:
            details["Ground truth"] = "none: no YOLO label file was chosen or found"
            message = f"Predicted {len(predictions)} boxes (no ground truth to compare with)."
        else:
            assert match is not None
            assert request.labels is not None
            details["Ground truth"] = f"{len(truth)}: {class_counts(truth, names)} ({request.labels.name})"
            details["Match IoU"] = f"{request.match_iou:.2f}, same class, most confident first"
            details["Correct (TP)"] = match.tp
            details["False (FP)"] = len(match.false_positives)
            details["Missed (FN)"] = len(match.missed)
            if match.precision is not None:
                details["Precision"] = f"{match.precision:.3f}"
            if match.recall is not None:
                details["Recall"] = f"{match.recall:.3f}"
            if match.matched:
                details["Mean IoU of matches"] = f"{np.mean([m[2] for m in match.matched]):.3f}"
            for class_id, class_name in enumerate(names):
                gt = sum(b.class_id == class_id for b in truth)
                pred = sum(b.class_id == class_id for b in predictions)
                hit = sum(predictions[p].class_id == class_id for p, _, _ in match.matched)
                if gt or pred:
                    details[f"  {class_name}"] = f"{gt} true · {pred} predicted · {hit} correct"
            if document.is_modified:
                details["Note"] = "The image was edited; labels assume the original framing."
            message = (
                f"{match.tp} of {len(truth)} objects found, {len(match.false_positives)} false boxes "
                f"(P {match.precision if match.precision is not None else 0:.2f}, "
                f"R {match.recall if match.recall is not None else 0:.2f})."
            )
        details["Time"] = f"{elapsed * 1000:.0f} ms"
        return ToolResult(image=sheet, message=message, details=details)

    def _ask_params(self, parent: tk.Misc | None, labels: Path | None) -> TestRequest | None:
        dialog: ParamDialog[TestRequest] = ParamDialog(
            parent,
            title=self.title,
            intro="Detect with trained weights (best.pt from YOLO training) and compare "
            "with the image's ground truth: …/images/x.jpg is labelled by …/labels/x.txt.",
            action="Test",
        )
        initial = self._weights or last_weights()
        chooser = weights_section(dialog, initial, "Choose trained YOLO weights (best.pt).")
        label_chooser = FileChooser(
            dialog,
            dialog.section("Ground truth"),
            prompt="Choose the image's YOLO label file (.txt).",
            filetypes=LABEL_TYPES,
            describe=lambda path: f"{sum(1 for line in path.read_text().splitlines() if line.strip())} boxes"
            if path.is_file() else "file not found",
            initial=labels,
        )
        frame = dialog.section("Detection")
        variables = dialog.int_fields(frame, DETECT_FIELDS)
        (match_var,) = dialog.int_fields(frame, MATCH_FIELDS)

        def validate() -> tuple[TestRequest, str]:
            weights = chosen_weights(chooser, None)
            params = parse_detect_params(weights, *(v.get() for v in variables))
            (match_iou,) = parse_int_fields(MATCH_FIELDS, (match_var.get(),))
            label_path = label_chooser.require() if label_chooser.path is not None else None
            readout = f"{weights.name} · " + (
                f"vs {label_path.name}" if label_path else "no labels: predictions only"
            )
            return TestRequest(params, label_path, match_iou / 100), readout

        request = dialog.show(validate)
        if request is not None:
            self._weights = request.detect.weights
        return request
