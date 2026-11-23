"""Prepare a detection dataset and train YOLO on it (Set 9.2).

*COCO → YOLO dataset* is the notebook's ``coco_to_yolo`` exercise for every
split at once: it copies the images, writes one label file per image and a
``data.yaml``, and shows the class balance per split beside a training image
drawn from the *written* labels, so a wrong conversion is visible at once.

*YOLO training* fine-tunes the pretrained ``yolo11n.pt`` (or trains
``yolo11n.yaml`` from scratch, as in the notebook) with Ultralytics. Training
runs on a worker thread so the window keeps drawing; Ultralytics callbacks
report each batch, and Cancel asks the trainer to stop after the current
batch, after which it still validates and saves ``best.pt``. Runs are written
to ``runs/`` beside the ``data.yaml``; the last ``best.pt`` is offered by
*YOLO test vs ground truth*.

Measured on an Apple M1 (MPS), blood cells at 640 px: about 0.07 s per
training image per epoch plus ~20 s to validate 288 images, so one epoch of
all 3,027 images takes ~4 minutes. Fine-tuned from yolo11n.pt, five epochs on
30 % of the images (7 minutes) reached mAP50 0.96 and mAP50-95 0.78 on the
validation split (the notebook's 50 epochs from scratch on a T4: 0.97, 0.83).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import threading
import time
import tkinter as tk
from tkinter import messagebox

import numpy as np
from matplotlib.figure import Figure
from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .face_recognition import to_figure_image
from .param_dialog import FileChooser, FolderChooser, IntField, ParamDialog, ProgressWindow, parse_int_fields
from .yolo_data import (
    ConversionCancelled,
    ConversionSummary,
    DATA_YAML,
    convert_coco_dataset,
    dataset_folder,
    describe_coco_root,
    find_coco_splits,
    label_path_for,
    list_images,
    read_data_yaml,
    read_yolo_labels,
)
from . import yolo_detection
from .yolo_detection import DEFAULT_PRETRAINED, INSTALL_HINT, check_weights, draw_boxes_on_axis, ultralytics_available


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COCO = ROOT / "data" / "blood_cell" / "coco"
YAML_TYPES = [("Dataset YAML", "*.yaml *.yml"), ("All files", "*.*")]
SCRATCH_CONFIG = "yolo11n.yaml"  # the architecture only, random weights

TRAIN_FIELDS = (
    IntField("Epochs", 10, 300, "passes over the training images"),
    IntField("Batch", 16, 64, "images per step"),
    IntField("Image size", 640, 1280, "px, multiple of 32", minimum=64),
    IntField("Data used", 100, 100, "% of training images (less = quicker)"),
)
# Measured on an M1 (see the module docstring); CPU is roughly 4× slower.
SECONDS_PER_IMAGE = {"mps": 0.07, "cuda": 0.02, "cpu": 0.3}
VALIDATE_SECONDS_PER_IMAGE = {"mps": 0.07, "cuda": 0.01, "cpu": 0.15}

_last_data_yaml: Path | None = None


def default_output(coco_root: Path) -> Path:
    """Where the YOLO copy of ``coco_root`` goes: a sibling ``<name>_yolo`` folder."""
    return coco_root.parent / f"{coco_root.name}_yolo"


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def render_dataset_sheet(summary: ConversionSummary, sample: Path | None) -> Image.Image:
    """Boxes per class and split (bars) beside a sample image drawn from its YOLO labels."""
    figure = Figure(figsize=(11, 4.6), dpi=120, layout="constrained")
    ax_bar, ax_img = figure.subplots(1, 2, width_ratios=[1.1, 1])
    names = summary.names
    width = 0.8 / max(1, len(summary.splits))
    x = np.arange(len(names))
    for offset, split in enumerate(summary.splits):
        counts = [split.boxes.get(name, 0) for name in names]
        bars = ax_bar.bar(x + (offset - (len(summary.splits) - 1) / 2) * width, counts, width,
                          label=f"{split.name} ({split.images} images)")
        ax_bar.bar_label(bars, fontsize=7, padding=1)
    ax_bar.set_xticks(x, names)
    ax_bar.set_ylabel("Boxes")
    ax_bar.set_title("Boxes per class")
    ax_bar.legend(fontsize=8)

    ax_img.axis("off")
    if sample is not None:
        with Image.open(sample) as opened:
            image = opened.convert("RGB")
        boxes = read_yolo_labels(label_path_for(sample), image.width, image.height)
        ax_img.imshow(np.asarray(image))
        draw_boxes_on_axis(ax_img, boxes, names, set(), scores=False)
        ax_img.set_title(f"Read back from the YOLO labels ({len(boxes)} boxes): {sample.name[:20]}…", fontsize=9)
    figure.suptitle(f"YOLO dataset: {summary.output.name}", fontsize=11)
    return to_figure_image(figure)


def read_results_csv(path: Path) -> dict[str, list[float]]:
    """Ultralytics' ``results.csv`` as ``{column: values}`` (headers stripped)."""
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    columns: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            if key is None:
                continue
            try:
                columns.setdefault(key.strip(), []).append(float(value))
            except (TypeError, ValueError):
                continue
    return columns


def render_training_curves(columns: dict[str, list[float]], title: str = "") -> Image.Image:
    """Training and validation losses and the validation metrics per epoch."""
    epochs = columns.get("epoch") or list(range(1, len(next(iter(columns.values()), [])) + 1))
    figure = Figure(figsize=(12, 3.8), dpi=120, layout="constrained")
    axes = figure.subplots(1, 3)
    groups = (
        ("Training loss", ("train/box_loss", "train/cls_loss", "train/dfl_loss")),
        ("Validation loss", ("val/box_loss", "val/cls_loss", "val/dfl_loss")),
        ("Validation metrics", ("metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)")),
    )
    for axis, (heading, keys) in zip(axes, groups):
        for key in keys:
            values = columns.get(key)
            if values:
                axis.plot(epochs[: len(values)], values, marker="o", markersize=3,
                          label=key.split("/", 1)[1].replace("(B)", ""))
        axis.set_title(heading)
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.3)
        if axis.lines:
            axis.legend(fontsize=8)
    axes[2].set_ylim(0, 1)
    if title:
        figure.suptitle(title, fontsize=11)
    return to_figure_image(figure)


# --------------------------------------------------------------------------
# COCO -> YOLO tool
# --------------------------------------------------------------------------


class CocoToYoloTool(ForensicsTool):
    tool_id = "coco_to_yolo"
    produces_figure = True
    requires_image = False
    title = "COCO → YOLO dataset"
    category = "Detection"
    description = "Convert a COCO-format dataset (train/valid/test) to YOLO labels and a data.yaml (Set 9.2)."

    def __init__(self) -> None:
        self._folder = DEFAULT_COCO

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        global _last_data_yaml
        root = self._ask_folder(parent)
        if root is None:
            return None
        output = default_output(root)
        window = ProgressWindow(parent, title="Converting to YOLO…", total=1) if parent else None

        def progress(done: int, total: int, text: str) -> bool:
            return window is None or window.step(done, f"{done + 1} of {total} · {text}", total)

        started = time.perf_counter()
        try:
            summary = convert_coco_dataset(root, output, progress=progress)
        except ConversionCancelled:
            return None
        except (ValueError, OSError, KeyError) as error:
            if window is not None:
                window.close()
                window = None
            messagebox.showinfo(self.title, f"Could not convert {root.name}: {error}", parent=parent)
            return None
        finally:
            if window is not None:
                window.close()
        elapsed = time.perf_counter() - started
        _last_data_yaml = summary.data_yaml

        train = next((s for s in summary.splits if s.name == "train"), summary.splits[0])
        images = list_images(output / train.name / "images")
        sample = images[0] if images else None
        details: dict[str, object] = {
            "Operation": self.title,
            "COCO folder": str(root),
            "YOLO folder": str(output),
            "data.yaml": str(summary.data_yaml),
            "Classes": ", ".join(f"{i} {name}" for i, name in enumerate(summary.names)),
        }
        for split in summary.splits:
            line = f"{split.images} images · {sum(split.boxes.values())} boxes ({', '.join(f'{split.boxes[n]} {n}' for n in summary.names)})"
            if split.skipped_boxes:
                line += f" · {split.skipped_boxes} boxes skipped (empty or outside)"
            if split.missing_images:
                line += f" · {split.missing_images} listed images missing"
            details[f"  {split.name}"] = line
        details["Label format"] = "class x_center y_center width height (0..1)"
        details["Time"] = f"{elapsed:.1f} s"
        return ToolResult(
            image=render_dataset_sheet(summary, sample),
            message=f"Converted {summary.total_images} images to YOLO format in {output.name}; wrote {DATA_YAML}.",
            details=details,
        )

    def _ask_folder(self, parent: tk.Misc | None) -> Path | None:
        dialog: ParamDialog[Path] = ParamDialog(
            parent,
            title=self.title,
            intro="Choose a COCO dataset folder: train/, valid/ and test/ each holding images "
            "and _annotations.coco.json (Roboflow's export). The YOLO copy is written "
            "beside it, as <folder>_yolo.",
            action="Convert",
        )
        chooser = FolderChooser(
            dialog,
            dialog.section("COCO dataset"),
            prompt="Choose the COCO dataset folder.",
            describe=describe_coco_root,
            initial=self._folder,
        )

        def validate() -> tuple[Path, str]:
            root = chooser.require()
            find_coco_splits(root)
            return root, f"writes {default_output(root).name}/ with images, labels and {DATA_YAML}"

        root = dialog.show(validate)
        if root is not None:
            self._folder = root
        return root


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


def training_device() -> str:
    """``mps`` on Apple silicon, ``cuda`` with an NVIDIA GPU, else ``cpu``."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass(frozen=True)
class TrainRequest:
    data_yaml: Path
    pretrained: bool
    epochs: int
    batch: int
    imgsz: int
    fraction: float  # 0..1 of the training images
    device: str

    @property
    def weights(self) -> str:
        return str(DEFAULT_PRETRAINED) if self.pretrained else SCRATCH_CONFIG

    @property
    def run_name(self) -> str:
        return f"{'pretrained' if self.pretrained else 'scratch'}_{self.epochs}e"


@dataclass(frozen=True)
class TrainOutcome:
    save_dir: Path
    best: Path
    epochs_done: int
    stopped_early: bool
    map50: float
    map50_95: float
    per_class: dict[str, tuple[float, float]]  # name -> (AP50, AP50-95)
    seconds: float


class TrainingJob:
    """One Ultralytics training run on a worker thread, observable from the UI thread.

    The callbacks only write plain attributes; the UI thread reads them
    through :meth:`status` and asks for a stop with :meth:`request_stop`.
    """

    def __init__(self, request: TrainRequest) -> None:
        self.request = request
        self.outcome: TrainOutcome | None = None
        self.error: BaseException | None = None
        self.phase = "Preparing the dataset…"
        self.epoch = 0
        self.batch = 0
        self.batches = 0
        self.loss = ""
        self.last_map50: float | None = None
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="yolo-training", daemon=True)

    # UI thread -----------------------------------------------------------
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

    def status(self) -> tuple[int, int, str]:
        """``(done, total, text)`` in batches, for a progress bar."""
        epochs = self.request.epochs
        if self.batches == 0:
            return 0, 1, self.phase
        done = (self.epoch - 1) * self.batches + self.batch
        text = f"Epoch {self.epoch}/{epochs} · batch {self.batch}/{self.batches}"
        if self.loss:
            text += f" · {self.loss}"
        if self.last_map50 is not None:
            text += f" · mAP50 {self.last_map50:.2f}"
        if self.phase:
            text = f"{text}\n{self.phase}"
        return done, epochs * self.batches, text

    # Worker thread --------------------------------------------------------
    def _on_epoch_start(self, trainer) -> None:
        self.epoch = trainer.epoch + 1
        self.batch = 0
        self.batches = len(trainer.train_loader)
        self.phase = ""

    def _on_batch_end(self, trainer) -> None:
        self.batch += 1
        try:
            losses = trainer.tloss.tolist() if hasattr(trainer.tloss, "tolist") else [float(trainer.tloss)]
            names = [n.replace("_loss", "") for n in trainer.loss_names]
            self.loss = "loss " + " ".join(f"{n} {v:.2f}" for n, v in zip(names, losses))
        except (AttributeError, TypeError, ValueError):
            pass
        if self.batch >= self.batches:
            self.phase = "Validating…"
        if self._stop:
            trainer.stop = True
            self.phase = "Stopping: validating and saving best.pt…"

    def _on_fit_epoch_end(self, trainer) -> None:
        value = (trainer.metrics or {}).get("metrics/mAP50(B)")
        if value is not None:
            self.last_map50 = float(value)

    def _run(self) -> None:
        try:
            self.outcome = self._train()
        except BaseException as error:  # reported on the UI thread
            self.error = error

    def _train(self) -> TrainOutcome:
        from ultralytics import YOLO

        r = self.request
        started = time.perf_counter()
        if r.pretrained:
            DEFAULT_PRETRAINED.parent.mkdir(parents=True, exist_ok=True)
        model = YOLO(r.weights)
        model.add_callback("on_train_epoch_start", self._on_epoch_start)
        model.add_callback("on_train_batch_end", self._on_batch_end)
        model.add_callback("on_fit_epoch_end", self._on_fit_epoch_end)
        metrics = model.train(
            data=str(r.data_yaml),
            epochs=r.epochs,
            batch=r.batch,
            imgsz=r.imgsz,
            fraction=r.fraction,
            device=r.device,
            project=str(r.data_yaml.parent / "runs"),
            name=r.run_name,
            plots=True,
            verbose=False,
        )
        save_dir = Path(model.trainer.save_dir)
        best = save_dir / "weights" / "best.pt"
        if not best.is_file():
            best = save_dir / "weights" / "last.pt"
        names = [metrics.names[i] for i in sorted(metrics.names)]
        per_class = {}
        for index, class_id in enumerate(metrics.box.ap_class_index):
            per_class[names[int(class_id)]] = (float(metrics.box.ap50[index]), float(metrics.box.ap[index]))
        results = save_dir / "results.csv"
        epochs_done = len(read_results_csv(results).get("epoch", [])) if results.is_file() else self.epoch
        return TrainOutcome(
            save_dir=save_dir,
            best=best,
            epochs_done=epochs_done,
            stopped_early=self._stop or epochs_done < r.epochs,
            map50=float(metrics.box.map50),
            map50_95=float(metrics.box.map),
            per_class=per_class,
            seconds=time.perf_counter() - started,
        )


def describe_data_yaml(path: Path) -> str:
    try:
        data = read_data_yaml(path)
    except ValueError as error:
        return str(error)
    parts = []
    for key in ("train", "val", "test"):
        folder = dataset_folder(data, path, key)
        if folder is not None:
            count = len(list_images(folder)) if folder.is_dir() else 0
            parts.append(f"{key} {count}")
    return f"{' · '.join(parts)} images · classes {', '.join(data['names'])}"


def estimate_minutes(train_images: int, val_images: int, request: TrainRequest) -> float:
    per = SECONDS_PER_IMAGE.get(request.device, SECONDS_PER_IMAGE["cpu"])
    val = VALIDATE_SECONDS_PER_IMAGE.get(request.device, VALIDATE_SECONDS_PER_IMAGE["cpu"])
    scale = (request.imgsz / 640) ** 2
    per_epoch = train_images * request.fraction * per * scale + val_images * val * scale
    return request.epochs * per_epoch / 60


class YoloTrainingTool(ForensicsTool):
    tool_id = "yolo_training"
    produces_figure = True
    requires_image = False
    title = "YOLO training"
    category = "Detection"
    description = "Fine-tune YOLO11n (or train from scratch) on a YOLO dataset's data.yaml and save best.pt (Set 9.2)."

    def __init__(self) -> None:
        self._data_yaml: Path | None = None

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        if not ultralytics_available():
            messagebox.showinfo(self.title, INSTALL_HINT, parent=parent)
            return None
        request = self._ask_params(parent)
        if request is None:
            return None
        job = TrainingJob(request)
        window = ProgressWindow(parent, title="Training YOLO…", total=1)
        job.start()
        try:
            while job.is_alive():
                done, total, text = job.status()
                if not window.cancelled:
                    if not window.step(done, text, total):
                        job.request_stop()
                else:
                    job.request_stop()
                    window.note(f"{text}\nStopping after this batch; validating and saving best.pt…")
                job.join(0.1)
        finally:
            window.close()
        if job.error is not None:
            messagebox.showinfo(self.title, f"Training failed: {job.error}", parent=parent)
            return None
        outcome = job.outcome
        assert outcome is not None
        yolo_detection.set_last_weights(outcome.best)

        columns = read_results_csv(outcome.save_dir / "results.csv") if (outcome.save_dir / "results.csv").is_file() else {}
        source = "yolo11n.pt (pretrained on COCO)" if request.pretrained else "yolo11n.yaml (from scratch)"
        sheet = render_training_curves(columns, f"{outcome.save_dir.name}: {source}, {outcome.epochs_done} epochs")
        details: dict[str, object] = {
            "Operation": self.title,
            "Dataset": str(request.data_yaml),
            "Start": source,
            "Epochs": f"{outcome.epochs_done} of {request.epochs}" + (" (stopped early)" if outcome.stopped_early else ""),
            "Batch · size": f"{request.batch} · {request.imgsz} px",
            "Data used": f"{request.fraction:.0%} of the training images",
            "Device": request.device,
            "mAP50 (val)": f"{outcome.map50:.3f}",
            "mAP50-95 (val)": f"{outcome.map50_95:.3f}",
        }
        for name, (ap50, ap) in outcome.per_class.items():
            details[f"  {name}"] = f"AP50 {ap50:.3f} · AP50-95 {ap:.3f}"
        details["Weights"] = str(outcome.best)
        details["Run folder"] = str(outcome.save_dir)
        details["Time"] = f"{outcome.seconds / 60:.1f} min"
        return ToolResult(
            image=sheet,
            message=f"Trained {outcome.epochs_done} epochs: mAP50 {outcome.map50:.3f}, "
            f"mAP50-95 {outcome.map50_95:.3f}; saved {outcome.best.parent.parent.name}/weights/{outcome.best.name}.",
            details=details,
        )

    def _ask_params(self, parent: tk.Misc | None) -> TrainRequest | None:
        dialog: ParamDialog[TrainRequest] = ParamDialog(
            parent,
            title=self.title,
            intro="Train a YOLO11n detector on a dataset's data.yaml (e.g. the one COCO → YOLO "
            "wrote). Runs are saved in runs/ beside the data.yaml. Cancel stops after the "
            "current batch and keeps the best weights so far.",
            action="Train",
        )
        initial = self._data_yaml or _last_data_yaml
        chooser = FileChooser(
            dialog,
            dialog.section("Dataset"),
            prompt="Choose the dataset's data.yaml.",
            filetypes=YAML_TYPES,
            describe=describe_data_yaml,
            initial=initial,
        )
        frame = dialog.section("Training")
        pretrained_var = dialog.checkbox(
            frame, "Start from pretrained yolo11n.pt", default=True,
            hint="untick: from scratch, as in the notebook",
        )
        variables = dialog.int_fields(frame, TRAIN_FIELDS)
        device = training_device()
        counts: dict[Path, tuple[int, int]] = {}

        def validate() -> tuple[TrainRequest, str]:
            path = chooser.require()
            data = read_data_yaml(path)
            epochs, batch, imgsz, used = parse_int_fields(TRAIN_FIELDS, tuple(v.get() for v in variables))
            if imgsz % 32:
                raise ValueError("Image size must be a multiple of 32.")
            if path not in counts:
                train_dir, val_dir = dataset_folder(data, path, "train"), dataset_folder(data, path, "val")
                counts[path] = (
                    len(list_images(train_dir)) if train_dir and train_dir.is_dir() else 0,
                    len(list_images(val_dir)) if val_dir and val_dir.is_dir() else 0,
                )
            n_train, n_val = counts[path]
            if n_train == 0:
                raise ValueError("The data.yaml's train folder has no images.")
            pretrained = bool(pretrained_var.get())
            if pretrained:
                check_weights(DEFAULT_PRETRAINED)
            request = TrainRequest(path, pretrained, epochs, batch, imgsz, used / 100, device)
            minutes = estimate_minutes(n_train, n_val, request)
            readout = (
                f"{round(n_train * request.fraction)} training images · {device} · "
                f"about {minutes:.0f} min" if minutes >= 1 else f"{device} · under a minute"
            )
            return request, readout

        request = dialog.show(validate)
        if request is not None:
            self._data_yaml = request.data_yaml
        return request
