"""Train a CIFAR-10 classifier and save its weights and passport (Set 8.1).

*CNN training* trains one of the two networks of ``cnn.py`` on CIFAR-10 and
saves ``model.pt`` and ``passport.json`` to ``data/cifar10/models/<model>/``,
replacing the model saved there before. The validation split is the last 10%
of the 50,000 training images (Keras' ``validation_split=0.1``, as in the
notebook); the 10,000 test images are used once, after training, for the
accuracy written into the passport.

* Small CNN: SGD (lr 0.01, momentum 0.9) on 32×32 images.
* InceptionV3: the notebook's transfer learning. First the new head alone,
  on features InceptionV3 computes once (Adam, lr 1e-3); then optionally
  "fine-tune epochs" of the whole network (Adam, lr 1e-4).

Training runs on a worker thread so the window keeps drawing. Cancel stops
after the current batch and keeps the weights of the last finished epoch,
which are then tested and saved; cancelled before an epoch finished, nothing
is saved.

Measured on an Apple M3 Pro (MPS), seed 0, all 45,000 training images: the
small CNN takes ~5 s per epoch; 10 epochs (0.8 min) reached 66.1% validation
and 65.0% test accuracy. InceptionV3 computes features at ~205 images/s (~4 min
for all 60,000 images), after which head epochs take seconds; 10 of them (5.8
min in all) reached 86.2% validation and 86.1% test accuracy. Fine-tuning all
layers runs at ~57 images/s (~13 min per epoch of 45,000; not measured end to end).

Nothing here imports PyTorch until training starts, so the app opens without
it; the tool then says how to install it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
import importlib.util
from pathlib import Path
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

from matplotlib.figure import Figure
from PIL import Image

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .cnn_data import (
    CIFAR10_CLASSES,
    CIFAR_DIR,
    INCEPTION_V3,
    MODELS_DIR,
    SMALL_CNN,
    WEIGHTS_FILE,
    Passport,
)
from .face_recognition import to_figure_image
from .param_dialog import IntField, ParamDialog, ProgressWindow, parse_int_fields


INSTALL_HINT = (
    "The CNN tools need PyTorch and torchvision.\n\n"
    "Install them with  python -m pip install -r requirements.txt  (see README)."
)
MODEL_CHOICES = {
    "Small CNN": SMALL_CNN,
    "InceptionV3 (transfer learning)": INCEPTION_V3,
}
TRAIN_IMAGES = 45_000  # CIFAR-10's 50,000 minus the validation tenth
VAL_IMAGES = 5_000
TEST_IMAGES = 10_000
TRAIN_FIELDS = (
    IntField("Epochs", 10, 200, "InceptionV3: epochs of the new head"),
    IntField("Fine-tune epochs", 0, 20, "InceptionV3 only: then all layers", minimum=0),
    IntField("Training images", TRAIN_IMAGES, TRAIN_IMAGES, "random subset (less = quicker)", minimum=100),
)

# Seconds per image on Apple MPS (M3 Pro), measured; other devices get no estimate.
SMALL_SECONDS_PER_IMAGE = 0.00013  # one training epoch, validation included
INCEPTION_FEATURE_SECONDS = 1 / 205  # one forward pass (features, validation, test)
INCEPTION_FINE_TUNE_SECONDS = 1 / 57  # one training step of the whole network


# The model the training tool saved last, offered first by the classifier.
_last_model: Path | None = None


def last_model() -> Path | None:
    return _last_model


def set_last_model(folder: Path | None) -> None:
    global _last_model
    _last_model = folder


def torch_available() -> bool:
    return all(importlib.util.find_spec(name) is not None for name in ("torch", "torchvision"))


def cifar_downloaded(root: Path = CIFAR_DIR) -> bool:
    return (root / "cifar-10-batches-py").is_dir()


def best_device() -> str:
    """``cuda``, ``mps`` or ``cpu``, without importing torch if it is missing."""
    if not torch_available():
        return "cpu"
    from .cnn import best_device as device

    return device()


@dataclass(frozen=True)
class CnnTrainRequest:
    architecture: str  # SMALL_CNN or INCEPTION_V3
    epochs: int  # the small CNN's, or InceptionV3's head epochs
    fine_tune_epochs: int  # InceptionV3 only
    train_samples: int
    device: str
    output: Path  # folder for model.pt and passport.json
    seed: int = 0


def estimate_seconds(request: CnnTrainRequest) -> float | None:
    """Training, validation and testing time on MPS; ``None`` elsewhere (not measured)."""
    if request.device != "mps":
        return None
    if request.architecture == SMALL_CNN:
        return request.epochs * request.train_samples * SMALL_SECONDS_PER_IMAGE
    forward = (request.train_samples + VAL_IMAGES + TEST_IMAGES) * INCEPTION_FEATURE_SECONDS
    fine_tune = request.fine_tune_epochs * (
        request.train_samples * INCEPTION_FINE_TUNE_SECONDS + VAL_IMAGES * INCEPTION_FEATURE_SECONDS
    )
    return forward + fine_tune


def describe_duration(seconds: float) -> str:
    if seconds < 60:
        return "under a minute"
    return f"about {seconds / 60:.0f} min"


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------


def render_history(passport: Passport, title: str = "") -> Image.Image:
    """Loss and accuracy per epoch, training and validation, from a passport."""
    figure = Figure(figsize=(11, 3.8), dpi=120, layout="constrained")
    plot_history(figure.subplots(1, 2), passport)
    if title:
        figure.suptitle(title, fontsize=11)
    return to_figure_image(figure)


def plot_history(axes, passport: Passport) -> None:
    """Draw the loss (``axes[0]``) and accuracy (``axes[1]``) curves of a passport."""
    history = passport.history
    stages = passport.training.get("stages", [])
    head_epochs = stages[0].get("epochs_done", 0) if len(stages) > 1 else 0
    for axis, (heading, key) in zip(axes, (("Cross-entropy loss", "loss"), ("Accuracy", "accuracy"))):
        for name, label in ((key, "train"), (f"val_{key}", "validation")):
            values = history.get(name, [])
            if values:
                axis.plot(range(1, len(values) + 1), values, marker="o", markersize=3, label=label)
        if head_epochs:
            axis.axvline(head_epochs + 0.5, color="0.5", linestyle="--", linewidth=1)
            axis.text(head_epochs + 0.6, 0.98, "fine-tuning", transform=axis.get_xaxis_transform(),
                      va="top", fontsize=8, color="0.4")
        if "accuracy" in passport.test and key == "accuracy":
            axis.axhline(passport.test["accuracy"], color="tab:green", linestyle=":", linewidth=1.2,
                         label=f"test {passport.test['accuracy']:.1%}")
        axis.set_title(heading)
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.3)
        if axis.lines:
            axis.legend(fontsize=8)
    axes[1].set_ylim(0, 1)


# --------------------------------------------------------------------------
# Training job
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CnnTrainOutcome:
    folder: Path
    passport: Passport
    seconds: float


# () -> (training images, test images, class names); CIFAR-10 unless a test swaps it.
DataLoader = Callable[[], tuple[object, object, tuple[str, ...]]]


class CnnTrainingJob:
    """One training run on a worker thread, observable from the UI thread.

    The worker only writes plain attributes; the UI thread reads them through
    :meth:`status` and asks for a stop with :meth:`request_stop`. When the
    thread ends, ``outcome`` holds the saved model, or is ``None`` with
    ``error`` set, or ``None`` alone if Cancel came before the first epoch.
    """

    def __init__(self, request: CnnTrainRequest, load_data: DataLoader | None = None) -> None:
        self.request = request
        self._load_data = load_data
        self.outcome: CnnTrainOutcome | None = None
        self.error: BaseException | None = None
        self.stage = "Loading CIFAR-10…"
        self.done = 0
        self.total = 0
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="cnn-training", daemon=True)

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
        """``(done, total, text)`` of the current stage, for a progress bar."""
        if self.total == 0:
            return 0, 1, self.stage
        return self.done, self.total, f"{self.stage} · {self.done}/{self.total}"

    # Worker thread --------------------------------------------------------
    def _progress(self, stage: str, done: int, total: int) -> None:
        self.stage, self.done, self.total = stage, done, total

    def _phase(self, text: str) -> None:
        self.stage, self.done, self.total = text, 0, 0

    def _run(self) -> None:
        from .cnn import TrainingCancelled

        try:
            self.outcome = self._train()
        except TrainingCancelled:
            self.outcome = None
        except BaseException as error:  # reported on the UI thread
            self.error = error

    def _train(self) -> CnnTrainOutcome:
        from . import cnn

        r = self.request
        started = time.perf_counter()
        if self._load_data is not None:
            everything, test, classes = self._load_data()
        else:
            if not cifar_downloaded():
                self._phase("Downloading CIFAR-10 (170 MB, once)…")
            everything, test = cnn.load_cifar10()
            classes = CIFAR10_CLASSES
        train_set, val_set = cnn.split_train_val(everything, train_samples=r.train_samples, seed=r.seed)
        self._phase("Building the model" + (" (ImageNet weights download once)…" if r.architecture == INCEPTION_V3 else "…"))
        model = cnn.build_model(r.architecture, len(classes), seed=r.seed)
        pre = cnn.preprocessing_for(r.architecture)

        config = replace(cnn.default_config(r.architecture, r.epochs, r.device), seed=r.seed)
        result = cnn.train(model, pre, train_set, val_set, config, self._progress, lambda: self._stop)
        stages = [(config, result)]
        if r.architecture == INCEPTION_V3 and r.fine_tune_epochs and not self._stop:
            tune = replace(cnn.default_config(r.architecture, r.fine_tune_epochs, r.device, fine_tune=True), seed=r.seed)
            try:
                stages.append(
                    (tune, cnn.train(model, pre, train_set, val_set, tune, self._fine_tune_progress, lambda: self._stop))
                )
            except cnn.TrainingCancelled:
                pass  # the head's weights are kept, untouched by the stopped fine-tuning

        # Not cancellable: whatever is saved gets its test score.
        score = cnn.evaluate(model, pre, test, config.batch_size, r.device, self._progress)
        passport = cnn.passport_for(r.architecture, classes, train_set, val_set, test, stages, score)
        cnn.save_model(model, passport, r.output)
        return CnnTrainOutcome(r.output, passport, time.perf_counter() - started)

    def _fine_tune_progress(self, stage: str, done: int, total: int) -> None:
        self._progress(f"Fine-tuning · {stage}", done, total)


# --------------------------------------------------------------------------
# Tool
# --------------------------------------------------------------------------


class CnnTrainingTool(ForensicsTool):
    tool_id = "cnn_training"
    produces_figure = True
    requires_image = False
    title = "CNN training"
    category = "Classification"
    description = "Train a small CNN or InceptionV3 on CIFAR-10; save the weights and the model passport (Set 8.1)."

    def __init__(self) -> None:
        self._architecture = SMALL_CNN

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        if not torch_available():
            messagebox.showinfo(self.title, INSTALL_HINT, parent=parent)
            return None
        request = self._ask_params(parent)
        if request is None:
            return None
        job = CnnTrainingJob(request)
        window = ProgressWindow(parent, title="Training the CNN…", total=1)
        job.start()
        try:
            while job.is_alive():
                done, total, text = job.status()
                if not window.cancelled:
                    if not window.step(done, text, total):
                        job.request_stop()
                else:
                    job.request_stop()
                    window.note(f"{text}\nStopping: keeping the last finished epoch, then testing and saving…")
                job.join(0.1)
        finally:
            window.close()
        if job.error is not None:
            messagebox.showinfo(self.title, f"Training failed: {job.error}", parent=parent)
            return None
        if job.outcome is None:
            messagebox.showinfo(self.title, "Cancelled before the first epoch finished; nothing was saved.", parent=parent)
            return None
        set_last_model(job.outcome.folder)
        return self._result(request, job.outcome)

    def _result(self, request: CnnTrainRequest, outcome: CnnTrainOutcome) -> ToolResult:
        passport = outcome.passport
        name = next(label for label, key in MODEL_CHOICES.items() if key == request.architecture)
        training = passport.training
        stages = training.get("stages", [])
        planned = request.epochs + (request.fine_tune_epochs if request.architecture == INCEPTION_V3 else 0)
        epochs = f"{training['epochs_done']} of {planned}" + (" (stopped early)" if training["cancelled"] else "")
        if len(stages) > 1:
            epochs += f": {stages[0]['epochs_done']} head + {stages[1]['epochs_done']} fine-tuning"
        history = passport.history
        test_accuracy = passport.test.get("accuracy")
        sheet = render_history(
            passport, f"{name} on {passport.dataset['name']}: {training['epochs_done']} epochs, "
            f"test accuracy {test_accuracy:.1%}" if test_accuracy is not None else name,
        )
        dataset = passport.dataset
        details: dict[str, object] = {
            "Operation": self.title,
            "Model": name,
            "Dataset": f"{dataset['name']}: {dataset['train_samples']:,} train · "
            f"{dataset['val_samples']:,} validation · {dataset['test_samples']:,} test images",
            "Epochs": epochs,
            "Device": request.device,
            "Preprocessing": passport.preprocessing.describe(),
            "Train accuracy (last epoch)": f"{history['accuracy'][-1]:.1%}",
            "Validation accuracy (last)": f"{history['val_accuracy'][-1]:.1%}",
            "Test accuracy": f"{test_accuracy:.1%}" if test_accuracy is not None else "not measured",
            "Weights": str(outcome.folder / WEIGHTS_FILE),
            "Passport": str(outcome.folder / "passport.json"),
            "Time": f"{outcome.seconds / 60:.1f} min",
        }
        return ToolResult(
            image=sheet,
            message=f"Trained {name}: validation {history['val_accuracy'][-1]:.1%}, "
            f"test {test_accuracy:.1%}; saved to {outcome.folder.name}/.",
            details=details,
        )

    def _ask_params(self, parent: tk.Misc | None) -> CnnTrainRequest | None:
        dialog: ParamDialog[CnnTrainRequest] = ParamDialog(
            parent,
            title=self.title,
            intro="Train a classifier on CIFAR-10 (10 classes, 32×32 images) and save its weights "
            "with a passport: dataset, class names, preprocessing and training history. "
            "Cancel stops after the current batch and keeps the last finished epoch.",
            action="Train",
        )
        frame = dialog.section("Model")
        labels = list(MODEL_CHOICES)
        current = next(label for label, key in MODEL_CHOICES.items() if key == self._architecture)
        model_var = tk.StringVar(value=current)
        ttk.Combobox(frame, textvariable=model_var, values=labels, state="readonly", width=30).grid(
            row=0, column=0, sticky="w"
        )
        model_var.trace_add("write", lambda *_args: dialog.refresh())
        variables = dialog.int_fields(dialog.section("Training"), TRAIN_FIELDS)
        device = best_device()

        def validate() -> tuple[CnnTrainRequest, str]:
            architecture = MODEL_CHOICES[model_var.get()]
            epochs, fine_tune, samples = parse_int_fields(TRAIN_FIELDS, tuple(v.get() for v in variables))
            if architecture != INCEPTION_V3:
                fine_tune = 0
            output = MODELS_DIR / architecture
            request = CnnTrainRequest(architecture, epochs, fine_tune, samples, device, output)
            seconds = estimate_seconds(request)
            parts = [f"{samples:,} training images", device]
            if seconds is not None:
                parts.append(describe_duration(seconds))
            if not cifar_downloaded():
                parts.append("downloads CIFAR-10 (170 MB) first")
            if architecture == SMALL_CNN and parse_int_fields(TRAIN_FIELDS[1:2], (variables[1].get(),))[0]:
                parts.append("fine-tune epochs apply to InceptionV3 only")
            replaces = " (replaces the saved model)" if (output / WEIGHTS_FILE).is_file() else ""
            parts.append(f"saves to data/cifar10/models/{architecture}/{replaces}")
            return request, " · ".join(parts)

        request = dialog.show(validate)
        if request is not None:
            self._architecture = request.architecture
        return request
