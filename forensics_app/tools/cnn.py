"""CNN classifiers for Set 8: build, train, evaluate, save, load, extract features.

Two architectures from the course notebook, in PyTorch instead of Keras:

* ``small_cnn``: the notebook's "deeper" VGG-style model: two 3×3 conv
  layers of 32 filters (same padding), 2×2 max pooling, a 128-unit dense
  layer and the classifier, on 32×32 inputs, pixels scaled to 0..1.
* ``inception_v3``: torchvision's InceptionV3 with ImageNet weights, the
  1000-class layer replaced by the notebook's head (1024-unit dense layer,
  then the classifier), on 299×299 inputs scaled to -1..1 (what the Google
  weights expect; Keras' ``inception_v3.preprocess_input`` does the same).

Every model is ``features`` (image -> feature vector: 128 or 2048 numbers)
followed by ``classifier``; ``features`` is what Set 8.3 projects with t-SNE
and UMAP. With a frozen backbone, training first runs ``features`` once over
the data (in eval mode, so BatchNorm statistics stay ImageNet's) and then
trains only the head on those vectors, which is the same computation but
needs one pass of InceptionV3 instead of one per epoch.

Images are passed around as uint8 arrays at the dataset's size (N, 32, 32, 3)
and turned into network input on the device by :func:`to_input`, the one
place preprocessing happens, for training and for classifying alike.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .cnn_data import (
    ARCHITECTURES,
    CIFAR_DIR,
    INCEPTION_V3,
    SMALL_CNN,
    WEIGHTS_FILE,
    Passport,
    Preprocessing,
    load_passport,
    passport_path,
    save_passport,
)


Progress = Callable[[str, int, int], None]  # (stage, done, total)
Cancelled = Callable[[], bool]


def best_device() -> str:
    """``cuda`` with an NVIDIA GPU, ``mps`` on Apple silicon, else ``cpu``."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


class FeatureClassifier(nn.Module):
    """``classifier(features(x))``, with the two parts reachable on their own."""

    features: nn.Module
    classifier: nn.Module
    feature_dim: int

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class SmallCNN(FeatureClassifier):
    feature_dim = 128

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, self.feature_dim), nn.ReLU(),
        )  # fmt: skip
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        for layer in self.modules():  # Keras' he_uniform with zero biases, as in the notebook
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)


class InceptionTransfer(FeatureClassifier):
    feature_dim = 2048

    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        from torchvision.models import Inception_V3_Weights, inception_v3

        # One constructor for training and loading: with weights torchvision
        # would switch transform_input on unless told otherwise, and requires
        # the auxiliary classifier, which is dropped here either way.
        net = inception_v3(
            weights=Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None,
            aux_logits=pretrained,
            transform_input=False,
            init_weights=False,
        )
        net.aux_logits = False
        net.AuxLogits = None
        net.fc = nn.Identity()  # type: ignore[assignment]  # pooled 2048-d vector (after the net's dropout)
        self.features = net
        self.classifier = nn.Sequential(nn.Linear(2048, 1024), nn.ReLU(), nn.Linear(1024, num_classes))


def preprocessing_for(architecture: str) -> Preprocessing:
    if architecture == SMALL_CNN:
        return Preprocessing(image_size=32, input_size=32, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))
    if architecture == INCEPTION_V3:
        return Preprocessing(image_size=32, input_size=299, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    raise ValueError(f"Unknown architecture {architecture!r}; expected one of {', '.join(ARCHITECTURES)}.")


def build_model(
    architecture: str, num_classes: int, pretrained: bool = True, seed: int | None = None
) -> FeatureClassifier:
    """A new model; ``pretrained`` loads InceptionV3's ImageNet weights (downloaded once).

    ``seed`` fixes the random initial weights (the small CNN, Inception's new head).
    """
    if seed is not None:
        torch.manual_seed(seed)
    if architecture == SMALL_CNN:
        return SmallCNN(num_classes)
    if architecture == INCEPTION_V3:
        return InceptionTransfer(num_classes, pretrained)
    raise ValueError(f"Unknown architecture {architecture!r}; expected one of {', '.join(ARCHITECTURES)}.")


def to_input(images: torch.Tensor, pre: Preprocessing) -> torch.Tensor:
    """uint8 images (N, H, W, 3) -> float32 network input (N, 3, S, S) on the same device."""
    x = images.permute(0, 3, 1, 2).to(torch.float32) / 255.0
    if x.shape[-2:] != (pre.input_size, pre.input_size):
        x = F.interpolate(x, size=(pre.input_size, pre.input_size), mode=pre.resize, align_corners=False)
    mean = torch.tensor(pre.mean, dtype=torch.float32, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(pre.std, dtype=torch.float32, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ImageSet:
    images: np.ndarray  # (N, H, W, 3) uint8
    labels: np.ndarray  # (N,) int64 class indices

    def __len__(self) -> int:
        return len(self.labels)

    def subset(self, indices: np.ndarray) -> ImageSet:
        return ImageSet(self.images[indices], self.labels[indices])


def load_cifar10(root: Path = CIFAR_DIR, download: bool = True) -> tuple[ImageSet, ImageSet]:
    """CIFAR-10's 50,000 training and 10,000 test images (torchvision checks the download's MD5)."""
    from torchvision.datasets import CIFAR10

    sets = []
    for train in (True, False):
        data = CIFAR10(str(root), train=train, download=download)
        sets.append(ImageSet(np.asarray(data.data, dtype=np.uint8), np.asarray(data.targets, dtype=np.int64)))
    return sets[0], sets[1]


def split_train_val(
    data: ImageSet,
    val_fraction: float = 0.1,
    train_samples: int | None = None,
    val_samples: int | None = None,
    seed: int = 0,
) -> tuple[ImageSet, ImageSet]:
    """The last ``val_fraction`` of ``data`` for validation (as Keras' validation_split does),
    the rest for training; each optionally cut to a random subset of the given size."""
    count = len(data)
    cut = count - int(round(count * val_fraction))
    rng = np.random.default_rng(seed)

    def pick(indices: np.ndarray, size: int | None) -> np.ndarray:
        if size is None or size >= len(indices):
            return indices
        return np.sort(rng.choice(indices, size=size, replace=False))

    return data.subset(pick(np.arange(cut), train_samples)), data.subset(pick(np.arange(cut, count), val_samples))


def _batches(count: int, batch_size: int):
    for start in range(0, count, batch_size):
        yield slice(start, min(start + batch_size, count))


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


class TrainingCancelled(Exception):
    """Cancel was pressed before the first epoch finished."""


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    batch_size: int = 64
    optimizer: str = "sgd"  # "sgd" or "adam"
    learning_rate: float = 0.01
    momentum: float = 0.9  # SGD only
    freeze_backbone: bool = False  # train only the classifier head
    seed: int = 0
    device: str = "cpu"

    def describe(self) -> dict:
        info = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "frozen_backbone": self.freeze_backbone,
            "seed": self.seed,
            "device": self.device,
        }
        if self.optimizer == "sgd":
            info["momentum"] = self.momentum
        return info


def default_config(architecture: str, epochs: int, device: str, fine_tune: bool = False) -> TrainConfig:
    """The notebook's choices: SGD for the small CNN, Adam 1e-3 for InceptionV3's
    head with the backbone frozen, Adam 1e-4 to fine-tune all of it."""
    if architecture == SMALL_CNN:
        return TrainConfig(epochs, optimizer="sgd", learning_rate=0.01, device=device)
    if fine_tune:
        return TrainConfig(epochs, batch_size=32, optimizer="adam", learning_rate=1e-4, device=device)
    return TrainConfig(epochs, optimizer="adam", learning_rate=1e-3, freeze_backbone=True, device=device)


@dataclass(frozen=True)
class TrainResult:
    history: dict[str, list[float]]  # loss, accuracy, val_loss, val_accuracy per epoch
    epochs_done: int
    cancelled: bool
    seconds: float


def train(
    model: FeatureClassifier,
    pre: Preprocessing,
    train_set: ImageSet,
    val_set: ImageSet,
    config: TrainConfig,
    progress: Progress | None = None,
    cancelled: Cancelled | None = None,
) -> TrainResult:
    """Train ``model`` in place. Cancel stops after the current batch and puts
    the weights back to the last finished epoch, so they always match the
    history (``TrainingCancelled``, weights as before, if none finished)."""
    started = time.perf_counter()
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    model.to(device)
    kept = _snapshot(model)
    for parameter in model.features.parameters():
        parameter.requires_grad = not config.freeze_backbone
    trainable = [p for p in model.parameters() if p.requires_grad]
    if config.optimizer == "adam":
        optimizer = torch.optim.Adam(trainable, lr=config.learning_rate)
    elif config.optimizer == "sgd":
        optimizer = torch.optim.SGD(trainable, lr=config.learning_rate, momentum=config.momentum)
    else:
        raise ValueError(f"Unknown optimizer {config.optimizer!r}.")

    if config.freeze_backbone:  # the head learns from vectors computed once
        head = model.classifier
        train_x = _feature_tensor(model, pre, train_set, config, device, progress, cancelled, "training")
        val_x = _feature_tensor(model, pre, val_set, config, device, progress, cancelled, "validation")

        def forward(index: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
            return head(inputs[index.to(device)])

        def set_training(on: bool) -> None:
            model.eval()
            head.train(on)

    else:
        train_x = torch.from_numpy(train_set.images)
        val_x = torch.from_numpy(val_set.images)

        def forward(index: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
            return model(to_input(inputs[index].to(device), pre))

        def set_training(on: bool) -> None:
            model.train(on)

    train_y = torch.from_numpy(train_set.labels).to(device)
    val_y = torch.from_numpy(val_set.labels).to(device)
    generator = torch.Generator().manual_seed(config.seed)
    history: dict[str, list[float]] = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    batches_per_epoch = -(-len(train_set) // config.batch_size)

    for epoch in range(config.epochs):
        set_training(True)
        order = torch.randperm(len(train_set), generator=generator)
        total_loss, correct = 0.0, 0
        for step, batch in enumerate(_batches(len(train_set), config.batch_size)):
            if cancelled is not None and cancelled():
                return _stopped(model, kept, history, started)
            index = order[batch]
            labels = train_y[index.to(device)]
            logits = forward(index, train_x)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(labels)
            correct += int((logits.argmax(1) == labels).sum())
            if progress is not None:
                progress(f"Epoch {epoch + 1}/{config.epochs}", step + 1, batches_per_epoch)
        set_training(False)
        scored = _score(
            lambda index: forward(index, val_x), val_y, config.batch_size, progress, cancelled,
            f"Epoch {epoch + 1}/{config.epochs}: validating",
        )  # fmt: skip
        if scored is None:
            return _stopped(model, kept, history, started)
        val_loss, val_accuracy = scored
        history["loss"].append(total_loss / len(train_set))
        history["accuracy"].append(correct / len(train_set))
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        kept = _snapshot(model)
    return TrainResult(history, config.epochs, False, time.perf_counter() - started)


def _snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def _stopped(model: nn.Module, kept: dict[str, torch.Tensor], history: dict[str, list[float]], started: float) -> TrainResult:
    model.load_state_dict(kept)
    if not history["loss"]:
        raise TrainingCancelled
    return TrainResult(history, len(history["loss"]), True, time.perf_counter() - started)


def _feature_tensor(model, pre, data, config, device, progress, cancelled, name) -> torch.Tensor:
    features = extract_features(
        model, pre, data.images, config.batch_size, config.device, progress, cancelled, f"Features of the {name} images"
    )
    if features is None:
        raise TrainingCancelled
    return torch.from_numpy(features).to(device)


@torch.no_grad()
def _score(forward, labels: torch.Tensor, batch_size: int, progress, cancelled, stage: str) -> tuple[float, float] | None:
    """(mean loss, accuracy) of ``forward(index)`` against ``labels``; ``None`` if cancelled.

    ``forward`` gets the batch's positions as a CPU index tensor."""
    total_loss, correct = 0.0, 0
    total = -(-len(labels) // batch_size)
    for step, batch in enumerate(_batches(len(labels), batch_size)):
        if cancelled is not None and cancelled():
            return None
        logits = forward(torch.arange(batch.start, batch.stop))
        total_loss += float(F.cross_entropy(logits, labels[batch], reduction="sum"))
        correct += int((logits.argmax(1) == labels[batch]).sum())
        if progress is not None:
            progress(stage, step + 1, total)
    return total_loss / len(labels), correct / len(labels)


def evaluate(
    model: FeatureClassifier,
    pre: Preprocessing,
    data: ImageSet,
    batch_size: int = 64,
    device: str = "cpu",
    progress: Progress | None = None,
    cancelled: Cancelled | None = None,
) -> tuple[float, float] | None:
    """(mean cross-entropy, accuracy) on ``data``; ``None`` if cancelled."""
    model.to(device).eval()
    images = torch.from_numpy(data.images)
    labels = torch.from_numpy(data.labels).to(device)
    return _score(
        lambda index: model(to_input(images[index].to(device), pre)), labels, batch_size, progress, cancelled, "Testing"
    )


# --------------------------------------------------------------------------
# Using a trained model
# --------------------------------------------------------------------------


@torch.no_grad()
def extract_features(
    model: FeatureClassifier,
    pre: Preprocessing,
    images: np.ndarray,
    batch_size: int = 64,
    device: str = "cpu",
    progress: Progress | None = None,
    cancelled: Cancelled | None = None,
    stage: str = "Extracting features",
    layer: str = "features",
) -> np.ndarray | None:
    """Feature vectors (N, D) float32 of uint8 ``images``; ``None`` if cancelled.

    ``layer="features"`` gives the classifier's input (``feature_dim`` numbers);
    ``"head"`` runs the classifier head up to its last layer (InceptionV3: 1024).
    """
    model.to(device).eval()
    embed = _layer(model, layer)
    out: np.ndarray | None = None
    total = -(-len(images) // batch_size)
    for step, batch in enumerate(_batches(len(images), batch_size)):
        if cancelled is not None and cancelled():
            return None
        x = to_input(torch.from_numpy(images[batch]).to(device), pre)
        vectors = embed(x).cpu().numpy()
        if out is None:
            out = np.empty((len(images), vectors.shape[1]), dtype=np.float32)
        out[batch] = vectors
        if progress is not None:
            progress(stage, step + 1, total)
    return out if out is not None else np.empty((0, model.feature_dim), dtype=np.float32)


def _layer(model: FeatureClassifier, layer: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if layer == "features":
        return model.features
    if layer == "head" and isinstance(model.classifier, nn.Sequential) and len(model.classifier) > 1:
        hidden = model.classifier[:-1]
        return lambda x: hidden(model.features(x))
    raise ValueError(f"A {type(model).__name__} has no {layer!r} layer to take features from.")


@torch.no_grad()
def predict_proba(
    model: FeatureClassifier, pre: Preprocessing, images: np.ndarray, device: str = "cpu"
) -> np.ndarray:
    """Class probabilities (N, classes) of uint8 ``images`` (N, H, W, 3)."""
    model.to(device).eval()
    logits = model(to_input(torch.from_numpy(np.ascontiguousarray(images)).to(device), pre))
    return F.softmax(logits, dim=1).cpu().numpy()


def passport_for(
    architecture: str,
    classes: tuple[str, ...],
    train_set: ImageSet,
    val_set: ImageSet,
    test_set: ImageSet | None,
    stages: list[tuple[TrainConfig, TrainResult]],
    test_score: tuple[float, float] | None,
    dataset: str = "CIFAR-10",
) -> Passport:
    """The passport of a finished training run (Set 8.1): dataset and sample
    sizes, class names, preprocessing, training settings and history, test score.

    ``stages`` are the runs of :func:`train` in order (InceptionV3: the head
    alone, then optionally fine-tuning everything); their histories are joined.
    """
    import torchvision

    history: dict[str, list[float]] = {}
    for _, result in stages:
        for key, values in result.history.items():
            history.setdefault(key, []).extend(values)
    return Passport(
        architecture=architecture,
        classes=tuple(classes),
        preprocessing=preprocessing_for(architecture),
        feature_dim=SmallCNN.feature_dim if architecture == SMALL_CNN else InceptionTransfer.feature_dim,
        dataset={
            "name": dataset,
            "train_samples": len(train_set),
            "val_samples": len(val_set),
            "test_samples": 0 if test_set is None else len(test_set),
            "split": "validation = the last 10% of the training set; test = the official test set",
            "seed": stages[0][0].seed if stages else None,
        },
        training={
            "stages": [
                {**config.describe(), "epochs_done": result.epochs_done, "cancelled": result.cancelled,
                 "seconds": round(result.seconds, 1)}
                for config, result in stages
            ],
            "epochs_done": sum(result.epochs_done for _, result in stages),
            "cancelled": any(result.cancelled for _, result in stages),
            "seconds": round(sum(result.seconds for _, result in stages), 1),
            "framework": f"PyTorch {torch.__version__}, torchvision {torchvision.__version__}",
        },
        history=history,
        test={} if test_score is None else {"loss": test_score[0], "accuracy": test_score[1]},
    )


def save_model(model: FeatureClassifier, passport: Passport, folder: Path) -> Path:
    """Write ``model.pt`` (the state_dict, on the CPU) and ``passport.json`` to ``folder``."""
    folder.mkdir(parents=True, exist_ok=True)
    weights = folder / WEIGHTS_FILE
    torch.save({name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}, weights)
    save_passport(passport, folder)
    return weights


def load_model(path: Path, device: str = "cpu") -> tuple[FeatureClassifier, Passport]:
    """The model saved in a folder (or its ``model.pt``) with its passport; ``ValueError`` if unusable."""
    passport = load_passport(path)
    weights = passport_path(path).with_name(WEIGHTS_FILE)
    model = build_model(passport.architecture, len(passport.classes), pretrained=False)
    try:
        state = torch.load(weights, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    except FileNotFoundError as error:
        raise ValueError(f"No {WEIGHTS_FILE} beside the passport in {weights.parent}.") from error
    except (RuntimeError, OSError, ValueError) as error:
        raise ValueError(f"{weights.name} does not hold weights for a {passport.architecture} model ({error}).") from error
    return model.to(device).eval(), passport
