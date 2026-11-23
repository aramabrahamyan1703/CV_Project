"""Model passports and image preparation for the CNN tools (Set 8).

Nothing here needs PyTorch, so a passport can be read and shown, and tested,
without it. The networks, training and features live in ``cnn.py``.

A trained model is a folder with two files::

    data/cifar10/models/small_cnn/model.pt        weights (a state_dict)
    data/cifar10/models/small_cnn/passport.json   everything needed to use them

The passport (Set 8.1) records the dataset and its sample sizes, the class
names in output order, the preprocessing, and the training history.

Preprocessing. Every image, CIFAR-10 or not, is first brought to the
dataset's own size (32×32 RGB, uint8) by :func:`to_dataset_image`; the
network then resizes that to its input size, scales to 0..1 and normalizes
with the passport's mean and std. Training batches go through the same
steps, so an uploaded photo reaches InceptionV3 as the same kind of
upsampled 32-pixel image it was trained on, not as a sharp 299-pixel one.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
CIFAR_DIR = ROOT / "data" / "cifar10"
MODELS_DIR = CIFAR_DIR / "models"
WEIGHTS_FILE = "model.pt"
PASSPORT_FILE = "passport.json"
PASSPORT_FORMAT = "forensics-cnn-passport/1"

CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)  # fmt: skip

SMALL_CNN = "small_cnn"
INCEPTION_V3 = "inception_v3"
ARCHITECTURES = {
    SMALL_CNN: "Small CNN (2 conv layers, VGG-style block)",
    INCEPTION_V3: "InceptionV3, ImageNet weights, new classifier head",
}


@dataclass(frozen=True)
class Preprocessing:
    """How an RGB image becomes the network's input tensor."""

    image_size: int  # square size of the dataset images (32 for CIFAR-10)
    input_size: int  # square size the network takes (32, or 299 for InceptionV3)
    mean: tuple[float, float, float]  # per channel, after scaling to 0..1
    std: tuple[float, float, float]
    resize: str = "bilinear"  # image_size -> input_size

    def describe(self) -> str:
        steps = [f"RGB, resized to {self.image_size}×{self.image_size} (dataset size)"]
        if self.input_size != self.image_size:
            steps.append(f"{self.resize} upsampling to {self.input_size}×{self.input_size}")
        steps.append("pixels / 255")
        if any(self.mean) or any(s != 1 for s in self.std):
            steps.append(f"(x − {_triple(self.mean)}) / {_triple(self.std)} per channel")
        return " → ".join(steps)


def _triple(values: tuple[float, float, float]) -> str:
    if len(set(values)) == 1:
        return f"{values[0]:g}"
    return "(" + ", ".join(f"{v:g}" for v in values) + ")"


@dataclass
class Passport:
    architecture: str  # a key of ARCHITECTURES
    classes: tuple[str, ...]  # in the order of the network's outputs
    preprocessing: Preprocessing
    feature_dim: int  # length of the feature vector before the classifier
    dataset: dict = field(default_factory=dict)  # name, sample sizes, split, seed
    training: dict = field(default_factory=dict)  # optimizer, epochs, device, time...
    history: dict[str, list[float]] = field(default_factory=dict)  # per epoch
    test: dict = field(default_factory=dict)  # loss, accuracy on the test split
    created: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def to_json(self) -> dict:
        data = asdict(self)
        data["classes"] = list(self.classes)
        data["preprocessing"]["mean"] = list(self.preprocessing.mean)
        data["preprocessing"]["std"] = list(self.preprocessing.std)
        return {"format": PASSPORT_FORMAT, **data}

    @classmethod
    def from_json(cls, data: dict) -> Passport:
        try:
            pre = data["preprocessing"]
            return cls(
                architecture=str(data["architecture"]),
                classes=tuple(str(name) for name in data["classes"]),
                preprocessing=Preprocessing(
                    image_size=int(pre["image_size"]),
                    input_size=int(pre["input_size"]),
                    mean=_rgb(pre["mean"]),
                    std=_rgb(pre["std"]),
                    resize=str(pre.get("resize", "bilinear")),
                ),
                feature_dim=int(data["feature_dim"]),
                dataset=dict(data.get("dataset", {})),
                training=dict(data.get("training", {})),
                history={key: [float(v) for v in values] for key, values in data.get("history", {}).items()},
                test=dict(data.get("test", {})),
                created=str(data.get("created", "")),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Not a model passport ({error}).") from error


def _rgb(values) -> tuple[float, float, float]:
    red, green, blue = (float(v) for v in values)
    return red, green, blue


def passport_path(path: Path) -> Path:
    """The passport for a model folder, its ``model.pt`` or the passport itself."""
    path = Path(path)
    return path / PASSPORT_FILE if path.is_dir() else path.with_name(PASSPORT_FILE)


def save_passport(passport: Passport, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / PASSPORT_FILE
    path.write_text(json.dumps(passport.to_json(), indent=2), encoding="utf-8")
    return path


def load_passport(path: Path) -> Passport:
    """Read the passport beside a model; ``ValueError`` if it is missing or not one."""
    path = passport_path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"No {PASSPORT_FILE} beside the model in {path.parent}.") from error
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{path.name} is not a readable passport.") from error
    if not isinstance(data, dict) or data.get("format") != PASSPORT_FORMAT:
        raise ValueError(f"{path.name} is not a model passport.")
    return Passport.from_json(data)


def to_dataset_image(image: Image.Image, size: int) -> np.ndarray:
    """``image`` as the dataset stores it: RGB, ``size``×``size``, uint8 (H, W, 3).

    Transparent pixels are put on white. The image is squashed to a square,
    as CIFAR-10's own crops are square; crop it first to keep its aspect.
    """
    if image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, rgba)
    rgb = image.convert("RGB")
    if rgb.size != (size, size):
        rgb = rgb.resize((size, size), Image.Resampling.LANCZOS, reducing_gap=3.0)
    return np.array(rgb, dtype=np.uint8)  # a writable copy (torch refuses read-only arrays)


# --------------------------------------------------------------------------
# Saved feature vectors (Set 8.3)
# --------------------------------------------------------------------------

FEATURES_DIR = CIFAR_DIR / "features"

# Where a feature vector is taken: "features" is the input of the classifier
# head (the small CNN's trained 128-unit layer; InceptionV3's pooled 2048-d
# backbone output, which keeps its ImageNet weights when the backbone is
# frozen); "head" is InceptionV3's 1024-unit head layer, trained on CIFAR-10.
FEATURE_LAYERS = {
    SMALL_CNN: {"features": ("128-unit dense layer (trained)", 128)},
    INCEPTION_V3: {
        "head": ("1024-unit head layer (trained on CIFAR-10)", 1024),
        "features": ("2048-d pooled backbone (ImageNet weights)", 2048),
    },
}


@dataclass(frozen=True)
class FeatureSet:
    """Feature vectors of a dataset split, saved once so projections load them in a second."""

    features: np.ndarray  # (N, D) float32, the network's output before its classifier
    labels: np.ndarray  # (N,) int64 true class of each vector
    classes: tuple[str, ...]
    architecture: str
    model: str  # the model.pt they came from
    split: str = "test"
    layer: str = "features"  # a key of FEATURE_LAYERS[architecture]
    created: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


def features_path(architecture: str, split: str = "test", layer: str = "features") -> Path:
    """``<model>_test.npz`` for the classifier's input, ``<model>_<layer>_test.npz`` otherwise."""
    name = architecture if layer == "features" else f"{architecture}_{layer}"
    return FEATURES_DIR / f"{name}_{split}.npz"


def save_features(features: FeatureSet, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=features.features.astype(np.float32),
        labels=features.labels.astype(np.int64),
        classes=np.array(features.classes),
        architecture=np.array(features.architecture),
        model=np.array(features.model),
        split=np.array(features.split),
        layer=np.array(features.layer),
        created=np.array(features.created),
    )
    return path


def load_features(path: Path) -> FeatureSet:
    """Read a file written by :func:`save_features`; ``ValueError`` if it is not one."""
    try:
        with np.load(path, allow_pickle=False) as data:
            fields = {key: data[key] for key in data.files}
    except (OSError, ValueError) as error:
        raise ValueError(f"{Path(path).name} is not a readable feature file.") from error
    missing = {"features", "labels", "classes", "architecture"} - set(fields)
    if missing:
        raise ValueError(f"{Path(path).name} is not a feature file (missing {', '.join(sorted(missing))}).")
    features, labels = fields["features"], fields["labels"]
    if features.ndim != 2 or labels.shape != (len(features),):
        raise ValueError(f"{Path(path).name}: features {features.shape} and labels {labels.shape} do not match.")
    classes = tuple(str(name) for name in fields["classes"])
    if len(labels) and not (0 <= labels.min() and labels.max() < len(classes)):
        raise ValueError(f"{Path(path).name}: labels outside the {len(classes)} classes.")
    return FeatureSet(
        features=features.astype(np.float32),
        labels=labels.astype(np.int64),
        classes=classes,
        architecture=str(fields["architecture"]),
        model=str(fields.get("model", "")),
        split=str(fields.get("split", "test")),
        layer=str(fields.get("layer", "features")),
        created=str(fields.get("created", "")),
    )
