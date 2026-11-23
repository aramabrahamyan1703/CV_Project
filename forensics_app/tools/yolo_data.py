"""Detection datasets and box maths for the YOLO tools (Set 9).

Nothing here needs PyTorch or Ultralytics, so it is fast to test:

* COCO -> YOLO conversion (the notebook's ``coco_to_yolo`` TODO), with a
  ``data.yaml`` for Ultralytics;
* reading YOLO label files (``class x_center y_center width height``, all
  normalized to 0..1) back into pixel boxes;
* IoU and the greedy matching that turns predictions into true positives,
  false positives and missed objects for one image.

Formats. COCO stores one JSON per split with ``bbox = [x, y, width, height]``
in pixels, the top-left corner first, and category ids that need not start at
0. YOLO stores one text file per image, named like the image, in a ``labels``
folder beside ``images``, with class ids 0..N-1.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
import json
from pathlib import Path
import shutil

import numpy as np


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
COCO_FILE = "_annotations.coco.json"  # Roboflow's name, used in the notebook
DATA_YAML = "data.yaml"
# Split folder -> data.yaml key. Roboflow calls the validation split "valid".
SPLIT_KEYS = {"train": "train", "valid": "val", "val": "val", "test": "test"}


class ConversionCancelled(Exception):
    """The user pressed Cancel while the dataset was being converted."""


@dataclass(frozen=True)
class Box:
    """One box in pixels: corners ``(x1, y1)`` top-left and ``(x2, y2)`` bottom-right."""

    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 1.0  # 1 for ground truth

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


# --------------------------------------------------------------------------
# COCO -> YOLO
# --------------------------------------------------------------------------


def coco_classes(categories: list[dict]) -> list[dict]:
    """The categories that are real classes, sorted by id.

    Roboflow exports add a placeholder category that is only the
    *supercategory* of the others (``1-2-3-SmhR`` above PC, RBC and WBC); it
    never labels a box. Dropping categories that another category names as
    its supercategory removes it, and works for any COCO file.
    """
    parents = {c.get("supercategory") for c in categories if c.get("supercategory") != c["name"]}
    classes = [c for c in categories if c["name"] not in parents]
    return sorted(classes or categories, key=lambda c: c["id"])


def coco_box_to_yolo(bbox: list[float], width: int, height: int) -> tuple[float, float, float, float] | None:
    """``[x, y, w, h]`` pixels -> normalized ``(xc, yc, w, h)``, clipped to the image.

    Returns None for a box with no area left inside the image.
    """
    x, y, w, h = (float(v) for v in bbox)
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(float(width), x + w), min(float(height), y + h)
    if x2 <= x1 or y2 <= y1:
        return None
    return ((x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height)


@dataclass
class SplitSummary:
    name: str
    images: int = 0
    boxes: Counter = field(default_factory=Counter)  # class name -> count
    skipped_boxes: int = 0  # zero-area or outside the image
    missing_images: int = 0  # listed in the JSON but not on disk


@dataclass
class ConversionSummary:
    output: Path
    names: list[str]
    splits: list[SplitSummary]

    @property
    def data_yaml(self) -> Path:
        return self.output / DATA_YAML

    @property
    def total_images(self) -> int:
        return sum(s.images for s in self.splits)


def find_coco_splits(root: Path) -> list[Path]:
    """Split folders (``train``, ``valid``, ``test``…) under ``root`` that hold a COCO file.

    ``root`` itself counts as one split when it holds the file directly.
    """
    if not root.is_dir():
        raise ValueError(f"Not a folder: {root}")
    if (root / COCO_FILE).is_file():
        return [root]
    splits = [root / name for name in SPLIT_KEYS if (root / name / COCO_FILE).is_file()]
    if not splits:
        raise ValueError(
            f"No {COCO_FILE} found in {root.name} or in its train/valid/test folders."
        )
    return splits


def describe_coco_root(root: Path) -> str:
    """One line for the folder chooser: splits and image counts."""
    try:
        splits = find_coco_splits(root)
    except ValueError as error:
        return str(error)
    parts, categories = [], []
    for split in splits:
        with open(split / COCO_FILE, encoding="utf-8") as handle:
            coco = json.load(handle)
        parts.append(f"{split.name} {len(coco.get('images', []))}")
        categories = categories or coco.get("categories", [])
    names = ", ".join(c["name"] for c in coco_classes(categories))
    return f"{' · '.join(parts)} images · classes {names}"


def convert_coco_split(
    split_dir: Path,
    out_dir: Path,
    *,
    names_by_id: dict[int, tuple[int, str]],
    progress: Callable[[str], bool] | None = None,
) -> SplitSummary:
    """Write ``out_dir/images`` and ``out_dir/labels`` for one COCO split folder.

    ``names_by_id`` maps a COCO category id to ``(yolo_class, name)``; boxes
    of other categories are skipped. ``progress(text)`` is called before each
    image and may return False to cancel.
    """
    with open(split_dir / COCO_FILE, encoding="utf-8") as handle:
        coco = json.load(handle)
    images_dir = out_dir / "images"
    labels_dir = out_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    by_image: dict[int, list[dict]] = {}
    for ann in coco.get("annotations", []):
        by_image.setdefault(ann["image_id"], []).append(ann)

    summary = SplitSummary(split_dir.name)
    for info in coco.get("images", []):
        name = Path(info["file_name"]).name
        if progress is not None and not progress(f"{split_dir.name}/{name}"):
            raise ConversionCancelled
        source = split_dir / name
        if not source.is_file():
            summary.missing_images += 1
            continue
        target = images_dir / name
        if not target.exists():
            shutil.copy2(source, target)
        lines = []
        for ann in by_image.get(info["id"], []):
            mapped = names_by_id.get(ann["category_id"])
            box = coco_box_to_yolo(ann["bbox"], int(info["width"]), int(info["height"]))
            if mapped is None or box is None:
                summary.skipped_boxes += 1
                continue
            class_id, class_name = mapped
            lines.append(f"{class_id} " + " ".join(f"{v:.6f}" for v in box))
            summary.boxes[class_name] += 1
        # An image without boxes still gets an (empty) label file: a background image.
        (labels_dir / f"{Path(name).stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        summary.images += 1
    return summary


def write_data_yaml(output: Path, names: list[str], split_names: list[str]) -> Path:
    """``data.yaml`` for Ultralytics, with ``path`` set so it works from any folder.

    Strings are written JSON-quoted (valid YAML), so paths with spaces or
    non-Latin letters survive.
    """
    lines = [f"path: {json.dumps(str(output.resolve()), ensure_ascii=False)}"]
    keys = {}
    for split in split_names:
        key = SPLIT_KEYS.get(split, split)
        keys.setdefault(key, split)
    if "val" not in keys and "test" in keys:
        keys["val"] = keys["test"]  # Ultralytics needs a validation split
    if "val" not in keys and "train" in keys:
        keys["val"] = keys["train"]
    for key in ("train", "val", "test"):
        if key in keys:
            lines.append(f"{key}: {json.dumps(keys[key] + '/images', ensure_ascii=False)}")
    lines.append(f"nc: {len(names)}")
    lines.append(f"names: {json.dumps(names, ensure_ascii=False)}")
    target = output / DATA_YAML
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def convert_coco_dataset(
    root: Path,
    output: Path,
    *,
    progress: Callable[[int, int, str], bool] | None = None,
) -> ConversionSummary:
    """Convert every COCO split under ``root`` into a YOLO dataset at ``output``.

    Class ids come from the first split's categories (all Roboflow splits
    share them) so every split numbers the classes the same way.
    """
    splits = find_coco_splits(root)
    if output.resolve() == root.resolve() or root.resolve() in output.resolve().parents:
        raise ValueError("Choose an output folder outside the COCO folder.")
    cocos = []
    for split in splits:
        with open(split / COCO_FILE, encoding="utf-8") as handle:
            cocos.append(json.load(handle))
    classes = coco_classes(cocos[0].get("categories", []))
    if not classes:
        raise ValueError(f"{splits[0].name}/{COCO_FILE} lists no categories.")
    names_by_id = {c["id"]: (index, c["name"]) for index, c in enumerate(classes)}
    total = sum(len(c.get("images", [])) for c in cocos)
    done = 0

    def step(text: str) -> bool:
        nonlocal done
        keep_going = progress is None or progress(done, total, text)
        done += 1
        return keep_going

    output.mkdir(parents=True, exist_ok=True)
    single = splits == [root]
    summaries = [
        convert_coco_split(
            split, output / ("train" if single else split.name), names_by_id=names_by_id, progress=step
        )
        for split in splits
    ]
    names = [c["name"] for c in classes]
    write_data_yaml(output, names, [s.name if not single else "train" for s in summaries])
    if single:
        summaries[0].name = "train"
    return ConversionSummary(output, names, summaries)


# --------------------------------------------------------------------------
# YOLO labels
# --------------------------------------------------------------------------


def label_path_for(image_path: Path) -> Path:
    """Where YOLO keeps an image's labels: ``…/images/x.jpg`` -> ``…/labels/x.txt``.

    For an image outside an ``images`` folder, the ``.txt`` beside it.
    """
    parts = list(image_path.parts)
    for index in range(len(parts) - 2, -1, -1):
        if parts[index] == "images":
            parts[index] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def read_yolo_labels(path: Path, width: int, height: int) -> list[Box]:
    """Boxes in pixels from a YOLO label file, for an image of ``width`` × ``height``."""
    boxes = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f"{path.name}, line {number}: expected 5 numbers, found {len(fields)}.")
        class_id = int(float(fields[0]))
        xc, yc, w, h = (float(v) for v in fields[1:])
        boxes.append(
            Box(
                class_id,
                (xc - w / 2) * width,
                (yc - h / 2) * height,
                (xc + w / 2) * width,
                (yc + h / 2) * height,
            )
        )
    return boxes


def count_yolo_labels(labels_dir: Path) -> tuple[int, Counter]:
    """``(label files, Counter of class ids)`` for a YOLO ``labels`` folder."""
    counts: Counter = Counter()
    files = sorted(labels_dir.glob("*.txt"))
    for path in files:
        for line in path.read_text().splitlines():
            if line.strip():
                counts[int(float(line.split()[0]))] += 1
    return len(files), counts


def read_data_yaml(path: Path) -> dict:
    """The parsed ``data.yaml`` with its class ``names`` as a list.

    Uses PyYAML when installed, else Ultralytics' own YAML reader.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"Cannot read {path.name}: {error}") from error
    try:
        import yaml

        data = yaml.safe_load(text)
    except ImportError:
        try:
            from ultralytics.utils import YAML
        except ImportError as error:
            raise ValueError("Reading data.yaml needs PyYAML or Ultralytics (pip install ultralytics).") from error
        data = YAML.load(path)
    except Exception as error:  # yaml.YAMLError, without importing yaml at module level
        raise ValueError(f"{path.name} is not valid YAML: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} is not a dataset description.")
    names = data.get("names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    if not names or "train" not in data:
        raise ValueError(f"{path.name} needs 'train' and 'names' entries.")
    data["names"] = list(names)
    return data


def dataset_folder(data: dict, yaml_path: Path, key: str) -> Path | None:
    """The images folder a ``data.yaml`` entry points to, resolved like Ultralytics does."""
    entry = data.get(key)
    if not isinstance(entry, str):
        return None
    base = Path(data["path"]) if data.get("path") else yaml_path.parent
    if not base.is_absolute():
        base = yaml_path.parent / base
    folder = Path(entry)
    if folder.is_absolute():
        return folder
    resolved = (base / folder).resolve()
    if not resolved.exists() and entry.startswith("../"):  # Roboflow's exports, as Ultralytics reads them
        resolved = (base / entry[3:]).resolve()
    return resolved


# --------------------------------------------------------------------------
# Matching predictions to ground truth
# --------------------------------------------------------------------------


def iou_matrix(a: list[Box], b: list[Box]) -> np.ndarray:
    """``(len(a), len(b))`` intersection-over-union of every pair of boxes."""
    if not a or not b:
        return np.zeros((len(a), len(b)))
    A = np.array([[x.x1, x.y1, x.x2, x.y2] for x in a], dtype=np.float64)
    B = np.array([[x.x1, x.y1, x.x2, x.y2] for x in b], dtype=np.float64)
    ix1 = np.maximum(A[:, None, 0], B[None, :, 0])
    iy1 = np.maximum(A[:, None, 1], B[None, :, 1])
    ix2 = np.minimum(A[:, None, 2], B[None, :, 2])
    iy2 = np.minimum(A[:, None, 3], B[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1])
    area_b = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


@dataclass(frozen=True)
class MatchResult:
    """Outcome of matching one image's predictions to its ground truth."""

    matched: tuple[tuple[int, int, float], ...]  # (prediction, ground truth, IoU)
    false_positives: tuple[int, ...]  # prediction indices
    missed: tuple[int, ...]  # ground-truth indices

    @property
    def tp(self) -> int:
        return len(self.matched)

    @property
    def precision(self) -> float | None:
        n = self.tp + len(self.false_positives)
        return self.tp / n if n else None

    @property
    def recall(self) -> float | None:
        n = self.tp + len(self.missed)
        return self.tp / n if n else None


def match_boxes(predictions: list[Box], truth: list[Box], iou_threshold: float = 0.5) -> MatchResult:
    """Greedy matching as in VOC/COCO evaluation.

    Predictions are taken from the most to the least confident; each takes the
    unmatched ground-truth box of the same class it overlaps most, if that IoU
    reaches ``iou_threshold``. Anything left over is a false positive; ground
    truth nobody took was missed.
    """
    ious = iou_matrix(predictions, truth)
    taken = np.zeros(len(truth), dtype=bool)
    matched, false_positives = [], []
    for p in sorted(range(len(predictions)), key=lambda i: -predictions[i].score):
        best, best_iou = -1, iou_threshold
        for g, box in enumerate(truth):
            if taken[g] or box.class_id != predictions[p].class_id:
                continue
            if ious[p, g] >= best_iou:
                best, best_iou = g, float(ious[p, g])
        if best < 0:
            false_positives.append(p)
        else:
            taken[best] = True
            matched.append((p, best, best_iou))
    missed = tuple(int(g) for g in np.flatnonzero(~taken))
    return MatchResult(tuple(matched), tuple(sorted(false_positives)), missed)


def class_counts(boxes: list[Box], names: list[str]) -> str:
    """``"1 PC, 17 RBC, 1 WBC"``, in class order, for titles and details."""
    counts = Counter(b.class_id for b in boxes)
    if not counts:
        return "none"
    return ", ".join(
        f"{counts[i]} {names[i] if 0 <= i < len(names) else f'class {i}'}" for i in sorted(counts)
    )


def list_images(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.startswith(".")
    )
