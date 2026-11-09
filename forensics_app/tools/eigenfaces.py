"""Eigenfaces: build a PCA face model from a folder and recognise faces with it (Sets 6.2, 6.3).

The model follows the slides' formulas. Training faces x_i (grayscale, one
size, flattened) give the mean face mu and the eigenfaces U_K, the first K
principal components; each face is stored as its coefficients
w_i = U_K^T (x_i - mu). A new face is projected the same way and recognised as
the training face nearest to it, i* = argmin_i ||w_test - w_i||_2.

Whitening is optional and on by default: distances are measured after
dividing each coefficient by its eigenface's spread over the training set, so
the few directions with the most variance (mostly lighting) do not dominate.
On the course's LFW split, nearest-neighbour accuracy with 100 eigenfaces was
69% whitened and 55% without (always guessing the most common person: 45%).

A training folder holds one subfolder per person, named after them
(``George_W_Bush/0003.png``); ``scripts/export_lfw.py`` writes LFW that way.
The model is saved as one ``.npz`` file with the mean, the eigenfaces, every
training face's coefficients, name and pixels, so recognition needs neither
the training folder nor a rebuild.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".pgm"}
MODEL_FILE = "eigenfaces_model.npz"


def person_name(path: Path, folder: Path) -> str:
    """The person a face belongs to: its subfolder's name, underscores as spaces."""
    relative = path.relative_to(folder)
    label = relative.parts[0] if len(relative.parts) > 1 else path.stem
    return label.replace("_", " ")


def list_faces(folder: Path) -> list[Path]:
    """Image files in ``folder`` and its subfolders, sorted, without hidden ones."""
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and not any(part.startswith(".") for part in path.relative_to(folder).parts)
    )


def to_face_vector(image: Image.Image, shape: tuple[int, int]) -> np.ndarray:
    """Grayscale pixels of ``image`` at ``shape`` (height, width), flattened, 0..255."""
    height, width = shape
    gray = image.convert("L")
    if gray.size != (width, height):
        gray = gray.resize((width, height), Image.Resampling.LANCZOS)
    return np.asarray(gray, dtype=np.float64).ravel()


@dataclass(frozen=True)
class FaceSet:
    files: tuple[Path, ...]
    names: tuple[str, ...]
    pixels: np.ndarray  # (N, height*width) float64, 0..255
    shape: tuple[int, int]  # (height, width)
    skipped: tuple[tuple[Path, str], ...] = ()


def load_faces(folder: Path, shape: tuple[int, int] | None = None) -> FaceSet:
    """All faces in ``folder`` as vectors of one size.

    The size is ``shape`` if given, otherwise the most common image size in the
    folder (LFW faces are all 37×50); other images are resized to it.
    Unreadable files are skipped with a reason.
    """
    files = list_faces(folder)
    images: list[tuple[Path, Image.Image]] = []
    skipped: list[tuple[Path, str]] = []
    for path in files:
        try:
            with Image.open(path) as opened:
                images.append((path, opened.convert("L")))
        except OSError:
            skipped.append((path, "not a readable image"))
    if not images:
        raise ValueError(f"No face images found in {folder}.")
    if shape is None:
        width, height = Counter(image.size for _, image in images).most_common(1)[0][0]
        shape = (height, width)
    pixels = np.stack([to_face_vector(image, shape) for _, image in images])
    return FaceSet(
        tuple(path for path, _ in images),
        tuple(person_name(path, folder) for path, _ in images),
        pixels,
        shape,
        tuple(skipped),
    )


@dataclass(frozen=True)
class EigenfaceModel:
    mean: np.ndarray  # (D,)
    eigenfaces: np.ndarray  # (K, D), orthonormal rows, most variance first
    coefficients: np.ndarray  # (N, K): w_i = U_K^T (x_i - mean)
    names: tuple[str, ...]  # (N,) person of each training face
    files: tuple[str, ...]  # (N,) training file, relative to the training folder
    faces: np.ndarray  # (N, D) uint8 training pixels, for showing neighbours
    shape: tuple[int, int]  # (height, width)
    explained: np.ndarray  # (K,) share of the variance each eigenface explains
    spread: np.ndarray  # (K,) standard deviation of each coefficient over the training faces
    source: str  # the training folder

    @property
    def components(self) -> int:
        return len(self.eigenfaces)

    @property
    def people(self) -> list[str]:
        return sorted(set(self.names))


def build_model(faces: FaceSet, components: int, source: Path | str = "") -> EigenfaceModel:
    """PCA of the training faces by SVD, keeping the first ``components`` eigenfaces.

    Each eigenface's sign is chosen as scikit-learn does (its largest entry
    positive), so the grid looks like the slide's rather than its negative.
    """
    count, dims = faces.pixels.shape
    if count < 2:
        raise ValueError("At least 2 face images are needed to build eigenfaces.")
    limit = min(count, dims)
    if not 1 <= components <= limit:
        raise ValueError(f"Components must be between 1 and {limit} for {count} faces.")
    mean = faces.pixels.mean(axis=0)
    centred = faces.pixels - mean
    _, singular, rows = np.linalg.svd(centred, full_matrices=False)
    largest = np.argmax(np.abs(rows), axis=1)
    rows = rows * np.sign(rows[np.arange(len(rows)), largest])[:, None]
    variance = singular**2
    eigenfaces = rows[:components]
    spread = singular[:components] / np.sqrt(count - 1)
    folder = Path(source) if source else None
    return EigenfaceModel(
        mean=mean,
        eigenfaces=eigenfaces,
        coefficients=centred @ eigenfaces.T,
        names=faces.names,
        files=tuple(
            path.relative_to(folder).as_posix() if folder else path.name for path in faces.files
        ),
        faces=np.clip(np.rint(faces.pixels), 0, 255).astype(np.uint8),
        shape=faces.shape,
        explained=variance[:components] / variance.sum(),
        spread=spread,
        source=str(source),
    )


def save_model(model: EigenfaceModel, path: Path) -> None:
    np.savez_compressed(
        path,
        mean=model.mean,
        eigenfaces=model.eigenfaces,
        coefficients=model.coefficients,
        names=np.array(model.names),
        files=np.array(model.files),
        faces=model.faces,
        shape=np.array(model.shape),
        explained=model.explained,
        spread=model.spread,
        source=np.array(model.source),
    )


def load_model(path: Path) -> EigenfaceModel:
    """Read a model written by :func:`save_model`; ``ValueError`` if it is not one."""
    try:
        with np.load(path, allow_pickle=False) as data:
            fields = {key: data[key] for key in data.files}
    except (OSError, ValueError) as error:
        raise ValueError(f"{path.name} is not a readable eigenface model.") from error
    missing = {
        "mean", "eigenfaces", "coefficients", "names", "files", "faces", "shape", "explained"
    } - set(fields)
    if missing:
        raise ValueError(f"{path.name} is not an eigenface model (missing {', '.join(sorted(missing))}).")
    height, width = (int(v) for v in fields["shape"])
    return EigenfaceModel(
        mean=fields["mean"],
        eigenfaces=fields["eigenfaces"],
        coefficients=fields["coefficients"],
        names=tuple(str(name) for name in fields["names"]),
        files=tuple(str(name) for name in fields["files"]),
        faces=fields["faces"],
        shape=(height, width),
        explained=fields["explained"],
        spread=fields["spread"] if "spread" in fields else fields["coefficients"].std(axis=0, ddof=1),
        source=str(fields.get("source", "")),
    )


def project(model: EigenfaceModel, image: Image.Image, components: int | None = None) -> np.ndarray:
    """w = U_K^T (x - mu) for ``image``, converted to the model's grayscale size."""
    k = model.components if components is None else components
    return model.eigenfaces[:k] @ (to_face_vector(image, model.shape) - model.mean)


def distances(model: EigenfaceModel, weights: np.ndarray, whiten: bool = True) -> np.ndarray:
    """d_i = ||w_test - w_i||_2 to every training face, using the first len(weights) eigenfaces.

    With ``whiten`` both sides are divided by each coefficient's spread first.
    """
    k = len(weights)
    train = model.coefficients[:, :k]
    if whiten:
        scale = np.where(model.spread[:k] > 0, model.spread[:k], 1.0)
        return np.linalg.norm(train / scale - weights / scale, axis=1)
    return np.linalg.norm(train - weights, axis=1)


def nearest(
    model: EigenfaceModel, weights: np.ndarray, count: int = 3, whiten: bool = True
) -> list[tuple[int, float]]:
    """The ``count`` training faces closest to ``weights``: (index, distance), nearest first."""
    found = distances(model, weights, whiten)
    order = np.argsort(found, kind="stable")[:count]
    return [(int(index), float(found[index])) for index in order]
