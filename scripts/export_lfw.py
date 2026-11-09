"""Export the LFW faces to train/test folders for the eigenface tools (Set 6.2).

    python scripts/export_lfw.py            # writes data/lfw/

Downloads "Labeled Faces in the Wild" once with scikit-learn's
``fetch_lfw_people`` (about 200 MB, cached in ~/scikit_learn_data) using the
course notebook's settings: people with at least 70 photos, images scaled by
0.4 (50×37 px), and a 75/25 train/test split with ``random_state=42``. The
result is the notebook's exact split: 966 train and 322 test faces of 7 people.

Every face is saved as an 8-bit grayscale PNG in a folder named after its
person, which is how the app pairs images with names::

    data/lfw/train/George_W_Bush/0003.png
    data/lfw/test/Tony_Blair/0412.png
    data/lfw/labels.csv          split,file,name,sample

The four-digit number is the face's index in the LFW array, so any file can
be traced back to the dataset. ``labels.csv`` lists the same pairs for use
outside the app.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "lfw", help="output folder")
    parser.add_argument("--min-faces", type=int, default=70, help="photos needed per person")
    parser.add_argument("--resize", type=float, default=0.4, help="scale of the 250 px LFW crops")
    parser.add_argument("--test-size", type=float, default=0.25, help="share kept for testing")
    parser.add_argument("--seed", type=int, default=42, help="random_state of the split")
    parser.add_argument(
        "--overwrite", action="store_true", help="replace train/, test/ and labels.csv in --out"
    )
    args = parser.parse_args(argv)

    try:
        from sklearn.datasets import fetch_lfw_people
        from sklearn.model_selection import train_test_split
    except ImportError:
        print("scikit-learn is needed: python -m pip install scikit-learn", file=sys.stderr)
        return 1

    out: Path = args.out
    existing = [out / "train", out / "test", out / "labels.csv"]
    if any(path.exists() for path in existing):
        if not args.overwrite:
            print(f"{out} already has an export; pass --overwrite to replace it.", file=sys.stderr)
            return 1
        for path in existing:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()

    print("Loading LFW (downloads about 200 MB the first time)…", flush=True)
    lfw = fetch_lfw_people(min_faces_per_person=args.min_faces, resize=args.resize)
    images, targets, names = lfw.images, lfw.target, lfw.target_names
    # fetch_lfw_people gives float pixels; older releases use 0..255, newer 0..1.
    scale = 255.0 if images.max() <= 1.0 else 1.0
    pixels = np.clip(np.rint(images * scale), 0, 255).astype(np.uint8)

    indices = np.arange(len(images))
    train_idx, test_idx = train_test_split(
        indices, test_size=args.test_size, random_state=args.seed
    )

    rows = []
    for split, chosen in (("train", train_idx), ("test", test_idx)):
        for index in sorted(chosen):
            name = str(names[targets[index]])
            folder = out / split / name.replace(" ", "_")
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{index:04d}.png"
            Image.fromarray(pixels[index], mode="L").save(path)
            rows.append((split, path.relative_to(out).as_posix(), name, int(index)))

    with open(out / "labels.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("split", "file", "name", "sample"))
        writer.writerows(rows)

    height, width = pixels.shape[1:]
    print(
        f"Saved {len(train_idx)} train and {len(test_idx)} test faces of {len(names)} people "
        f"({width}×{height} px) to {out}"
    )
    for number, name in enumerate(names):
        train = int(np.sum(targets[train_idx] == number))
        test = int(np.sum(targets[test_idx] == number))
        print(f"  {name:20s} {train:4d} train  {test:4d} test")
    return 0


if __name__ == "__main__":
    sys.exit(main())
