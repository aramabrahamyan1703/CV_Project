# ForensicsApp course starter

ForensicsApp is the semester project shell for the MSc Computer Vision course.  
Students add a small, demonstrable image-forensics feature as the course moves  
from classical image processing to deep learning and interpretability.

The starter already provides:

- a Tkinter desktop interface with tool, image, and results areas;
- opening and exporting common image formats;
- a fit-to-window image preview;
- undo, redo, and reset-to-original behavior;
- a tiny plug-in-style contract for adding course functionality;
- one analysis example and one image-processing example.

## Quick start

Python 3.10 or newer is recommended.

```
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python run.py
```

On some Linux distributions, Tkinter is a separate system package. For example,  
Ubuntu/Debian users may first need `sudo apt install python3-tk`.

The complete student handout is [docs/STUDENT_GUIDE.md](docs/STUDENT_GUIDE.md).

## Useful commands

```
python -m forensics_app
python -m unittest discover -s tests -v
python -m compileall forensics_app
```

## Face tools (Set 6): model and data

These are not in the repository; set them up once.

**YuNet face detector** (Face detection, Face detection (webcam)). Download
`face_detection_yunet_2023mar.onnx` from the
[OpenCV model zoo](https://huggingface.co/opencv/opencv_zoo/resolve/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx)
into `models/`. The dialogs can also point at a copy anywhere else.

**LFW faces** (Eigenface builder, Face recognition, Recognition accuracy):

```
python scripts/export_lfw.py
```

downloads Labeled Faces in the Wild with scikit-learn (about 200 MB, once) and
writes the course notebook's split into `data/lfw/`: 966 train and 322 test
faces of 7 people, one folder per person (`train/George_W_Bush/0003.png`), plus
`labels.csv`. Build the model with *Eigenface builder* on `data/lfw/train`; it
saves `data/lfw/eigenfaces_model.npz`, which the recognition tools load. Expect
about 69% test accuracy with the defaults (100 eigenfaces, whitened distances;
always answering the most common person scores 45%).

**Webcam on macOS.** Allow camera access for the app you start ForensicsApp
from (Terminal, iTerm, VS Code…) in System Settings → Privacy & Security →
Camera, then restart that app.

## Detection tools (Set 9): YOLO with Ultralytics

The four **Detection** tools need Ultralytics, which installs PyTorch (about
500 MB); it is in `requirements.txt`. Without it the rest of the app still
works and the YOLO tools say how to install it.

1. **YOLO object detection** (9.1) runs the pretrained `yolo11n.pt` (80 COCO
   classes). It is downloaded into `models/` (5 MB) the first time; any
   detection `.pt` file can be chosen instead.
2. **COCO → YOLO dataset** (9.2) converts a Roboflow COCO export (`train/`,
   `valid/`, `test/`, each with `_annotations.coco.json`) into a sibling
   `<folder>_yolo/` with `images/`, `labels/` and a `data.yaml`.
3. **YOLO training** (9.2) fine-tunes `yolo11n.pt` (or trains from scratch)
   on a `data.yaml`, on Apple MPS or CUDA when available. Runs are saved in
   `runs/` beside the `data.yaml`; Cancel stops early and keeps `best.pt`.
4. **YOLO test vs ground truth** (9.3) runs trained weights on a test image
   and compares its boxes with `…/labels/<image>.txt`: correct, false and
   missed boxes, precision and recall.

**Blood-cell data.** Unzip `blood_cell.v1i.coco.zip` from the Lab 9 materials
(about 170 MB) somewhere outside the repository, for example
`data/blood_cell/coco` (ignored by git), and point *COCO → YOLO dataset* at it.
On an Apple M1, one epoch over all 3,027 training images takes about 4
minutes; fine-tuning for 5 epochs on 30 % of them (*Data used* = 30) reached
mAP50 0.96 on the validation split.

## Course roadmap

The shell is intentionally neutral about the algorithms. Possible tool modules  
can follow the syllabus: filtering and edges, HOG, SIFT, bag of visual words,  
face detection/recognition, CNN classification and detection, transformers,  
semantic segmentation, and visualization/interpretability. Model weights and  
datasets should not be committed to the starter repository; document where they  
come from and keep paths configurable.
