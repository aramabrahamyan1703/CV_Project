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

## Course roadmap

The shell is intentionally neutral about the algorithms. Possible tool modules  
can follow the syllabus: filtering and edges, HOG, SIFT, bag of visual words,  
face detection/recognition, CNN classification and detection, transformers,  
semantic segmentation, and visualization/interpretability. Model weights and  
datasets should not be committed to the starter repository; document where they  
come from and keep paths configurable.
