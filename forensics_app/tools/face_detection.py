"""Face detection with YuNet, on the working image or a webcam stream (Set 6.1).

YuNet is a small CNN face detector shipped with OpenCV as
``cv2.FaceDetectorYN``; the slide asks for it instead of Haar cascades. Each
face comes with a box, a confidence score and five landmarks (eyes, nose tip,
mouth corners), drawn as on the slide: a green box labelled "Face: 0.91" and
blue dots.

The model file is not in the repository (see README): by default it is read
from ``models/face_detection_yunet_2023mar.onnx`` and the dialog can point
anywhere else.
"""

from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path
import time
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .param_dialog import FileChooser, IntField, ParamDialog, parse_int_fields


ACCEPTED_MODES = ("L", "RGB", "RGBA", "P")

MODEL_NAME = "face_detection_yunet_2023mar.onnx"
DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "models" / MODEL_NAME
MODEL_URL = (
    "https://huggingface.co/opencv/opencv_zoo/resolve/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
MODEL_TYPES = [("ONNX model", "*.onnx"), ("All files", "*.*")]

# The tutorial notebook's settings, entered as percentages.
DETECTION_FIELDS = (
    IntField("Min. score", 60, 99, "% confidence to keep a face"),
    IntField("NMS overlap", 30, 99, "% overlap at which boxes merge"),
)
TOP_K = 5000  # candidates kept before non-maximum suppression, as in the notebook

# Photos are shrunk to this longest side for detection (boxes are scaled
# back): YuNet takes ~4 ms on 512 px, and a 12 MP photo would take seconds.
MAX_DETECT_SIDE = 1280

LANDMARKS = ("right eye", "left eye", "nose tip", "right mouth corner", "left mouth corner")
BOX_COLOUR = (0, 255, 0)  # RGB, as on the slide
LANDMARK_COLOUR = (0, 0, 255)


@dataclass(frozen=True)
class Face:
    box: tuple[int, int, int, int]  # x, y, width, height in image pixels
    score: float
    landmarks: tuple[tuple[int, int], ...]  # five (x, y) points, in LANDMARKS order


@dataclass(frozen=True)
class DetectionParams:
    model: Path
    min_score: float  # 0..1
    nms: float  # 0..1


def parse_detection_params(model: Path, score_text: str, nms_text: str) -> DetectionParams:
    score, nms = parse_int_fields(DETECTION_FIELDS, (score_text, nms_text))
    return DetectionParams(model, score / 100, nms / 100)


def check_model(path: Path) -> None:
    """Raise ``ValueError`` saying where to get the model when ``path`` is missing."""
    if not path.is_file():
        raise ValueError(
            f"The YuNet model was not found at {path}.\n\n"
            f"Download {MODEL_NAME} from {MODEL_URL} and put it in the models folder, "
            "or choose the file in the dialog."
        )


class YuNetDetector:
    """``cv2.FaceDetectorYN``, created once and reused for every image or frame."""

    def __init__(self, params: DetectionParams) -> None:
        check_model(params.model)
        try:
            self._detector = cv2.FaceDetectorYN.create(
                str(params.model), "", (320, 320), params.min_score, params.nms, TOP_K
            )
        except cv2.error as error:
            raise ValueError(f"OpenCV could not load {params.model.name} as a YuNet model.") from error
        self._size: tuple[int, int] | None = None

    def detect_bgr(self, frame: np.ndarray) -> list[Face]:
        """Faces in a BGR uint8 frame, at that frame's own pixel size."""
        height, width = frame.shape[:2]
        if self._size != (width, height):
            self._detector.setInputSize((width, height))
            self._size = (width, height)
        _, found = self._detector.detect(frame)
        if found is None:
            return []
        faces = []
        for row in found:
            x, y, w, h = (int(round(v)) for v in row[:4])
            points = tuple((int(round(px)), int(round(py))) for px, py in row[4:14].reshape(5, 2))
            faces.append(Face((x, y, w, h), float(row[14]), points))
        return sorted(faces, key=lambda face: -face.score)

    def detect(self, image: Image.Image) -> list[Face]:
        """Faces in a PIL image, shrunk to MAX_DETECT_SIDE for speed, in its own pixels."""
        rgb = image.convert("RGB")
        scale = min(1.0, MAX_DETECT_SIDE / max(rgb.size))
        if scale < 1.0:
            rgb = rgb.resize(
                (max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale))),
                Image.Resampling.LANCZOS,
            )
        frame = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
        faces = self.detect_bgr(frame)
        if scale == 1.0:
            return faces
        back = 1.0 / scale
        return [
            Face(
                tuple(int(round(v * back)) for v in face.box),
                face.score,
                tuple((int(round(px * back)), int(round(py * back))) for px, py in face.landmarks),
            )
            for face in faces
        ]


def draw_faces(rgb: np.ndarray, faces: list[Face]) -> np.ndarray:
    """A copy of an RGB uint8 array with each face's box, score and landmarks drawn."""
    out = np.ascontiguousarray(rgb).copy()
    # Line widths follow the image size, so the drawing reads the same at any size.
    thick = max(2, round(max(out.shape[:2]) / 300))
    for face in faces:
        x, y, w, h = face.box
        cv2.rectangle(out, (x, y), (x + w, y + h), BOX_COLOUR, thick)
        cv2.putText(
            out,
            f"Face: {face.score:.2f}",
            (x, max(12 * thick, y - 4 * thick)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3 * thick,
            BOX_COLOUR,
            thick,
            cv2.LINE_AA,
        )
        for point in face.landmarks:
            cv2.circle(out, point, 2 * thick, LANDMARK_COLOUR, -1, cv2.LINE_AA)
    return out


def render_sheet(image: Image.Image, faces: list[Face], name: str) -> Image.Image:
    """The original and the annotated image side by side, as on the slide."""
    rgb = np.asarray(image.convert("RGB"))
    annotated = draw_faces(rgb, faces)
    height, width = rgb.shape[:2]
    panel_w = 6.0
    figure = Figure(
        figsize=(2 * panel_w, panel_w * height / width + 0.6),
        dpi=max(100, min(200, width / (panel_w * 0.9))),
        layout="constrained",
    )
    for axis, picture, title in zip(
        figure.subplots(1, 2),
        (rgb, annotated),
        (f"Original {name}", f"YuNet Face Detection (OpenCV {cv2.__version__.split('.')[0]})"),
    ):
        axis.imshow(picture)
        axis.set_title(title)
        axis.axis("off")
    FigureCanvasAgg(figure)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0.1)
    plt.close(figure)
    buffer.seek(0)
    with Image.open(buffer) as rendered:
        return rendered.convert("RGB")


def face_details(faces: list[Face], params: DetectionParams) -> dict[str, object]:
    details: dict[str, object] = {
        "Model": f"YuNet ({params.model.name})",
        "OpenCV": cv2.__version__,
        "Min. score": f"{params.min_score:.2f}",
        "NMS overlap": f"{params.nms:.2f}",
        "Faces found": len(faces),
    }
    for number, face in enumerate(faces, start=1):
        x, y, w, h = face.box
        details[f"Face {number}"] = f"score {face.score:.2f} · x={x}, y={y}, {w}×{h}"
    return details


def detection_sections(
    dialog: ParamDialog, initial_model: Path
) -> tuple[FileChooser, list[tk.StringVar]]:
    """The model file and the two thresholds, shared by the photo and webcam tools."""
    chooser = FileChooser(
        dialog,
        dialog.section("Model"),
        prompt="Choose the YuNet model file (.onnx).",
        filetypes=MODEL_TYPES,
        describe=lambda path: f"{path.stat().st_size / 1024:.0f} KB" if path.is_file() else "",
        initial=initial_model,
    )
    variables = dialog.int_fields(dialog.section("Detection"), DETECTION_FIELDS)
    return chooser, variables


def read_detection(chooser: FileChooser, variables: list[tk.StringVar]) -> tuple[DetectionParams, str]:
    model = chooser.require()
    params = parse_detection_params(model, *(v.get() for v in variables))
    return params, f"score ≥ {params.min_score:.2f} · NMS {params.nms:.2f}"


class FaceDetectionTool(ForensicsTool):
    tool_id = "face_detection"
    produces_figure = True
    title = "Face detection (YuNet)"
    category = "Faces"
    description = "Find faces with OpenCV's YuNet: boxes, scores and five landmarks (Set 6.1)."

    ACCEPTED_MODES = ACCEPTED_MODES

    def __init__(self) -> None:
        self._model = DEFAULT_MODEL  # the dialog remembers a model chosen elsewhere

    def is_available(self, document: ImageDocument) -> bool:
        if document.current is None:
            return True
        if document.current.mode not in self.ACCEPTED_MODES:
            return False
        return document.last_tool_id != self.tool_id

    def unavailable_message(self, document: ImageDocument) -> str:
        current = document.current
        if current is not None and current.mode not in self.ACCEPTED_MODES:
            return (
                f"This image mode is not supported for face detection (mode {current.mode!r}).\n\n"
                "Open an L or RGB image."
            )
        return "The detected faces are already shown.\n\nUndo or reset the image to detect again."

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        source = document.current
        assert source is not None

        params = self._ask_params(parent)
        if params is None:
            return None
        try:
            started = time.perf_counter()
            faces = YuNetDetector(params).detect(source)
            elapsed = time.perf_counter() - started
        except ValueError as error:
            messagebox.showinfo(self.title, str(error), parent=parent)
            return None

        name = document.path.stem if document.path is not None else "image"
        details: dict[str, object] = {"Operation": self.title, **face_details(faces, params)}
        details["Time"] = f"{elapsed * 1000:.0f} ms"
        count = len(faces)
        return ToolResult(
            image=render_sheet(source, faces, name),
            message=f"Found {count} face{'' if count == 1 else 's'}."
            if count
            else "No faces found. Lower the minimum score to see weaker detections.",
            details=details,
        )

    def _ask_params(self, parent: tk.Misc | None) -> DetectionParams | None:
        dialog: ParamDialog[DetectionParams] = ParamDialog(
            parent,
            title=self.title,
            intro="Detect faces with YuNet, a small neural network in OpenCV, "
            "and mark each with its box, score and five landmarks.",
            action="Detect",
        )
        chooser, variables = detection_sections(dialog, self._model)
        params = dialog.show(lambda: read_detection(chooser, variables))
        if params is not None:
            self._model = params.model
        return params


# --------------------------------------------------------------------------
# Webcam
# --------------------------------------------------------------------------

CAMERA_FIELDS = (
    IntField("Camera", 0, 9, "0 built-in; 1+ others, e.g. an iPhone", minimum=0),
)
# An iPhone (Continuity Camera) or USB camera can take seconds to send its
# first picture, so the window waits this long before giving up; once frames
# flow, a gap this long means the camera has stopped.
STARTUP_SECONDS = 10.0
GAP_SECONDS = 2.0
IPHONE_HINT = (
    "On a Mac, camera 0 is the built-in camera and 1 is usually an iPhone "
    "(Continuity Camera): keep it nearby, signed in to the same Apple ID, with "
    "Wi-Fi and Bluetooth on."
)
PREVIEW_MAX_W = 960
FRAME_DELAY_MS = 15


@dataclass(frozen=True)
class WebcamParams:
    detection: DetectionParams
    camera: int


@dataclass(frozen=True)
class Capture:
    frame: Image.Image  # annotated RGB frame
    faces: list[Face]
    fps: float


class LiveDetectionWindow:
    """Shows camera frames with detections until Capture or Close.

    ``read_frame`` returns the next BGR frame or None when the camera stops
    delivering; keeping it a plain function lets tests feed still images.
    """

    def __init__(
        self, parent: tk.Misc, detector: YuNetDetector, read_frame, title: str, camera: int = 0
    ) -> None:
        self._detector = detector
        self._read = read_frame
        self._camera = camera
        self._latest: tuple[np.ndarray, list[Face]] | None = None
        self._times: list[float] = []
        self._opened = time.perf_counter()
        self._last_frame: float | None = None  # when the last frame arrived
        self.result: Capture | None = None
        self.error: str | None = None

        self.top = tk.Toplevel(parent)
        self.top.title(title)
        self.top.transient(parent)
        frame = ttk.Frame(self.top, padding=12)
        frame.pack(fill="both", expand=True)
        self._view = ttk.Label(frame, text=f"Waiting for camera {camera} to start…", anchor="center")
        self._view.pack(fill="both", expand=True)
        bar = ttk.Frame(frame)
        bar.pack(fill="x", pady=(10, 0))
        self._status = ttk.Label(bar, text="", style="Hint.TLabel")
        self._status.pack(side="left")
        ttk.Button(bar, text="Close", command=self.close).pack(side="right")
        ttk.Button(bar, text="Capture", style="Primary.TButton", command=self.capture).pack(
            side="right", padx=(0, 8)
        )
        self.top.bind("<space>", lambda _event: self.capture())
        self.top.bind("<Return>", lambda _event: self.capture())
        self.top.bind("<Escape>", lambda _event: self.close())
        self.top.protocol("WM_DELETE_WINDOW", self.close)
        self._photo: ImageTk.PhotoImage | None = None
        self._job = self.top.after(1, self._tick)

    def run(self) -> Capture | None:
        self.top.grab_set()
        self.top.wait_window()
        return self.result

    def _tick(self) -> None:
        frame = self._read()
        now = time.perf_counter()
        if frame is None:
            if self._last_frame is None and now - self._opened > STARTUP_SECONDS:
                self.error = (
                    f"Camera {self._camera} opened but sent no picture within "
                    f"{STARTUP_SECONDS:.0f} s.\n\n{IPHONE_HINT}"
                )
                self.close()
                return
            if self._last_frame is not None and now - self._last_frame > GAP_SECONDS:
                self.error = f"Camera {self._camera} stopped sending pictures."
                self.close()
                return
            if self._last_frame is None:
                self._status.configure(
                    text=f"Waiting for camera {self._camera}… {now - self._opened:.0f} s"
                )
        else:
            self._last_frame = now
            faces = self._detector.detect_bgr(frame)
            rgb = draw_faces(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), faces)
            self._latest = (rgb, faces)
            self._times = [t for t in self._times if now - t < 1.0] + [now]
            shown = Image.fromarray(rgb)
            if shown.width > PREVIEW_MAX_W:
                shown = shown.resize(
                    (PREVIEW_MAX_W, round(shown.height * PREVIEW_MAX_W / shown.width)),
                    Image.Resampling.BILINEAR,
                )
            self._photo = ImageTk.PhotoImage(shown, master=self.top)
            self._view.configure(image=self._photo, text="")
            count = len(faces)
            self._status.configure(
                text=f"{count} face{'' if count == 1 else 's'} · {len(self._times)} fps · "
                "Space captures, Esc closes"
            )
        self._job = self.top.after(FRAME_DELAY_MS, self._tick)

    def capture(self) -> None:
        if self._latest is None:
            return
        rgb, faces = self._latest
        self.result = Capture(Image.fromarray(rgb), faces, float(len(self._times)))
        self.close()

    def close(self) -> None:
        if self._job is not None:
            self.top.after_cancel(self._job)
            self._job = None
        if self.top.winfo_exists():
            self.top.grab_release()
            self.top.destroy()


class WebcamFaceDetectionTool(ForensicsTool):
    tool_id = "face_detection_webcam"
    produces_figure = True
    title = "Face detection (webcam)"
    category = "Faces"
    description = "Run YuNet live on a webcam stream; capture a frame as the image (Set 6.1)."
    requires_image = False

    def __init__(self) -> None:
        self._model = DEFAULT_MODEL

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        params = self._ask_params(parent)
        if params is None:
            return None
        try:
            detector = YuNetDetector(params.detection)
        except ValueError as error:
            messagebox.showinfo(self.title, str(error), parent=parent)
            return None

        camera = cv2.VideoCapture(params.camera)
        if not camera.isOpened():
            camera.release()
            messagebox.showinfo(
                self.title,
                f"Camera {params.camera} could not be opened.\n\n"
                f"{IPHONE_HINT}\n\n"
                "If camera 0 fails too, allow camera access for the app you started "
                "ForensicsApp from (Terminal, iTerm, VS Code…) in System Settings → "
                "Privacy & Security → Camera, then restart it. Also check that no "
                "other app is using the camera.",
                parent=parent,
            )
            return None

        def read_frame() -> np.ndarray | None:
            ok, frame = camera.read()
            return frame if ok else None

        try:
            window = LiveDetectionWindow(parent, detector, read_frame, self.title, params.camera)
            captured = window.run()
        finally:
            camera.release()
        if window.error:
            messagebox.showinfo(self.title, window.error, parent=parent)
        if captured is None:
            return None

        details: dict[str, object] = {
            "Operation": self.title,
            "Camera": params.camera,
            "Frame": f"{captured.frame.width}×{captured.frame.height}",
            "Live rate": f"{captured.fps:.0f} fps",
            **face_details(captured.faces, params.detection),
        }
        count = len(captured.faces)
        return ToolResult(
            image=captured.frame,
            message=f"Captured a webcam frame with {count} face{'' if count == 1 else 's'}.",
            details=details,
        )

    def _ask_params(self, parent: tk.Misc | None) -> WebcamParams | None:
        dialog: ParamDialog[WebcamParams] = ParamDialog(
            parent,
            title=self.title,
            intro="Detect faces live in the camera image. Capture keeps the "
            "current frame, with its detections, as the working image.",
            action="Start camera",
        )
        chooser, variables = detection_sections(dialog, self._model)
        (camera_var,) = dialog.int_fields(dialog.section("Camera"), CAMERA_FIELDS)

        def validate() -> tuple[WebcamParams, str]:
            detection, readout = read_detection(chooser, variables)
            (camera,) = parse_int_fields(CAMERA_FIELDS, (camera_var.get(),))
            return WebcamParams(detection, camera), f"camera {camera} · {readout}"

        params = dialog.show(validate)
        if params is not None:
            self._model = params.detection.model
        return params
