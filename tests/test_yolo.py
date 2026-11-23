"""Tests for Set 9: COCO -> YOLO conversion, box matching, YOLO detection and training.

The dataset and matching tests need neither PyTorch nor Ultralytics. Model
tests need Ultralytics (``pip install ultralytics``) and are skipped without
it; the pretrained tests also need ``models/yolo11n.pt`` (downloaded by the
YOLO object detection tool on first use). The training test trains a tiny
model from scratch on a synthetic dataset, on the CPU, in a few seconds.
"""

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image, ImageDraw
from skimage import data

from forensics_app.core import ImageDocument
from forensics_app.tools import yolo_detection as yd
from forensics_app.tools import yolo_training as yt
from forensics_app.tools.yolo_data import (
    COCO_FILE,
    Box,
    ConversionCancelled,
    class_counts,
    coco_box_to_yolo,
    coco_classes,
    convert_coco_dataset,
    dataset_folder,
    find_coco_splits,
    iou_matrix,
    label_path_for,
    match_boxes,
    read_data_yaml,
    read_yolo_labels,
)

HAVE_ULTRALYTICS = yd.ultralytics_available()
# data.yaml is read with PyYAML, or Ultralytics' reader when PyYAML is absent.
HAVE_YAML = HAVE_ULTRALYTICS or importlib.util.find_spec("yaml") is not None
NEEDS_YAML = unittest.skipUnless(HAVE_YAML, "reading data.yaml needs PyYAML or ultralytics")
HAVE_PRETRAINED = HAVE_ULTRALYTICS and yd.DEFAULT_PRETRAINED.is_file()

# Roboflow's categories: id 0 is a placeholder supercategory that labels nothing.
CATEGORIES = [
    {"id": 0, "name": "cells", "supercategory": "none"},
    {"id": 1, "name": "PC", "supercategory": "cells"},
    {"id": 2, "name": "RBC", "supercategory": "cells"},
    {"id": 3, "name": "WBC", "supercategory": "cells"},
]


def write_coco_split(folder: Path, images: list[tuple[str, int, int, list[tuple[int, list[float]]]]]) -> None:
    """A COCO split folder: images on disk plus ``_annotations.coco.json``."""
    folder.mkdir(parents=True)
    coco = {"images": [], "annotations": [], "categories": CATEGORIES}
    ann_id = 0
    for image_id, (name, width, height, boxes) in enumerate(images):
        Image.new("RGB", (width, height), (200, 120, 130)).save(folder / name)
        coco["images"].append({"id": image_id, "file_name": name, "width": width, "height": height})
        for category, bbox in boxes:
            coco["annotations"].append(
                {"id": ann_id, "image_id": image_id, "category_id": category, "bbox": bbox}
            )
            ann_id += 1
    (folder / COCO_FILE).write_text(json.dumps(coco))


class CocoToYoloTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.base = Path(self._tmp.name)
        self.coco = self.base / "coco"
        write_coco_split(self.coco / "train", [
            ("a.jpg", 200, 100, [(2, [20, 10, 40, 30]), (3, [100, 50, 50, 40])]),
            ("b.jpg", 200, 100, [(1, [0, 0, 10, 10]), (2, [190, 90, 30, 30])]),  # 2nd crosses the edge
            ("empty.jpg", 200, 100, []),
        ])
        write_coco_split(self.coco / "valid", [("c.jpg", 100, 100, [(2, [10, 10, 20, 20])])])
        write_coco_split(self.coco / "test", [("d.jpg", 100, 100, [(3, [5, 5, 0, 10])])])  # zero width

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_placeholder_supercategory_is_dropped(self) -> None:
        self.assertEqual([c["name"] for c in coco_classes(CATEGORIES)], ["PC", "RBC", "WBC"])
        plain = [{"id": 5, "name": "cat"}, {"id": 7, "name": "dog"}]
        self.assertEqual([c["name"] for c in coco_classes(plain)], ["cat", "dog"])

    def test_box_conversion_normalizes_centres_and_clips(self) -> None:
        self.assertEqual(coco_box_to_yolo([20, 10, 40, 30], 200, 100), (0.2, 0.25, 0.2, 0.3))
        clipped = coco_box_to_yolo([190, 90, 30, 30], 200, 100)
        self.assertAlmostEqual(clipped[0], 0.975)
        self.assertAlmostEqual(clipped[2], 0.05)
        self.assertIsNone(coco_box_to_yolo([5, 5, 0, 10], 100, 100))
        self.assertIsNone(coco_box_to_yolo([300, 5, 10, 10], 100, 100))

    def test_converts_every_split_with_shared_class_ids(self) -> None:
        summary = convert_coco_dataset(self.coco, self.base / "yolo")
        self.assertEqual(summary.names, ["PC", "RBC", "WBC"])
        self.assertEqual([s.name for s in summary.splits], ["train", "valid", "test"])
        train = summary.splits[0]
        self.assertEqual(train.images, 3)
        self.assertEqual(dict(train.boxes), {"RBC": 2, "WBC": 1, "PC": 1})
        self.assertEqual(summary.splits[2].skipped_boxes, 1)

        lines = (self.base / "yolo/train/labels/a.txt").read_text().split()
        self.assertEqual(lines[:5], ["1", "0.200000", "0.250000", "0.200000", "0.300000"])
        self.assertEqual(lines[5], "2")  # WBC: COCO id 3 -> YOLO class 2
        self.assertEqual((self.base / "yolo/train/labels/empty.txt").read_text(), "")
        self.assertTrue((self.base / "yolo/valid/images/c.jpg").is_file())

    def test_labels_read_back_to_the_coco_pixel_boxes(self) -> None:
        convert_coco_dataset(self.coco, self.base / "yolo")
        boxes = read_yolo_labels(self.base / "yolo/train/labels/a.txt", 200, 100)
        self.assertEqual([b.class_id for b in boxes], [1, 2])
        np.testing.assert_allclose([boxes[0].x1, boxes[0].y1, boxes[0].width, boxes[0].height],
                                   [20, 10, 40, 30], atol=1e-3)
        np.testing.assert_allclose([boxes[1].x1, boxes[1].y1, boxes[1].x2, boxes[1].y2],
                                   [100, 50, 150, 90], atol=1e-3)

    @NEEDS_YAML
    def test_data_yaml_points_at_the_splits_from_any_folder(self) -> None:
        summary = convert_coco_dataset(self.coco, self.base / "out with spaces")
        data = read_data_yaml(summary.data_yaml)
        self.assertEqual(data["names"], ["PC", "RBC", "WBC"])
        self.assertEqual(data["nc"], 3)
        self.assertEqual(dataset_folder(data, summary.data_yaml, "val"),
                         (self.base / "out with spaces/valid/images").resolve())

    @NEEDS_YAML
    def test_roboflow_relative_paths_resolve_like_ultralytics(self) -> None:
        root = self.base / "rf"
        (root / "train/images").mkdir(parents=True)
        yaml_path = root / "data.yaml"
        yaml_path.write_text("train: ../train/images\nval: ../valid/images\nnames: ['a']\n")
        data = read_data_yaml(yaml_path)
        self.assertEqual(dataset_folder(data, yaml_path, "train"), (root / "train/images").resolve())

    @NEEDS_YAML
    def test_data_yaml_without_names_is_refused(self) -> None:
        bad = self.base / "bad.yaml"
        bad.write_text("train: images\n")
        with self.assertRaises(ValueError):
            read_data_yaml(bad)

    def test_a_folder_without_coco_files_or_an_output_inside_it_is_refused(self) -> None:
        self.assertEqual(len(find_coco_splits(self.coco)), 3)
        (self.base / "nothing").mkdir()
        with self.assertRaises(ValueError) as caught:
            find_coco_splits(self.base / "nothing")
        self.assertIn(COCO_FILE, str(caught.exception))
        with self.assertRaises(ValueError):
            convert_coco_dataset(self.coco, self.coco / "inside")

    @NEEDS_YAML
    def test_a_single_split_folder_becomes_train(self) -> None:
        summary = convert_coco_dataset(self.coco / "valid", self.base / "single")
        self.assertEqual([s.name for s in summary.splits], ["train"])
        data = read_data_yaml(summary.data_yaml)
        self.assertEqual(data["val"], "train/images")  # Ultralytics needs some validation split

    def test_progress_reports_every_image_and_can_cancel(self) -> None:
        seen = []
        convert_coco_dataset(self.coco, self.base / "yolo", progress=lambda d, t, s: seen.append((d, t)) or True)
        self.assertEqual(seen[0], (0, 5))
        self.assertEqual(len(seen), 5)
        with self.assertRaises(ConversionCancelled):
            convert_coco_dataset(self.coco, self.base / "yolo2", progress=lambda d, t, s: d < 2)

    def test_label_path_swaps_the_images_folder(self) -> None:
        self.assertEqual(label_path_for(Path("/d/test/images/x.jpg")), Path("/d/test/labels/x.txt"))
        self.assertEqual(label_path_for(Path("/d/photo.png")), Path("/d/photo.txt"))

    def test_converter_tool_draws_the_dataset_and_remembers_the_yaml(self) -> None:
        tool = yt.CocoToYoloTool()
        tool._ask_folder = lambda _parent: self.coco
        result = tool.run(None, ImageDocument())
        self.assertEqual(result.image.mode, "RGB")
        self.assertIn("data.yaml", result.message)
        self.assertEqual(yt._last_data_yaml, yt.default_output(self.coco) / "data.yaml")
        tool._ask_folder = lambda _parent: None
        self.assertIsNone(tool.run(None, ImageDocument()))


class MatchingTests(unittest.TestCase):
    def test_iou_of_known_boxes(self) -> None:
        a = [Box(0, 0, 0, 10, 10)]
        b = [Box(0, 0, 0, 10, 10), Box(0, 5, 0, 15, 10), Box(0, 20, 20, 30, 30)]
        np.testing.assert_allclose(iou_matrix(a, b), [[1.0, 50 / 150, 0.0]])
        self.assertEqual(iou_matrix([], b).shape, (0, 3))

    def test_duplicates_wrong_classes_and_misses(self) -> None:
        truth = [Box(0, 0, 0, 10, 10), Box(1, 50, 50, 70, 70), Box(1, 100, 100, 110, 110)]
        predictions = [
            Box(0, 0, 0, 10, 11, score=0.9),  # good
            Box(0, 1, 0, 10, 10, score=0.8),  # duplicate of the same object: false
            Box(0, 50, 50, 70, 70, score=0.7),  # right place, wrong class: false
            Box(1, 52, 50, 70, 70, score=0.6),  # good
        ]
        match = match_boxes(predictions, truth, 0.5)
        self.assertEqual({(p, g) for p, g, _ in match.matched}, {(0, 0), (3, 1)})
        self.assertEqual(match.false_positives, (1, 2))
        self.assertEqual(match.missed, (2,))
        self.assertAlmostEqual(match.precision, 0.5)
        self.assertAlmostEqual(match.recall, 2 / 3)

    def test_the_more_confident_prediction_takes_the_object(self) -> None:
        truth = [Box(0, 0, 0, 10, 10)]
        match = match_boxes([Box(0, 0, 0, 10, 12, 0.3), Box(0, 0, 0, 10, 10, 0.9)], truth)
        self.assertEqual(match.matched[0][0], 1)

    def test_a_stricter_threshold_never_matches_more(self) -> None:
        truth = [Box(0, 0, 0, 10, 10)]
        loose = match_boxes([Box(0, 3, 0, 13, 10)], truth, 0.5)
        strict = match_boxes([Box(0, 3, 0, 13, 10)], truth, 0.75)
        self.assertEqual((loose.tp, strict.tp), (1, 0))

    def test_empty_inputs(self) -> None:
        match = match_boxes([], [])
        self.assertIsNone(match.precision)
        self.assertIsNone(match.recall)
        self.assertEqual(class_counts([], ["a"]), "none")
        self.assertEqual(class_counts([Box(1, 0, 0, 1, 1)] * 2 + [Box(0, 0, 0, 1, 1)], ["PC", "RBC"]), "1 PC, 2 RBC")


class DrawingTests(unittest.TestCase):
    def test_boxes_are_drawn_in_rgb_not_bgr(self) -> None:
        # tab10 colour 3 is red: a swapped image would show it blue.
        out = yd.draw_detections(Image.new("RGB", (200, 200), (255, 255, 255)),
                                 [Box(3, 50, 60, 150, 160, 0.9)], ["a", "b", "c", "d"])
        self.assertEqual(out.size, (200, 200))
        r, g, b = out.getpixel((100, 160))  # the bottom edge
        self.assertGreater(r, 150)
        self.assertLess(b, 100)

    def test_comparison_sheet_with_and_without_ground_truth(self) -> None:
        image = Image.new("RGB", (120, 80), (240, 200, 200))
        truth = [Box(0, 10, 10, 40, 40)]
        predictions = [Box(0, 11, 10, 40, 41, 0.8), Box(1, 60, 20, 90, 60, 0.4)]
        match = match_boxes(predictions, truth)
        both = yd.render_comparison(image, predictions, truth, ["PC", "RBC"], match, title="t")
        alone = yd.render_comparison(image, predictions, None, ["PC", "RBC"], None)
        self.assertEqual(both.mode, "RGB")
        self.assertGreater(both.width, alone.width * 1.5)  # two panels against one

    def test_training_curves_from_results_csv(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.csv"
            path.write_text(
                "epoch,train/box_loss,train/cls_loss,metrics/mAP50(B),metrics/mAP50-95(B),val/box_loss\n"
                "1,1.5,2.0,0.4,0.2,1.6\n2,1.2,1.4,0.7,0.4,1.3\n"
            )
            columns = yt.read_results_csv(path)
        self.assertEqual(columns["metrics/mAP50(B)"], [0.4, 0.7])
        self.assertEqual(yt.render_training_curves(columns, "run").mode, "RGB")


class ParamTests(unittest.TestCase):
    def test_detect_params_are_fractions_and_the_size_a_multiple_of_32(self) -> None:
        params = yd.parse_detect_params(Path("w.pt"), "25", "70", "640")
        self.assertEqual((params.conf, params.iou, params.imgsz), (0.25, 0.7, 640))
        with self.assertRaises(ValueError) as caught:
            yd.parse_detect_params(Path("w.pt"), "25", "70", "650")
        self.assertIn("multiple of 32", str(caught.exception))
        with self.assertRaises(ValueError):
            yd.parse_detect_params(Path("w.pt"), "0", "70", "640")

    def test_missing_custom_weights_are_refused_but_official_ones_download(self) -> None:
        if not HAVE_ULTRALYTICS:
            with self.assertRaises(ValueError) as caught:
                yd.check_weights(Path("best.pt"))
            self.assertIn("pip install ultralytics", str(caught.exception))
            return
        with self.assertRaises(ValueError):
            yd.check_weights(Path("/nowhere/best.pt"))
        yd.check_weights(Path("/nowhere/yolo11n.pt"))  # Ultralytics can fetch it by name

    def test_time_estimate_grows_with_epochs_and_data(self) -> None:
        def request(epochs: int, fraction: float) -> yt.TrainRequest:
            return yt.TrainRequest(Path("d.yaml"), True, epochs, 16, 640, fraction, "mps")
        self.assertAlmostEqual(yt.estimate_minutes(1000, 0, request(2, 1.0)),
                               2 * yt.estimate_minutes(1000, 0, request(1, 1.0)))
        self.assertLess(yt.estimate_minutes(1000, 100, request(1, 0.3)), yt.estimate_minutes(1000, 100, request(1, 1.0)))


@unittest.skipUnless(HAVE_PRETRAINED, f"needs ultralytics and {yd.DEFAULT_PRETRAINED}")
class PretrainedDetectionTests(unittest.TestCase):
    PARAMS = yd.DetectParams(yd.DEFAULT_PRETRAINED, 0.25, 0.7, 640)

    def setUp(self) -> None:
        self.model = yd.load_yolo(yd.DEFAULT_PRETRAINED)
        self.names = yd.class_names(self.model)

    def test_finds_the_astronaut_as_a_person(self) -> None:
        boxes = yd.detect(self.model, Image.fromarray(data.astronaut()), self.PARAMS)
        top = boxes[0]
        self.assertEqual(self.names[top.class_id], "person")
        self.assertGreater(top.score, 0.6)
        self.assertTrue(all(boxes[i].score >= boxes[i + 1].score for i in range(len(boxes) - 1)))

    def test_finds_the_cat(self) -> None:
        boxes = yd.detect(self.model, Image.fromarray(data.chelsea()), self.PARAMS)
        self.assertIn("cat", {self.names[b.class_id] for b in boxes})

    def test_the_model_is_loaded_once(self) -> None:
        self.assertIs(yd.load_yolo(yd.DEFAULT_PRETRAINED), self.model)

    def test_detection_tool_returns_an_annotated_image(self) -> None:
        tool = yd.YoloDetectionTool()
        tool._ask_params = lambda _parent: self.PARAMS
        document = ImageDocument()
        document.current = Image.fromarray(data.astronaut())
        self.assertTrue(tool.is_available(document))
        result = tool.run(None, document)
        self.assertEqual(result.image.size, document.current.size)
        self.assertIn("person", result.message)
        self.assertIn("1. person", result.details)
        tool._ask_params = lambda _parent: None
        self.assertIsNone(tool.run(None, document))

    def test_ground_truth_tool_matches_a_labelled_person(self) -> None:
        with TemporaryDirectory() as tmp:
            images, labels = Path(tmp) / "test/images", Path(tmp) / "test/labels"
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            photo = images / "astronaut.png"
            Image.fromarray(data.astronaut()).save(photo)
            person = yd.detect(self.model, Image.open(photo), self.PARAMS)[0]
            w, h = 512, 512
            (labels / "astronaut.txt").write_text(
                f"0 {(person.x1 + person.x2) / 2 / w} {(person.y1 + person.y2) / 2 / h} "
                f"{person.width / w} {person.height / h}\n"
                "0 0.05 0.05 0.04 0.04\n"  # a "person" nobody can find: missed
            )
            document = ImageDocument()
            document.load(photo)
            tool = yd.YoloGroundTruthTool()
            seen = {}

            def ask(_parent, found_labels):
                seen["labels"] = found_labels
                return yd.TestRequest(self.PARAMS, found_labels, 0.5)

            tool._ask_params = ask
            result = tool.run(None, document)
        self.assertEqual(seen["labels"], labels / "astronaut.txt")  # found beside images/
        self.assertEqual(result.details["Missed (FN)"], 1)
        self.assertGreaterEqual(result.details["Correct (TP)"], 1)
        self.assertEqual(result.image.mode, "RGB")


def make_toy_dataset(root: Path, count: int = 8) -> Path:
    """Red squares (class 0) and blue circles (class 1) on noise, in YOLO format."""
    rng = np.random.default_rng(0)
    for split in ("train", "valid"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        for index in range(count):
            image = Image.fromarray(rng.integers(0, 80, (64, 64, 3), dtype=np.uint8))
            draw = ImageDraw.Draw(image)
            x, y = int(rng.integers(4, 30)), int(rng.integers(4, 30))
            if index % 2:
                draw.ellipse((x, y, x + 24, y + 24), fill=(40, 60, 230))
            else:
                draw.rectangle((x, y, x + 24, y + 24), fill=(230, 40, 40))
            image.save(root / split / "images" / f"{index}.png")
            (root / split / "labels" / f"{index}.txt").write_text(
                f"{index % 2} {(x + 12) / 64} {(y + 12) / 64} {24 / 64} {24 / 64}\n"
            )
    yaml_path = root / "data.yaml"
    yaml_path.write_text(
        f'path: "{root}"\ntrain: "train/images"\nval: "valid/images"\nnc: 2\nnames: ["square", "circle"]\n'
    )
    return yaml_path


@unittest.skipUnless(HAVE_ULTRALYTICS, "needs ultralytics")
class TrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.yaml = make_toy_dataset(Path(self._tmp.name))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _request(self, epochs: int) -> yt.TrainRequest:
        return yt.TrainRequest(self.yaml, False, epochs, 4, 64, 1.0, "cpu")

    def test_training_saves_weights_and_reports_progress(self) -> None:
        job = yt.TrainingJob(self._request(2))
        job.start()
        job.join(300)
        self.assertIsNone(job.error, repr(job.error))
        outcome = job.outcome
        self.assertTrue(outcome.best.is_file())
        self.assertEqual(outcome.save_dir.parent.resolve(), (self.yaml.parent / "runs").resolve())
        self.assertEqual(outcome.epochs_done, 2)
        self.assertFalse(outcome.stopped_early)
        self.assertTrue(0.0 <= outcome.map50 <= 1.0)
        done, total, text = job.status()
        self.assertEqual((done, total), (total, 2 * job.batches))
        self.assertIn("Epoch 2/2", text)
        # The saved weights work with the detection side.
        model = yd.load_yolo(outcome.best)
        self.assertEqual(yd.class_names(model), ["square", "circle"])

    def test_stop_ends_after_the_first_batch_and_keeps_weights(self) -> None:
        job = yt.TrainingJob(self._request(5))
        job.request_stop()  # before the first batch: stops after it
        job.start()
        job.join(300)
        self.assertIsNone(job.error, repr(job.error))
        self.assertTrue(job.outcome.stopped_early)
        self.assertEqual(job.outcome.epochs_done, 1)
        self.assertTrue(job.outcome.best.is_file())


if __name__ == "__main__":
    unittest.main()
