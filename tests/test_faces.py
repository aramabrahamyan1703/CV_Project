"""Tests for Set 6: YuNet face detection, eigenfaces and recognition.

Detection tests need the YuNet model file in models/ (see README) and are
skipped without it. Eigenface tests build a small synthetic face set, so they
do not need the LFW download.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import numpy as np
from PIL import Image
from skimage import data

from forensics_app.core import ImageDocument
from forensics_app.tools import face_detection as fd
from forensics_app.tools import face_recognition as fr
from forensics_app.tools.eigenfaces import (
    build_model,
    distances,
    load_faces,
    load_model,
    nearest,
    person_name,
    project,
    same_faces,
    save_model,
    to_face_vector,
)

HAVE_MODEL = fd.DEFAULT_MODEL.is_file()


def _params(score: float = 0.6) -> fd.DetectionParams:
    return fd.DetectionParams(fd.DEFAULT_MODEL, score, 0.3)


@unittest.skipUnless(HAVE_MODEL, f"YuNet model not found at {fd.DEFAULT_MODEL}")
class FaceDetectionTests(unittest.TestCase):
    def test_finds_the_astronauts_face_with_landmarks(self) -> None:
        (face,) = fd.YuNetDetector(_params()).detect(Image.fromarray(data.astronaut()))
        x, y, w, h = face.box
        self.assertGreater(face.score, 0.9)
        self.assertTrue(150 < x < 210 and 40 < y < 90 and 70 < w < 110, face.box)
        self.assertEqual(len(face.landmarks), 5)
        for px, py in face.landmarks:  # every landmark inside the box
            self.assertTrue(x <= px <= x + w and y <= py <= y + h)

    def test_finds_no_face_in_a_photo_without_one(self) -> None:
        self.assertEqual(fd.YuNetDetector(_params()).detect(Image.fromarray(data.coffee())), [])

    def test_large_images_are_detected_small_and_reported_full_size(self) -> None:
        detector = fd.YuNetDetector(_params())
        astronaut = Image.fromarray(data.astronaut())
        (small,) = detector.detect(astronaut)
        (large,) = detector.detect(astronaut.resize((2048, 2048)))  # shrunk to 1280 inside
        for a, b in zip(small.box, large.box):
            self.assertAlmostEqual(b / 4, a, delta=6)

    def test_drawing_keeps_the_image_and_marks_the_face(self) -> None:
        rgb = data.astronaut()
        faces = fd.YuNetDetector(_params()).detect(Image.fromarray(rgb))
        drawn = fd.draw_faces(rgb, faces)
        self.assertEqual(drawn.shape, rgb.shape)
        green = (drawn[..., 0] == 0) & (drawn[..., 1] == 255) & (drawn[..., 2] == 0)
        self.assertGreater(int(green.sum()), 100)
        self.assertFalse(np.array_equal(drawn, rgb))
        self.assertTrue(np.array_equal(rgb, data.astronaut()))  # input untouched

    def test_tool_returns_the_side_by_side_sheet(self) -> None:
        document = ImageDocument()
        document.current = Image.fromarray(data.astronaut())
        tool = fd.FaceDetectionTool()
        tool._ask_params = lambda _parent: None
        self.assertIsNone(tool.run(None, document))
        tool._ask_params = lambda _parent: _params()
        result = tool.run(None, document)
        self.assertEqual(result.details["Faces found"], 1)
        self.assertGreater(result.image.width, 1.5 * result.image.height)
        self.assertEqual(result.message, "Found 1 face.")
        document.apply(result.image, tool.tool_id)
        self.assertFalse(tool.is_available(document))


class FaceDetectionSetupTests(unittest.TestCase):
    def test_a_missing_model_says_where_to_get_it(self) -> None:
        with self.assertRaises(ValueError) as caught:
            fd.YuNetDetector(fd.DetectionParams(Path("/nowhere/yunet.onnx"), 0.6, 0.3))
        self.assertIn(fd.MODEL_URL, str(caught.exception))

    def test_thresholds_are_entered_as_percentages(self) -> None:
        params = fd.parse_detection_params(Path("m.onnx"), "75", "30")
        self.assertEqual((params.min_score, params.nms), (0.75, 0.3))
        with self.assertRaises(ValueError):
            fd.parse_detection_params(Path("m.onnx"), "0", "30")

    @unittest.skipUnless(HAVE_MODEL, "YuNet model not found")
    def test_webcam_that_cannot_open_explains_camera_access(self) -> None:
        class ClosedCamera:
            def __init__(self, _index):
                pass

            def isOpened(self):
                return False

            def release(self):
                pass

        tool = fd.WebcamFaceDetectionTool()
        tool._ask_params = lambda _parent: fd.WebcamParams(_params(), 0)
        with mock.patch.object(fd.cv2, "VideoCapture", ClosedCamera), mock.patch.object(
            fd, "messagebox"
        ) as boxes:
            self.assertIsNone(tool.run(None, ImageDocument()))
        self.assertIn("Privacy & Security", boxes.showinfo.call_args.args[1])


def _write_faces(folder: Path, per_person: int, seed: int) -> None:
    """Three 'people', each a fixed 12×10 pattern plus a little noise per photo."""
    rng = np.random.default_rng(seed)
    patterns = np.random.default_rng(0).integers(40, 215, (3, 12, 10))
    for person, pattern in zip(("Ada_Lovelace", "Alan_Turing", "Grace_Hopper"), patterns):
        (folder / person).mkdir(parents=True)
        for number in range(per_person):
            noisy = np.clip(pattern + rng.normal(0, 6, pattern.shape), 0, 255).astype(np.uint8)
            Image.fromarray(noisy, mode="L").save(folder / person / f"{number:02d}.png")


class FaceFolderCase(unittest.TestCase):
    """A temporary train/test pair of synthetic face folders."""

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.train = self.root / "train"
        self.test = self.root / "test"
        _write_faces(self.train, 6, seed=1)
        _write_faces(self.test, 2, seed=2)
        (self.train / ".DS_Store").write_text("hidden")
        self.faces = load_faces(self.train)

    def tearDown(self) -> None:
        self._tmp.cleanup()


class EigenfaceTests(FaceFolderCase):
    def test_faces_are_named_after_their_folder(self) -> None:
        self.assertEqual(len(self.faces.names), 18)
        self.assertEqual(self.faces.shape, (12, 10))
        self.assertEqual(sorted(set(self.faces.names)), ["Ada Lovelace", "Alan Turing", "Grace Hopper"])
        self.assertEqual(person_name(self.train / "solo.png", self.train), "solo")

    def test_odd_sized_images_are_resized_to_the_common_size(self) -> None:
        Image.new("L", (20, 24), 128).save(self.train / "Ada_Lovelace" / "big.png")
        self.assertEqual(load_faces(self.train).pixels.shape, (19, 120))

    def test_eigenfaces_are_orthonormal_and_ordered_by_variance(self) -> None:
        model = build_model(self.faces, 10)
        np.testing.assert_allclose(model.eigenfaces @ model.eigenfaces.T, np.eye(10), atol=1e-9)
        self.assertTrue(np.all(np.diff(model.explained) <= 1e-12))
        self.assertLessEqual(model.explained.sum(), 1.0 + 1e-12)
        centred = self.faces.pixels - model.mean
        np.testing.assert_allclose(model.coefficients, centred @ model.eigenfaces.T)

    def test_more_eigenfaces_reconstruct_the_faces_better(self) -> None:
        errors = []
        for k in (1, 3, 8, 17):
            model = build_model(self.faces, k)
            rebuilt = model.mean + model.coefficients @ model.eigenfaces
            errors.append(float(np.abs(rebuilt - self.faces.pixels).mean()))
        self.assertEqual(errors, sorted(errors, reverse=True))
        self.assertLess(errors[-1], 1e-6)  # all 17 non-trivial components: exact

    def test_signs_and_components_match_scikit_learn(self) -> None:
        try:
            from sklearn.decomposition import PCA
        except ImportError:
            self.skipTest("scikit-learn not installed")
        model = build_model(self.faces, 5)
        reference = PCA(n_components=5, svd_solver="full").fit(self.faces.pixels)
        np.testing.assert_allclose(model.eigenfaces, reference.components_, atol=1e-8)

    def test_rejects_impossible_component_counts(self) -> None:
        with self.assertRaises(ValueError):
            build_model(self.faces, 0)
        with self.assertRaises(ValueError):
            build_model(self.faces, 19)

    def test_model_survives_saving_and_loading(self) -> None:
        model = build_model(self.faces, 6, self.train)
        path = self.root / "model.npz"
        save_model(model, path)
        loaded = load_model(path)
        np.testing.assert_allclose(loaded.eigenfaces, model.eigenfaces)
        np.testing.assert_allclose(loaded.coefficients, model.coefficients)
        np.testing.assert_allclose(loaded.spread, model.spread)
        self.assertEqual(loaded.names, model.names)
        self.assertEqual(loaded.files[0], "Ada_Lovelace/00.png")
        self.assertEqual(loaded.shape, (12, 10))

    def test_loading_something_else_is_a_clear_error(self) -> None:
        other = self.root / "other.npz"
        np.savez(other, values=np.zeros(3))
        with self.assertRaises(ValueError) as caught:
            load_model(other)
        self.assertIn("not an eigenface model", str(caught.exception))
        garbage = self.root / "garbage.npz"
        garbage.write_text("not a numpy file")
        with self.assertRaises(ValueError):
            load_model(garbage)

    def test_a_training_face_is_its_own_nearest_neighbour(self) -> None:
        model = build_model(self.faces, 8)
        image = Image.open(self.faces.files[4])
        for whiten in (True, False):
            index, distance = nearest(model, project(model, image), 1, whiten)[0]
            self.assertEqual(index, 4)
            self.assertAlmostEqual(distance, 0.0, places=6)

    def test_whitening_divides_by_each_coefficients_spread(self) -> None:
        model = build_model(self.faces, 4)
        np.testing.assert_allclose(model.spread, model.coefficients.std(axis=0, ddof=1))
        weights = project(model, Image.open(self.faces.files[0]))
        expected = np.linalg.norm(
            (model.coefficients - weights) / model.spread, axis=1
        )
        np.testing.assert_allclose(distances(model, weights, whiten=True), expected, atol=1e-9)


class RecognitionToolTests(FaceFolderCase):
    def _build(self) -> Path:
        tool = fr.EigenfaceBuilderTool()
        tool._ask_params = lambda _parent: (self.train, 8, 6)
        result = tool.run(None, ImageDocument())
        self.assertIn("8 eigenfaces from 18 faces", result.message)
        self.assertEqual(result.details["Eigenfaces (K)"], 8)
        return self.root / "eigenfaces_model.npz"

    def test_builder_saves_the_model_next_to_the_training_folder(self) -> None:
        path = self._build()
        self.assertTrue(path.is_file())
        self.assertEqual(load_model(path).components, 8)

    def test_recognition_names_a_test_face_and_checks_its_folder(self) -> None:
        model = load_model(self._build())
        document = ImageDocument()
        document.load(self.test / "Alan_Turing" / "01.png")
        tool = fr.FaceRecognitionTool()
        tool._ask_params = lambda _parent: (model, 8, True, 3)
        result = tool.run(None, document)
        self.assertEqual(result.details["Predicted"], "Alan Turing")
        self.assertEqual(result.details["True"], "Alan Turing (correct)")
        self.assertTrue(result.details["Closest 1"].startswith("Alan Turing"))
        self.assertEqual(result.message, "Predicted Alan Turing: correct.")

    def test_a_training_face_is_not_matched_to_itself(self) -> None:
        model = load_model(self._build())
        document = ImageDocument()
        document.load(self.train / "Alan_Turing" / "02.png")
        tool = fr.FaceRecognitionTool()
        # Plain distances: in this tiny synthetic set most eigenfaces are noise,
        # which whitening would weigh as much as the real differences.
        tool._ask_params = lambda _parent: (model, 8, False, 3)
        result = tool.run(None, document)
        self.assertNotIn("Alan_Turing/02.png", result.details["Closest 1"])
        self.assertTrue(result.details["Closest 1"].startswith("Alan Turing"))  # another photo
        self.assertIn("left out of its own matches", result.details["Training face"])
        self.assertTrue(result.message.endswith("(a training face, compared without itself)"))

    def test_a_copy_of_a_training_face_counts_as_the_same_face(self) -> None:
        model = load_model(self._build())
        # The same face saved elsewhere under another name and format. (Resized and
        # JPEG copies were checked on LFW; these synthetic faces are pure noise.)
        Image.open(self.train / "Ada_Lovelace" / "03.png").save(self.root / "renamed.bmp")
        index = model.files.index("Ada_Lovelace/03.png")
        found = same_faces(model, to_face_vector(Image.open(self.root / "renamed.bmp"), model.shape))
        self.assertIn(index, list(found))
        self.assertNotIn(index, [i for i, _ in nearest(model, model.coefficients[index], 5, exclude=found)])

    def test_accuracy_on_the_training_folder_does_not_count_self_matches(self) -> None:
        model = load_model(self._build())
        tool = fr.RecognitionAccuracyTool()
        tool._ask_params = lambda _parent: (model, 8, True, self.train)
        result = tool.run(None, ImageDocument())
        self.assertIn("18 of 18 test faces are training faces", result.details["Also in training"])

    def test_accuracy_over_a_test_folder(self) -> None:
        model = load_model(self._build())
        tool = fr.RecognitionAccuracyTool()
        tool._ask_params = lambda _parent: (model, 8, True, self.test)
        result = tool.run(None, ImageDocument())
        self.assertEqual(result.details["Accuracy"], "100.0% (6 of 6)")
        self.assertEqual(result.details["  Grace Hopper"], "2/2 (100%)")
        self.assertIn("33.3%", result.details["Always guessing"])


if __name__ == "__main__":
    unittest.main()
