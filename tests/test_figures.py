"""The app-wide rule: tools that analyse photos refuse a figure as their input.

A figure is a picture about an image (the eigenface grid, a detection sheet, a
BoW histogram…). Running face detection on the eigenface grid "finds" the
eigenfaces, so while a figure is the working image only tools that accept
figures, or need no image, are available.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools import build_tool_registry

FIGURE_TOOLS = {
    "channel_split", "histogram", "edge_visualization", "ncc", "template_matching", "hog",
    "hog_person_detection", "corners", "descriptors", "feature_matching", "image_lookup",
    "face_detection", "face_detection_webcam", "eigenface_builder", "face_recognition",
    "recognition_accuracy", "bow_codebook", "bow_histogram", "bow_classify",
}


class FigureInputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = build_tool_registry()
        self.tools = {tool.tool_id: tool for tool in self.registry.all()}
        self._tmp = TemporaryDirectory()
        self.photo = Path(self._tmp.name) / "photo.png"
        Image.new("RGB", (64, 48), (120, 90, 60)).save(self.photo)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def available(self, tool_id: str, document: ImageDocument) -> bool:
        return self.registry.is_available(self.tools[tool_id], document)

    def test_exactly_the_figure_tools_are_marked(self) -> None:
        marked = {tool.tool_id for tool in self.registry.all() if tool.produces_figure}
        self.assertEqual(marked, FIGURE_TOOLS)

    def test_face_detection_refuses_the_eigenface_grid(self) -> None:
        document = ImageDocument()
        document.load(self.photo)
        self.assertTrue(self.available("face_detection", document))
        document.apply(Image.new("RGB", (64, 48)), "eigenface_builder")
        self.assertFalse(self.available("face_detection", document))
        message = self.registry.unavailable_message(self.tools["face_detection"], document)
        self.assertIn("figure made by Eigenface builder", message)
        self.assertIn("Undo", message)

    def test_every_photo_tool_refuses_a_figure_but_inspect_does_not(self) -> None:
        document = ImageDocument()
        document.load(self.photo)
        document.apply(Image.new("RGB", (64, 48)), "hog")
        for tool in self.registry.all():
            with self.subTest(tool=tool.tool_id):
                blocked = self.registry.blocked_by_figure(tool, document)
                if tool.requires_image and not tool.accepts_figures:
                    self.assertIsNotNone(blocked)
                    self.assertFalse(self.registry.is_available(tool, document))
                else:
                    self.assertIsNone(blocked)
        self.assertTrue(self.available("image_info", document))

    def test_tools_that_need_no_image_still_run_on_a_figure(self) -> None:
        document = ImageDocument()
        document.apply(Image.new("RGB", (64, 48)), "eigenface_builder")
        for tool_id in ("eigenface_builder", "recognition_accuracy", "face_detection_webcam", "bow_codebook"):
            self.assertIsNone(self.registry.blocked_by_figure(self.tools[tool_id], document))

    def test_an_edited_photo_is_still_a_photo(self) -> None:
        document = ImageDocument()
        document.load(self.photo)
        document.apply(Image.new("L", (64, 48)), "grayscale")
        self.assertIsNone(self.registry.figure_maker(document))
        self.assertTrue(self.available("face_detection", document))

    def test_undo_redo_and_reset_follow_the_figure(self) -> None:
        document = ImageDocument()
        document.load(self.photo)
        document.apply(Image.new("RGB", (64, 48)), "face_detection")
        self.assertFalse(self.available("face_recognition", document))
        document.undo()
        self.assertTrue(self.available("face_recognition", document))
        document.redo()
        self.assertFalse(self.available("face_recognition", document))
        document.reset()
        self.assertTrue(self.available("face_recognition", document))

    def test_a_document_started_by_a_figure_stays_one_after_reset(self) -> None:
        document = ImageDocument()  # nothing open: the builder's grid starts the document
        document.apply(Image.new("RGB", (64, 48)), "eigenface_builder")
        self.assertFalse(self.available("face_detection", document))
        document.apply(Image.new("L", (64, 48)), "grayscale")  # refused in the app; force it here
        document.reset()
        self.assertEqual(document.last_tool_id, "eigenface_builder")
        self.assertFalse(self.available("face_detection", document))


if __name__ == "__main__":
    unittest.main()
