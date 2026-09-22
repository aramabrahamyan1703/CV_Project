from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from forensics_app.core import ImageDocument


class ImageDocumentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.source = Path(self.temporary_directory.name) / "source.png"
        Image.new("RGB", (8, 6), "red").save(self.source)
        self.document = ImageDocument()
        self.document.load(self.source)

    def test_load_keeps_original_and_current_images(self) -> None:
        self.assertTrue(self.document.is_loaded)
        self.assertEqual(self.document.current.size, (8, 6))
        self.assertEqual(self.document.original.getpixel((0, 0)), (255, 0, 0))

    def test_apply_undo_and_redo(self) -> None:
        self.document.apply(Image.new("RGB", (8, 6), "blue"))
        self.assertEqual(self.document.current.getpixel((0, 0)), (0, 0, 255))
        self.assertTrue(self.document.undo())
        self.assertEqual(self.document.current.getpixel((0, 0)), (255, 0, 0))
        self.assertTrue(self.document.redo())
        self.assertEqual(self.document.current.getpixel((0, 0)), (0, 0, 255))

    def test_reset_is_undoable(self) -> None:
        self.document.apply(Image.new("RGB", (8, 6), "blue"))
        self.assertTrue(self.document.reset())
        self.assertEqual(self.document.current.getpixel((0, 0)), (255, 0, 0))
        self.assertTrue(self.document.undo())
        self.assertEqual(self.document.current.getpixel((0, 0)), (0, 0, 255))

    def test_save_writes_current_image(self) -> None:
        target = Path(self.temporary_directory.name) / "result.png"
        self.document.save(target)
        self.assertTrue(target.exists())


if __name__ == "__main__":
    unittest.main()
