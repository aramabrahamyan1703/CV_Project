from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.base import ForensicsTool
from forensics_app.tools.channel_split import ChannelSplitTool
from forensics_app.tools.grayscale import GrayscaleTool
from forensics_app.tools.registry import ToolRegistry


class ToolTests(unittest.TestCase):
    def test_grayscale_returns_image_without_mutating_document(self) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (4, 3), "red")
        result = GrayscaleTool().run(None, document)  # parent is unused by this tool
        self.assertEqual(result.image.mode, "L")
        self.assertEqual(document.current.mode, "RGB")

    def test_registry_rejects_duplicate_ids(self) -> None:
        with self.assertRaises(ValueError):
            ToolRegistry([GrayscaleTool(), GrayscaleTool()])

    def test_tools_are_available_by_default(self) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (4, 3), "red")
        self.assertTrue(GrayscaleTool().is_available(document))

    def test_default_unavailable_message_names_the_tool(self) -> None:
        document = ImageDocument()
        self.assertIn(GrayscaleTool.title, GrayscaleTool().unavailable_message(document))


class ChannelSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = ChannelSplitTool()
        self.document = ImageDocument()
        self.document.current = Image.new("RGB", (40, 30), (10, 120, 250))

    def _split(self) -> None:
        result = self.tool.run(None, self.document)
        self.document.apply(result.image, self.tool.tool_id)

    def test_splits_into_three_labelled_channels(self) -> None:
        result = self.tool.run(None, self.document)
        self.assertEqual(result.image.mode, "RGB")
        self.assertGreater(result.image.width, result.image.height)  # three panels in a row
        self.assertEqual(result.details["Red min / max"], "10 / 10")
        self.assertEqual(result.details["Green min / max"], "120 / 120")
        self.assertEqual(result.details["Blue min / max"], "250 / 250")

    def test_accepts_rgba_and_drops_alpha_without_mutating_the_document(self) -> None:
        self.document.current = Image.new("RGBA", (40, 30), (10, 120, 250, 128))
        self.assertTrue(self.tool.is_available(self.document))
        result = self.tool.run(None, self.document)
        self.assertEqual(result.image.mode, "RGB")
        self.assertEqual(self.document.current.mode, "RGBA")
        self.assertEqual(result.details["Red min / max"], "10 / 10")

    def test_unavailable_for_a_single_channel_image(self) -> None:
        for mode, color in (("L", 90), ("1", 1), ("I", 90), ("F", 90.0)):
            with self.subTest(mode=mode):
                self.document.current = Image.new(mode, (40, 30), color)
                self.assertFalse(self.tool.is_available(self.document))

    def test_unavailable_after_the_image_is_converted_to_grayscale(self) -> None:
        grayscale = GrayscaleTool()
        result = grayscale.run(None, self.document)
        self.document.apply(result.image, grayscale.tool_id)
        self.assertEqual(self.document.current.mode, "L")
        self.assertFalse(self.tool.is_available(self.document))
        self.assertTrue(self.document.undo())
        self.assertTrue(self.tool.is_available(self.document))

    def test_explains_a_non_rgb_image_rather_than_the_split(self) -> None:
        self.document.current = Image.new("L", (40, 30), 90)
        message = self.tool.unavailable_message(self.document)
        self.assertIn("not RGB", message)
        self.assertIn("'L'", message)
        self.assertEqual(message.partition("\n")[0], "This image is not RGB (mode 'L').")

    def test_refuses_a_grayscale_file_from_the_moment_it_opens(self) -> None:
        with TemporaryDirectory() as directory:
            source = Path(directory) / "scan.png"
            Image.new("L", (40, 30), 90).save(source)
            self.document.load(source)
        self.assertFalse(self.tool.is_available(self.document))
        message = self.tool.unavailable_message(self.document)
        # Nothing to undo back to, so it must not suggest undoing.
        self.assertIn("Open an RGB image", message)
        self.assertNotIn("Undo", message)

    def test_advises_undo_only_when_a_colour_original_exists(self) -> None:
        self.document.original = self.document.current.copy()
        grayscale = GrayscaleTool()
        self.document.apply(grayscale.run(None, self.document).image, grayscale.tool_id)
        message = self.tool.unavailable_message(self.document)
        self.assertIn("Undo or reset", message)
        self.assertNotIn("Open an RGB image", message)

    def test_stays_available_with_no_image_so_the_window_can_prompt(self) -> None:
        self.assertTrue(self.tool.is_available(ImageDocument()))

    def test_sheet_stays_bounded_for_an_extreme_aspect_ratio(self) -> None:
        self.document.current = Image.new("RGB", (80, 20000), (1, 2, 3))
        result = self.tool.run(None, self.document)
        width, height = result.image.size
        self.assertLessEqual(width * height, self.tool.MAX_PIXELS)

    def test_unavailable_once_its_own_sheet_is_the_working_image(self) -> None:
        self.assertTrue(self.tool.is_available(self.document))
        self._split()
        self.assertFalse(self.tool.is_available(self.document))

    def test_explains_itself_rather_than_using_the_default_message(self) -> None:
        self._split()
        message = self.tool.unavailable_message(self.document)
        self.assertNotEqual(message, ForensicsTool.unavailable_message(self.tool, self.document))
        self.assertIn("already split", message)
        # The status bar shows the first line only, so it has to stand alone.
        self.assertEqual(message.partition("\n")[0], "The channels are already split.")

    def test_another_tool_stays_available_after_a_split(self) -> None:
        self._split()
        self.assertTrue(GrayscaleTool().is_available(self.document))

    def test_available_again_after_undo_and_unavailable_after_redo(self) -> None:
        self._split()
        self.assertTrue(self.document.undo())
        self.assertTrue(self.tool.is_available(self.document))
        self.assertTrue(self.document.redo())
        self.assertFalse(self.tool.is_available(self.document))

    def test_available_again_after_another_tool_runs(self) -> None:
        self._split()
        self.document.apply(Image.new("RGB", (4, 3), "blue"), GrayscaleTool.tool_id)
        self.assertTrue(self.tool.is_available(self.document))

    def test_available_again_after_reset(self) -> None:
        self.document.original = self.document.current.copy()
        self._split()
        self.assertTrue(self.document.reset())
        self.assertTrue(self.tool.is_available(self.document))

    def test_available_again_after_loading_another_image(self) -> None:
        with TemporaryDirectory() as directory:
            source = Path(directory) / "source.png"
            Image.new("RGB", (8, 6), "red").save(source)
            self._split()
            self.document.load(source)
            self.assertTrue(self.tool.is_available(self.document))


if __name__ == "__main__":
    unittest.main()
