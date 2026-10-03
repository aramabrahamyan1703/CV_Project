from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.base import ForensicsTool
from forensics_app.tools.channel_split import ChannelSplitTool
from forensics_app.tools.channel_swap import ChannelSwapTool, parse_order, swap_channels
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



class ParseOrderTests(unittest.TestCase):
    """The pure validation behind the dialog, tested without a display."""

    def test_reads_the_letters_as_the_source_of_each_output_slot(self) -> None:
        self.assertEqual(parse_order("BGR"), (2, 1, 0))
        self.assertEqual(parse_order("GBR"), (1, 2, 0))  # a rotation, not a reversal
        self.assertEqual(parse_order("RBG"), (0, 2, 1))

    def test_accepts_lowercase_and_surrounding_whitespace(self) -> None:
        self.assertEqual(parse_order("  bgr\n"), (2, 1, 0))

    def test_rejects_a_wrong_number_of_letters(self) -> None:
        for text in ("", "BG", "BGRB"):
            with self.subTest(text=text):
                with self.assertRaises(ValueError) as caught:
                    parse_order(text)
                self.assertIn("three letters", str(caught.exception))

    def test_rejects_a_letter_outside_rgb(self) -> None:
        with self.assertRaises(ValueError) as caught:
            parse_order("BGX")
        self.assertIn("only the letters R, G and B", str(caught.exception))
        self.assertIn("'X'", str(caught.exception))

    def test_rejects_a_repeated_channel(self) -> None:
        with self.assertRaises(ValueError) as caught:
            parse_order("RRG")
        self.assertIn("exactly once", str(caught.exception))

    def test_rejects_the_identity_order(self) -> None:
        with self.assertRaises(ValueError) as caught:
            parse_order("RGB")
        self.assertIn("change nothing", str(caught.exception))


class SwapChannelsTests(unittest.TestCase):
    """The pure pixel work, independent of the tool and its dialog."""

    def test_moves_each_source_channel_into_the_named_slot(self) -> None:
        source = Image.new("RGB", (4, 3), (10, 120, 250))
        self.assertEqual(swap_channels(source, (2, 1, 0)).getpixel((0, 0)), (250, 120, 10))
        # A rotation would look identical to a reversal on a symmetric input,
        # so the three distinct values above are what make this meaningful.
        self.assertEqual(swap_channels(source, (1, 2, 0)).getpixel((0, 0)), (120, 250, 10))

    def test_applying_a_reversal_twice_restores_the_original(self) -> None:
        source = Image.new("RGB", (4, 3), (10, 120, 250))
        restored = swap_channels(swap_channels(source, (2, 1, 0)), (2, 1, 0))
        self.assertEqual(restored.tobytes(), source.tobytes())

    def test_keeps_the_alpha_band_and_the_rgba_mode(self) -> None:
        source = Image.new("RGBA", (4, 3), (10, 120, 250, 64))
        output = swap_channels(source, (2, 1, 0))
        self.assertEqual(output.mode, "RGBA")
        self.assertEqual(output.getpixel((0, 0)), (250, 120, 10, 64))
        self.assertEqual(output.split()[3].tobytes(), source.split()[3].tobytes())


class ChannelSwapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = ChannelSwapTool()
        self.document = ImageDocument()
        self.document.current = Image.new("RGB", (40, 30), (10, 120, 250))

    def _run(self, order: str | None) -> object:
        """Run the tool with the dialog answered by ``order`` (``None`` cancels)."""
        self.tool._ask_order = lambda _parent: None if order is None else parse_order(order)
        return self.tool.run(None, self.document)  # parent is only used by the dialog

    def test_swaps_the_channels_without_mutating_the_document(self) -> None:
        result = self._run("BGR")
        self.assertEqual(result.image.mode, "RGB")
        self.assertEqual(result.image.size, (40, 30))
        self.assertEqual(result.image.getpixel((0, 0)), (250, 120, 10))
        self.assertEqual(self.document.current.getpixel((0, 0)), (10, 120, 250))

    def test_details_name_the_source_of_each_output_slot(self) -> None:
        result = self._run("GBR")
        self.assertEqual(result.details["New order"], "GBR")
        self.assertEqual(result.details["Output R"], "source G")
        self.assertEqual(result.details["Output G"], "source B")
        self.assertEqual(result.details["Output B"], "source R")
        self.assertEqual(result.details["Output mode"], "RGB")
        # The reported mapping has to agree with the pixels it describes.
        self.assertEqual(result.image.getpixel((0, 0)), (120, 250, 10))

    def test_message_names_the_order_that_was_applied(self) -> None:
        self.assertIn("BRG", self._run("BRG").message)

    def test_cancelling_the_dialog_returns_none(self) -> None:
        self.assertIsNone(self._run(None))

    def test_keeps_transparency_for_an_rgba_image(self) -> None:
        self.document.current = Image.new("RGBA", (40, 30), (10, 120, 250, 64))
        self.assertTrue(self.tool.is_available(self.document))
        result = self._run("BGR")
        self.assertEqual(result.details["Output mode"], "RGBA")
        self.assertEqual(result.image.getpixel((0, 0)), (250, 120, 10, 64))

    def test_expands_a_palette_image_to_rgb(self) -> None:
        self.document.current = Image.new("RGB", (40, 30), (10, 120, 250)).convert("P")
        self.assertTrue(self.tool.is_available(self.document))
        result = self._run("BGR")
        self.assertEqual(result.image.mode, "RGB")
        # Building the palette quantised the colour, so the expectation comes
        # from the palette image itself rather than the colour it was made from.
        red, green, blue = self.document.current.convert("RGB").getpixel((0, 0))
        self.assertEqual(result.image.getpixel((0, 0)), (blue, green, red))

    def test_stays_available_on_its_own_output_so_it_can_be_undone_by_hand(self) -> None:
        # Unlike the split, a swap is meaningful twice over: BGR then BGR again
        # is how a user gets back to RGB without reaching for Undo.
        result = self._run("BGR")
        self.document.apply(result.image, self.tool.tool_id)
        self.assertTrue(self.tool.is_available(self.document))
        restored = self._run("BGR")
        self.assertEqual(restored.image.getpixel((0, 0)), (10, 120, 250))

    def test_unavailable_for_a_single_channel_image(self) -> None:
        for mode, color in (("L", 90), ("1", 1), ("I", 90), ("F", 90.0)):
            with self.subTest(mode=mode):
                self.document.current = Image.new(mode, (40, 30), color)
                self.assertFalse(self.tool.is_available(self.document))

    def test_unavailable_after_the_image_is_converted_to_grayscale(self) -> None:
        grayscale = GrayscaleTool()
        self.document.apply(grayscale.run(None, self.document).image, grayscale.tool_id)
        self.assertEqual(self.document.current.mode, "L")
        self.assertFalse(self.tool.is_available(self.document))
        self.assertTrue(self.document.undo())
        self.assertTrue(self.tool.is_available(self.document))

    def test_explains_a_non_rgb_image(self) -> None:
        self.document.current = Image.new("L", (40, 30), 90)
        message = self.tool.unavailable_message(self.document)
        self.assertNotEqual(message, ForensicsTool.unavailable_message(self.tool, self.document))
        self.assertIn("'L'", message)
        # The status bar shows the first line only, so it has to stand alone.
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

if __name__ == "__main__":
    unittest.main()
