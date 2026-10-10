from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image

from forensics_app.core import ImageDocument
from forensics_app.tools.base import ForensicsTool
from forensics_app.tools.channel_split import ChannelSplitTool
from forensics_app.tools.canny import CannyEdgeTool, canny_edges
from forensics_app.tools.channel_swap import ChannelSwapTool, parse_order, swap_channels
from forensics_app.tools.convolution import (
    ConvolutionTool,
    convolve_image,
    parse_kernel,
)
from forensics_app.tools.contrast_stretch import ContrastStretchTool, stretch_contrast
from forensics_app.tools.grayscale import GrayscaleTool
from forensics_app.tools.histogram import HistogramTool, compute_histogram
from forensics_app.tools.masking import MaskingTool, apply_threshold_mask
from forensics_app.tools.registry import ToolRegistry
from forensics_app.tools.skimage_filters import (
    GaussianFilterTool,
    MedianFilterTool,
    SobelFilterTool,
    UnsharpMaskTool,
    gaussian_blur,
    median_filter,
    sobel_edges,
    unsharp_mask,
)


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


class MaskingTests(unittest.TestCase):
    def test_keeps_bright_pixels_and_zeros_the_rest(self) -> None:
        source = Image.new("L", (4, 1), 0)
        source.putpixel((0, 0), 50)
        source.putpixel((1, 0), 135)
        source.putpixel((2, 0), 136)
        source.putpixel((3, 0), 200)
        output = apply_threshold_mask(source, 135, keep_above=True)
        self.assertEqual([output.getpixel((x, 0)) for x in range(4)], [0, 0, 136, 200])

    def test_inverse_keeps_dark_pixels_only(self) -> None:
        # Matches slide "masked image 2": under the threshold.
        source = Image.new("L", (4, 1), 0)
        source.putpixel((0, 0), 50)
        source.putpixel((1, 0), 135)
        source.putpixel((2, 0), 136)
        source.putpixel((3, 0), 200)
        output = apply_threshold_mask(source, 135, keep_above=False)
        self.assertEqual([output.getpixel((x, 0)) for x in range(4)], [50, 0, 0, 0])

    def test_masks_rgb_from_luminance_without_mutating_input(self) -> None:
        source = Image.new("RGB", (2, 1), (10, 10, 10))
        source.putpixel((1, 0), (200, 200, 200))
        output = apply_threshold_mask(source, 100)
        self.assertEqual(output.getpixel((0, 0)), (0, 0, 0))
        self.assertEqual(output.getpixel((1, 0)), (200, 200, 200))
        self.assertEqual(source.getpixel((0, 0)), (10, 10, 10))

    def test_rejects_when_no_pixels_under_the_threshold(self) -> None:
        # Every pixel is 200, so nothing is under 50.
        with self.assertRaises(ValueError) as caught:
            apply_threshold_mask(Image.new("L", (4, 4), 200), 50, keep_above=False)
        self.assertIn("No pixels under the threshold", str(caught.exception))

    def test_rejects_when_no_pixels_above_the_threshold(self) -> None:
        with self.assertRaises(ValueError) as caught:
            apply_threshold_mask(Image.new("L", (4, 4), 10), 50, keep_above=True)
        self.assertIn("No pixels above the threshold", str(caught.exception))

    def test_rejects_an_out_of_range_threshold(self) -> None:
        with self.assertRaises(ValueError) as caught:
            apply_threshold_mask(Image.new("L", (2, 2), 128), 300)
        self.assertIn("0 and 255", str(caught.exception))

    def test_tool_cancelling_returns_none(self) -> None:
        document = ImageDocument()
        document.current = Image.new("L", (8, 8), 180)
        tool = MaskingTool()
        tool._ask_mask_options = lambda _parent: None
        self.assertIsNone(tool.run(None, document))

    def test_tool_logs_under_threshold_choice(self) -> None:
        document = ImageDocument()
        document.current = Image.new("L", (8, 8), 40)
        document.current.putpixel((0, 0), 200)
        tool = MaskingTool()
        tool._ask_mask_options = lambda _parent: (100, False)
        result = tool.run(None, document)
        self.assertEqual(result.details["Keep"], "under threshold")
        self.assertEqual(result.details["Threshold"], 100)
        self.assertEqual(result.image.getpixel((0, 0)), 0)
        self.assertEqual(result.image.getpixel((1, 1)), 40)


class HistogramTests(unittest.TestCase):
    def test_counts_every_intensity_bin(self) -> None:
        channel = np.array([[0, 0, 255], [128, 128, 128]], dtype=np.uint8)
        counts = compute_histogram(channel)
        self.assertEqual(counts.shape, (256,))
        self.assertEqual(int(counts[0]), 2)
        self.assertEqual(int(counts[128]), 3)
        self.assertEqual(int(counts[255]), 1)

    def test_rejects_an_empty_channel(self) -> None:
        with self.assertRaises(ValueError) as caught:
            compute_histogram(np.zeros((0, 0), dtype=np.uint8))
        self.assertIn("empty", str(caught.exception))

    def test_tool_returns_a_plot_image_and_leaves_the_document_alone(self) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (20, 15), (10, 120, 250))
        result = HistogramTool().run(None, document)
        self.assertEqual(result.image.mode, "RGB")
        self.assertGreater(result.image.width, 100)
        self.assertEqual(document.current.getpixel((0, 0)), (10, 120, 250))
        self.assertEqual(result.details["Blue peak bin"], 250)

    def test_unavailable_once_its_own_plot_is_the_working_image(self) -> None:
        tool = HistogramTool()
        document = ImageDocument()
        document.current = Image.new("L", (20, 15), 90)
        self.assertTrue(tool.is_available(document))
        document.apply(tool.run(None, document).image, tool.tool_id)
        self.assertFalse(tool.is_available(document))


class ContrastStretchTests(unittest.TestCase):
    def test_spreads_a_narrow_range_to_nearly_full_scale(self) -> None:
        # Low-contrast strip: values only between 80 and 120.
        source = Image.new("L", (5, 1), 100)
        for x, value in enumerate((80, 90, 100, 110, 120)):
            source.putpixel((x, 0), value)
        output = stretch_contrast(source, low_percent=0, high_percent=100)
        values = [output.getpixel((x, 0)) for x in range(5)]
        self.assertEqual(values[0], 0)
        self.assertEqual(values[-1], 255)
        self.assertTrue(values[0] < values[2] < values[-1])

    def test_rejects_collapsed_percentiles_on_a_flat_image(self) -> None:
        with self.assertRaises(ValueError) as caught:
            stretch_contrast(Image.new("L", (8, 8), 90), 2, 98)
        self.assertIn("same value", str(caught.exception))

    def test_rejects_inverted_percentile_bounds(self) -> None:
        with self.assertRaises(ValueError) as caught:
            stretch_contrast(Image.new("L", (4, 4), 40), 90, 10)
        self.assertIn("low < high", str(caught.exception))

    def test_tool_cancelling_returns_none(self) -> None:
        document = ImageDocument()
        document.current = Image.new("L", (8, 8), 100)
        document.current.putpixel((0, 0), 40)
        document.current.putpixel((7, 7), 200)
        tool = ContrastStretchTool()
        tool._ask_percentiles = lambda _parent: None
        self.assertIsNone(tool.run(None, document))


class ParseKernelTests(unittest.TestCase):
    def test_reads_rows_of_fractions_and_decimals(self) -> None:
        kernel = parse_kernel("1/16 2/16 1/16\n0.125 0.25 0.125\n1/16 1/8 1/16")
        self.assertEqual(kernel.shape, (3, 3))
        self.assertAlmostEqual(float(kernel.sum()), 1.0)

    def test_accepts_commas_semicolons_and_blank_lines(self) -> None:
        kernel = parse_kernel("\n 0, -1, 0 ; -1, 5, -1\n\n0,-1,0 \n")
        np.testing.assert_array_equal(kernel, [[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    def test_accepts_a_rectangular_kernel(self) -> None:
        self.assertEqual(parse_kernel("1 2 3 2 1").shape, (1, 5))

    def test_rejects_ragged_rows(self) -> None:
        with self.assertRaises(ValueError) as caught:
            parse_kernel("1 2 1\n2 4\n1 2 1")
        self.assertIn("same number of entries", str(caught.exception))

    def test_rejects_an_even_side(self) -> None:
        with self.assertRaises(ValueError) as caught:
            parse_kernel("1 1\n1 1")
        self.assertIn("odd", str(caught.exception))

    def test_rejects_non_numbers_and_division_by_zero(self) -> None:
        for text in ("1 x 1", "1 1/0 1"):
            with self.assertRaises(ValueError) as caught:
                parse_kernel(text)
            self.assertIn("not a number", str(caught.exception))

    def test_rejects_an_empty_kernel(self) -> None:
        with self.assertRaises(ValueError):
            parse_kernel("  \n ; ")

    def test_rejects_an_oversized_kernel(self) -> None:
        row = " ".join(["1"] * 27)
        with self.assertRaises(ValueError) as caught:
            parse_kernel("\n".join([row] * 27))
        self.assertIn("largest side", str(caught.exception))


class ConvolveImageTests(unittest.TestCase):
    def test_flips_the_kernel_like_true_convolution(self) -> None:
        # An impulse convolved with k reproduces k unflipped around it;
        # correlation would give the kernel rotated by 180 degrees.
        source = Image.new("L", (5, 5), 0)
        source.putpixel((2, 2), 1)
        kernel = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64) * 20
        output = convolve_image(source, kernel, normalize=False, border="constant")
        patch = np.asarray(output)[1:4, 1:4]
        np.testing.assert_array_equal(patch, kernel.astype(np.uint8))

    def test_identity_kernel_changes_nothing_and_leaves_input_alone(self) -> None:
        source = Image.new("RGB", (6, 4), (10, 120, 250))
        source.putpixel((3, 2), (200, 0, 30))
        before = np.asarray(source).copy()
        output = convolve_image(source, parse_kernel("0 0 0\n0 1 0\n0 0 0"))
        np.testing.assert_array_equal(np.asarray(output), before)
        np.testing.assert_array_equal(np.asarray(source), before)

    def test_box_blur_keeps_a_flat_image_flat_up_to_the_border(self) -> None:
        # Zero padding would darken the edges; reflect must not.
        source = Image.new("L", (7, 5), 100)
        output = convolve_image(source, parse_kernel("1 1 1\n1 1 1\n1 1 1"))
        self.assertTrue((np.asarray(output) == 100).all())

    def test_zero_padding_darkens_the_corner(self) -> None:
        source = Image.new("L", (7, 5), 90)
        output = convolve_image(source, np.ones((3, 3)), border="constant")
        self.assertEqual(output.getpixel((0, 0)), 40)  # 4 of 9 neighbours inside

    def test_normalize_skips_a_zero_sum_kernel(self) -> None:
        output = convolve_image(Image.new("L", (5, 5), 80), parse_kernel("0 1 0\n1 -4 1\n0 1 0"))
        self.assertTrue((np.asarray(output) == 0).all())

    def test_abs_shows_both_sides_of_a_step_edge(self) -> None:
        source = Image.new("L", (6, 1), 0)
        for x in range(3, 6):
            source.putpixel((x, 0), 200)
        kernel = parse_kernel("-1 0 1")
        clipped = np.asarray(convolve_image(source, kernel, border="edge", negatives="clip"))[0]
        absolute = np.asarray(convolve_image(source, kernel, border="edge", negatives="abs"))[0]
        # Convolution flips [-1 0 1], so the rising edge gives -200 here.
        self.assertEqual(list(clipped), [0, 0, 0, 0, 0, 0])
        self.assertEqual(list(absolute), [0, 0, 200, 200, 0, 0])

    def test_keeps_alpha_and_rgba_mode(self) -> None:
        source = Image.new("RGBA", (4, 4), (50, 100, 150, 77))
        output = convolve_image(source, parse_kernel("1 2 1\n2 4 2\n1 2 1"))
        self.assertEqual(output.mode, "RGBA")
        self.assertEqual(output.getpixel((1, 1)), (50, 100, 150, 77))

    def test_handles_a_one_pixel_tall_image(self) -> None:
        source = Image.new("L", (5, 1), 60)
        output = convolve_image(source, np.ones((3, 3)))
        self.assertEqual(output.size, (5, 1))
        self.assertTrue((np.asarray(output) == 60).all())

    def test_rejects_an_unsupported_mode(self) -> None:
        with self.assertRaises(ValueError):
            convolve_image(Image.new("CMYK", (4, 4)), np.ones((3, 3)))


class ConvolutionToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = ConvolutionTool()
        self.document = ImageDocument()
        self.document.current = Image.new("L", (8, 8), 100)

    def test_cancelling_the_dialog_returns_none(self) -> None:
        self.tool._ask_kernel_options = lambda _parent: None
        self.assertIsNone(self.tool.run(None, self.document))

    def test_records_the_kernel_and_options_in_details(self) -> None:
        kernel = parse_kernel("1 2 1\n2 4 2\n1 2 1")
        self.tool._ask_kernel_options = lambda _parent: (kernel, True, "reflect", "clip")
        result = self.tool.run(None, self.document)
        self.assertEqual(result.details["Kernel size"], "3×3")
        self.assertIn("1 / 16", result.details["Normalization"])
        self.assertEqual(result.details["Border"], "Reflect")
        self.assertEqual(result.image.getpixel((4, 4)), 100)
        self.assertEqual(self.document.current.getpixel((4, 4)), 100)

    def test_stays_available_on_its_own_output(self) -> None:
        self.tool._ask_kernel_options = lambda _parent: (np.ones((3, 3)), True, "reflect", "clip")
        self.document.apply(self.tool.run(None, self.document).image, self.tool.tool_id)
        self.assertTrue(self.tool.is_available(self.document))

    def test_unavailable_for_an_unsupported_mode(self) -> None:
        self.document.current = Image.new("CMYK", (4, 4))
        self.assertFalse(self.tool.is_available(self.document))
        self.assertIn("CMYK", self.tool.unavailable_message(self.document))


class GaussianFilterTests(unittest.TestCase):
    def test_spreads_a_bright_dot_and_keeps_its_peak_in_place(self) -> None:
        source = Image.new("L", (9, 9), 0)
        source.putpixel((4, 4), 255)
        output = np.asarray(gaussian_blur(source, 1.0))
        self.assertLess(int(output[4, 4]), 255)
        self.assertGreater(int(output[4, 5]), 0)
        self.assertEqual(np.unravel_index(output.argmax(), output.shape), (4, 4))

    def test_larger_sigma_blurs_more(self) -> None:
        source = Image.new("L", (15, 15), 0)
        source.putpixel((7, 7), 255)
        mild = np.asarray(gaussian_blur(source, 1.0))
        strong = np.asarray(gaussian_blur(source, 3.0))
        self.assertGreater(int(mild[7, 7]), int(strong[7, 7]))

    def test_keeps_a_flat_colour_image_and_its_alpha(self) -> None:
        source = Image.new("RGBA", (6, 6), (40, 120, 220, 90))
        output = gaussian_blur(source, 2.0)
        self.assertEqual(output.mode, "RGBA")
        self.assertEqual(output.getpixel((0, 0)), (40, 120, 220, 90))

    def test_rejects_a_non_positive_sigma(self) -> None:
        with self.assertRaises(ValueError) as caught:
            gaussian_blur(Image.new("L", (4, 4), 50), 0)
        self.assertIn("Sigma", str(caught.exception))

    def test_tool_records_sigma_and_cancel_returns_none(self) -> None:
        document = ImageDocument()
        document.current = Image.new("L", (8, 8), 100)
        tool = GaussianFilterTool()
        tool._ask_sigma = lambda _parent: None
        self.assertIsNone(tool.run(None, document))
        tool._ask_sigma = lambda _parent: 1.5
        result = tool.run(None, document)
        self.assertEqual(result.details["Sigma"], 1.5)
        self.assertEqual(result.details["Function"], "skimage.filters.gaussian")


class MedianFilterTests(unittest.TestCase):
    def test_removes_an_isolated_noise_pixel(self) -> None:
        source = Image.new("L", (7, 7), 80)
        source.putpixel((3, 3), 255)  # "salt"
        output = median_filter(source, 3)
        self.assertTrue((np.asarray(output) == 80).all())

    def test_keeps_a_straight_edge_sharp(self) -> None:
        source = Image.new("L", (8, 8), 0)
        for x in range(4, 8):
            for y in range(8):
                source.putpixel((x, y), 200)
        np.testing.assert_array_equal(np.asarray(median_filter(source, 3)), np.asarray(source))

    def test_filters_each_colour_channel_and_keeps_alpha(self) -> None:
        source = Image.new("RGBA", (5, 5), (10, 20, 30, 128))
        source.putpixel((2, 2), (255, 0, 255, 128))
        self.assertEqual(median_filter(source, 3).getpixel((2, 2)), (10, 20, 30, 128))

    def test_rejects_an_even_window(self) -> None:
        with self.assertRaises(ValueError) as caught:
            median_filter(Image.new("L", (4, 4), 50), 4)
        self.assertIn("odd", str(caught.exception))

    def test_tool_cancelling_returns_none(self) -> None:
        document = ImageDocument()
        document.current = Image.new("L", (8, 8), 100)
        tool = MedianFilterTool()
        tool._ask_size = lambda _parent: None
        self.assertIsNone(tool.run(None, document))


class SobelFilterTests(unittest.TestCase):
    def test_marks_a_vertical_edge_and_nothing_else(self) -> None:
        source = Image.new("L", (8, 8), 0)
        for x in range(4, 8):
            for y in range(8):
                source.putpixel((x, y), 200)
        output, peak = sobel_edges(source)
        row = np.asarray(output)[4]
        self.assertGreater(peak, 0)
        self.assertEqual(int(row.max()), 255)
        self.assertEqual(int(row[0]), 0)
        self.assertEqual(int(row[7]), 0)
        self.assertGreater(int(row[3]), 0)

    def test_flat_image_gives_a_black_map(self) -> None:
        output, peak = sobel_edges(Image.new("RGB", (6, 6), (90, 90, 90)))
        self.assertEqual(peak, 0.0)
        self.assertTrue((np.asarray(output) == 0).all())

    def test_tool_returns_a_grayscale_map_without_mutating_the_document(self) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (8, 8), (10, 20, 30))
        document.current.putpixel((4, 4), (250, 250, 250))
        result = SobelFilterTool().run(None, document)
        self.assertEqual(result.image.mode, "L")
        self.assertEqual(document.current.mode, "RGB")
        self.assertEqual(document.current.getpixel((4, 4)), (250, 250, 250))

    def test_unavailable_for_an_unsupported_mode(self) -> None:
        document = ImageDocument()
        document.current = Image.new("CMYK", (4, 4))
        self.assertFalse(SobelFilterTool().is_available(document))


class UnsharpMaskTests(unittest.TestCase):
    @staticmethod
    def _step_image() -> Image.Image:
        source = Image.new("L", (12, 4), 60)
        source.paste(190, (6, 0, 12, 4))
        return source

    def test_overshoots_both_sides_of_an_edge(self) -> None:
        row = np.asarray(unsharp_mask(self._step_image(), 1.0, 1.0))[2]
        self.assertLess(int(row[5]), 60)  # dark side gets darker
        self.assertGreater(int(row[6]), 190)  # bright side gets brighter
        self.assertEqual(int(row[0]), 60)  # far from the edge: unchanged
        self.assertEqual(int(row[11]), 190)

    def test_larger_amount_sharpens_more(self) -> None:
        mild = np.asarray(unsharp_mask(self._step_image(), 1.0, 0.5))[2]
        strong = np.asarray(unsharp_mask(self._step_image(), 1.0, 2.0))[2]
        self.assertLess(int(strong[5]), int(mild[5]))
        self.assertGreater(int(strong[6]), int(mild[6]))

    def test_zero_amount_changes_nothing(self) -> None:
        source = self._step_image()
        np.testing.assert_array_equal(
            np.asarray(unsharp_mask(source, 2.0, 0.0)), np.asarray(source)
        )

    def test_keeps_a_flat_colour_image_and_its_alpha(self) -> None:
        source = Image.new("RGBA", (6, 6), (40, 120, 220, 90))
        output = unsharp_mask(source, 2.0, 1.5)
        self.assertEqual(output.mode, "RGBA")
        self.assertEqual(output.getpixel((3, 3)), (40, 120, 220, 90))

    def test_sharpens_each_colour_channel_like_a_grayscale_image(self) -> None:
        # Guards the channel_axis=2 workaround: each RGB band must come out
        # exactly as if that band were sharpened on its own.
        rgb = np.random.default_rng(1).integers(0, 256, (10, 14, 3)).astype(np.uint8)
        output = np.asarray(unsharp_mask(Image.fromarray(rgb, mode="RGB"), 1.5, 1.0))
        for band in range(3):
            alone = unsharp_mask(Image.fromarray(rgb[..., band], mode="L"), 1.5, 1.0)
            np.testing.assert_array_equal(output[..., band], np.asarray(alone))

    def test_rejects_out_of_range_settings(self) -> None:
        with self.assertRaises(ValueError) as caught:
            unsharp_mask(self._step_image(), 0, 1.0)
        self.assertIn("Radius", str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            unsharp_mask(self._step_image(), 1.0, -1.0)
        self.assertIn("Amount", str(caught.exception))

    def test_tool_records_settings_and_cancel_returns_none(self) -> None:
        document = ImageDocument()
        document.current = self._step_image()
        tool = UnsharpMaskTool()
        tool._ask_options = lambda _parent: None
        self.assertIsNone(tool.run(None, document))
        tool._ask_options = lambda _parent: (1.5, 2.0)
        result = tool.run(None, document)
        self.assertEqual(result.details["Radius (sigma)"], 1.5)
        self.assertEqual(result.details["Amount"], 2.0)
        self.assertEqual(document.current.getpixel((5, 2)), 60)


class CannyEdgeTests(unittest.TestCase):
    @staticmethod
    def _step_image() -> Image.Image:
        source = Image.new("L", (20, 20), 30)
        for x in range(10, 20):
            for y in range(20):
                source.putpixel((x, y), 220)
        return source

    def test_finds_a_thin_line_on_a_step_edge(self) -> None:
        # A mid-grey column gives the step a single centre; a step that falls
        # between two pixels ties them, and Canny then keeps both.
        source = self._step_image()
        source.paste(125, (10, 0, 11, 20))
        edges = np.asarray(canny_edges(source, 1.0))
        self.assertEqual(set(np.unique(edges)), {0, 255})
        # One thin line right at the step, nothing in the flat halves.
        self.assertEqual(list(np.nonzero(edges[10])[0]), [10])

    def test_flat_image_has_no_edges(self) -> None:
        edges = canny_edges(Image.new("RGB", (16, 16), (200, 200, 200)), 1.0)
        self.assertFalse(np.asarray(edges).any())

    def test_larger_sigma_suppresses_noise_edges(self) -> None:
        noise = np.random.default_rng(0).integers(0, 256, (40, 40)).astype(np.uint8)
        source = Image.fromarray(noise, mode="L")
        fine = np.count_nonzero(np.asarray(canny_edges(source, 1.0)))
        smooth = np.count_nonzero(np.asarray(canny_edges(source, 4.0)))
        self.assertGreater(fine, smooth)

    def test_rejects_a_non_positive_sigma(self) -> None:
        with self.assertRaises(ValueError) as caught:
            canny_edges(self._step_image(), 0)
        self.assertIn("Sigma", str(caught.exception))

    def test_tool_records_sigma_and_cancel_returns_none(self) -> None:
        document = ImageDocument()
        document.current = self._step_image().convert("RGBA")
        tool = CannyEdgeTool()
        tool._ask_sigma = lambda _parent: None
        self.assertIsNone(tool.run(None, document))
        tool._ask_sigma = lambda _parent: 2.0
        result = tool.run(None, document)
        self.assertEqual(result.image.mode, "L")
        self.assertEqual(result.details["Sigma"], 2.0)
        self.assertEqual(result.details["Function"], "skimage.feature.canny")
        self.assertEqual(document.current.mode, "RGBA")

    def test_unavailable_for_an_unsupported_mode(self) -> None:
        document = ImageDocument()
        document.current = Image.new("CMYK", (4, 4))
        self.assertFalse(CannyEdgeTool().is_available(document))


if __name__ == "__main__":
    unittest.main()
