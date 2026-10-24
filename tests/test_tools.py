from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image, ImageDraw
from skimage import feature

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
from forensics_app.tools.edge_visualization import EdgeVisualizationTool, superimpose_edges
from forensics_app.tools.grayscale import GrayscaleTool
from forensics_app.tools.hog import HOGTool, descriptor_layout, extract_hog, parse_hog_params
from forensics_app.tools.hog_detection import (
    DetectionParams,
    HOGPersonDetectionTool,
    default_size_range,
    match_hog_template,
    parse_search_params,
    plan_search,
    pyramid_scales,
    search_summary,
)
from forensics_app.tools.histogram import HistogramTool, compute_histogram
from forensics_app.tools.masking import MaskingTool, compose_with_mask
from forensics_app.tools.ncc import NCCTool, compute_ncc
from forensics_app.tools.registry import ToolRegistry
from forensics_app.tools.sharpen import SharpenTool, sharpen_image
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
from forensics_app.tools.template_matching import (
    Match as TemplateMatch,
    TemplateMatchingTool,
    find_template_matches,
    parse_match_params,
    rotate_template,
    superimpose_matches,
)
from forensics_app.tools.corners import CornerDetectionTool, detect_corners, parse_corner_params
from forensics_app.tools.descriptors import (
    DescriptorTool,
    extract_descriptors,
    parse_descriptor_params,
    render_keypoints,
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
    def test_pastes_overlay_colours_onto_the_base(self) -> None:
        base = Image.new("RGB", (4, 4), (10, 20, 30))
        overlay = Image.new("RGB", (4, 4), (0, 0, 0))
        overlay.putpixel((1, 1), (255, 200, 0))
        output, details = compose_with_mask(base, overlay)
        self.assertEqual(output.getpixel((1, 1)), (255, 200, 0))
        self.assertEqual(output.getpixel((0, 0)), (10, 20, 30))
        self.assertEqual(base.getpixel((1, 1)), (10, 20, 30))  # base unchanged
        self.assertEqual(details["Fill"], "overlay colours")

    def test_fills_the_mask_with_a_texture(self) -> None:
        base = Image.new("RGB", (4, 4), (10, 20, 30))
        overlay = Image.new("RGB", (4, 4), (0, 0, 0))
        overlay.putpixel((2, 2), (255, 0, 0))
        texture = Image.new("RGB", (4, 4), (1, 2, 3))
        texture.putpixel((2, 2), (9, 8, 7))
        output, details = compose_with_mask(base, overlay, texture)
        self.assertEqual(output.getpixel((2, 2)), (9, 8, 7))
        self.assertEqual(output.getpixel((0, 0)), (10, 20, 30))
        self.assertEqual(details["Fill"], "texture")

    def test_resizes_a_smaller_overlay_and_reports_it(self) -> None:
        base = Image.new("RGB", (8, 8), (0, 0, 50))
        overlay = Image.new("RGB", (2, 2), (0, 0, 0))
        overlay.putpixel((0, 0), (200, 100, 0))
        output, details = compose_with_mask(base, overlay)
        self.assertEqual(output.size, (8, 8))
        self.assertEqual(details["Overlay resized"], "yes")

    def test_rejects_an_all_black_overlay(self) -> None:
        with self.assertRaises(ValueError) as caught:
            compose_with_mask(
                Image.new("RGB", (4, 4), (1, 2, 3)),
                Image.new("RGB", (4, 4), (0, 0, 0)),
            )
        self.assertIn("No mask pixels", str(caught.exception))

    def test_inverted_mask_fills_the_background_instead(self) -> None:
        base = Image.new("RGB", (4, 4), (10, 20, 30))
        overlay = Image.new("RGB", (4, 4), (0, 0, 0))
        overlay.putpixel((1, 1), (255, 200, 0))
        texture = Image.new("RGB", (4, 4), (7, 8, 9))
        output, details = compose_with_mask(base, overlay, texture, invert=True)
        # Coat pixel stays as the base; everywhere else gets the texture (~mask).
        self.assertEqual(output.getpixel((1, 1)), (10, 20, 30))
        self.assertEqual(output.getpixel((0, 0)), (7, 8, 9))
        self.assertEqual(details["Invert mask"], "yes")

    def test_tool_cancelling_returns_none(self) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (8, 8), (10, 20, 30))
        tool = MaskingTool()
        tool._ask_images = lambda _parent: None
        self.assertIsNone(tool.run(None, document))


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


class EdgeVisualizationTests(unittest.TestCase):
    def _structured_image(self) -> Image.Image:
        arr = np.zeros((48, 48), dtype=np.uint8)
        arr[10:38, 10:38] = 220
        return Image.fromarray(arr, mode="L")

    def test_paints_red_edges_on_the_original(self) -> None:
        output, details = superimpose_edges(self._structured_image(), sigma=1.0, style="red")
        self.assertEqual(output.mode, "RGB")
        self.assertGreater(int(details["Edge pixels"].split("/")[0]), 0)
        # At least one pure-red edge pixel should appear.
        self.assertTrue(np.any(np.all(np.asarray(output) == (255, 0, 0), axis=-1)))

    def test_rejects_a_flat_image_with_no_edges(self) -> None:
        with self.assertRaises(ValueError) as caught:
            superimpose_edges(Image.new("L", (32, 32), 90), sigma=1.0, style="red")
        self.assertIn("No edges found", str(caught.exception))

    def test_tool_cancelling_returns_none(self) -> None:
        document = ImageDocument()
        document.current = self._structured_image()
        tool = EdgeVisualizationTool()
        tool._ask_params = lambda _parent: None
        self.assertIsNone(tool.run(None, document))


class SharpenTests(unittest.TestCase):
    def test_changes_a_soft_image_and_keeps_size(self) -> None:
        arr = np.zeros((32, 32), dtype=np.uint8)
        arr[:, :16] = 40
        arr[:, 16:] = 200
        source = Image.fromarray(arr, mode="L")
        output = sharpen_image(source, radius=1.0, amount=1.5)
        self.assertEqual(output.size, source.size)
        self.assertEqual(output.mode, "L")
        self.assertNotEqual(output.tobytes(), source.tobytes())

    def test_rgb_stays_bright_with_default_parameters(self) -> None:
        # Regression: skimage unsharp_mask(..., channel_axis=-1) could go black.
        arr = np.full((48, 48, 3), 120, dtype=np.uint8)
        arr[:, 24:, :] = 200
        source = Image.fromarray(arr, mode="RGB")
        output = sharpen_image(source, radius=2.0, amount=1.5)
        self.assertEqual(output.mode, "RGB")
        self.assertGreater(float(np.asarray(output).mean()), 50)

    def test_rejects_a_non_positive_radius(self) -> None:
        with self.assertRaises(ValueError) as caught:
            sharpen_image(Image.new("L", (8, 8), 90), radius=0, amount=1.0)
        self.assertIn("Radius", str(caught.exception))

    def test_tool_cancelling_returns_none(self) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (16, 16), (80, 90, 100))
        tool = SharpenTool()
        tool._ask_params = lambda _parent: None
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


class ParseHOGParamsTests(unittest.TestCase):
    def test_accepts_whole_numbers_with_spaces(self) -> None:
        self.assertEqual(parse_hog_params(" 9", "6 ", "3"), (9, 6, 3))

    def test_rejects_bad_values_and_names_the_field(self) -> None:
        cases = {
            ("a", "6", "3"): "Orientations",
            ("9", "2.5", "3"): "Pixels per cell",
            ("9", "6", "0"): "Cells per block",
            ("37", "6", "3"): "Orientations",
        }
        for fields, name in cases.items():
            with self.subTest(fields=fields), self.assertRaises(ValueError) as caught:
                parse_hog_params(*fields)
            self.assertIn(name, str(caught.exception))


class HOGTests(unittest.TestCase):
    @staticmethod
    def _noise_image(mode: str = "RGB") -> Image.Image:
        # 36x48 px = 6x8 cells of 6 px, enough for 3x3-cell blocks.
        noise = np.random.default_rng(0).integers(0, 256, (36, 48, 3)).astype(np.uint8)
        return Image.fromarray(noise, mode="RGB").convert(mode)

    def test_matches_skimage_called_directly(self) -> None:
        source = self._noise_image()
        descriptor, hog_image = extract_hog(
            source, orientations=9, pixels_per_cell=6, cells_per_block=3
        )
        expected, expected_image = feature.hog(
            np.asarray(source), orientations=9, pixels_per_cell=(6, 6),
            cells_per_block=(3, 3), visualize=True, channel_axis=2,
        )
        np.testing.assert_array_equal(descriptor, expected)
        np.testing.assert_array_equal(hog_image, expected_image)
        self.assertEqual(hog_image.shape, (36, 48))

    def test_grayscale_uses_no_channel_axis(self) -> None:
        descriptor, _ = extract_hog(
            self._noise_image("L"), orientations=4, pixels_per_cell=6, cells_per_block=3
        )
        # (6 - 3 + 1) x (8 - 3 + 1) blocks of 3x3 cells with 4 bins each.
        self.assertEqual(descriptor.size, 4 * 6 * 3 * 3 * 4)

    def test_descriptor_layout_predicts_the_footer_readout(self) -> None:
        source = self._noise_image()
        for orientations, ppc, cpb in ((9, 6, 3), (4, 5, 2), (2, 8, 1)):
            with self.subTest(params=(orientations, ppc, cpb)):
                options = dict(orientations=orientations, pixels_per_cell=ppc, cells_per_block=cpb)
                cells, length = descriptor_layout(source.size, **options)
                descriptor, _ = extract_hog(source, **options)
                self.assertEqual(length, descriptor.size)
                self.assertEqual(cells, (48 // ppc, 36 // ppc))

    def test_rejects_an_image_smaller_than_one_block(self) -> None:
        with self.assertRaises(ValueError) as caught:
            extract_hog(Image.new("L", (10, 10)), orientations=9,
                        pixels_per_cell=6, cells_per_block=3)
        self.assertIn("smaller than one HOG block", str(caught.exception))

    def test_tool_renders_original_and_hog_and_cancel_returns_none(self) -> None:
        document = ImageDocument()
        document.current = self._noise_image("RGBA")
        tool = HOGTool()
        tool._ask_params = lambda _parent, _source: None
        self.assertIsNone(tool.run(None, document))
        tool._ask_params = lambda _parent, _source: (9, 6, 3)
        result = tool.run(None, document)
        self.assertEqual(result.image.mode, "RGB")
        self.assertGreater(result.image.width, result.image.height)  # two panels in a row
        self.assertEqual(result.details["Orientations"], 9)
        self.assertEqual(result.details["Descriptor length"], 4 * 6 * 9 * 9)
        self.assertEqual(document.current.mode, "RGBA")

    def test_unavailable_on_its_own_sheet_and_unsupported_modes(self) -> None:
        document = ImageDocument()
        document.current = self._noise_image()
        tool = HOGTool()
        self.assertTrue(tool.is_available(document))
        document.apply(document.current, tool.tool_id)
        self.assertFalse(tool.is_available(document))
        self.assertIn("already shown", tool.unavailable_message(document))
        document.current = Image.new("CMYK", (40, 40))
        self.assertFalse(tool.is_available(document))


class HOGPersonDetectionTests(unittest.TestCase):
    OPTIONS = dict(orientations=9, pixels_per_cell=4, cells_per_block=2, step=4)

    @staticmethod
    def _scene(with_person: bool = True) -> Image.Image:
        """A light background, a distractor box and (optionally) a stick-figure person."""
        scene = Image.new("RGB", (160, 120), (200, 200, 200))
        draw = ImageDraw.Draw(scene)
        draw.rectangle((10, 60, 50, 80), fill=(120, 120, 120))
        if with_person:
            draw.ellipse((98, 20, 114, 36), fill=(40, 40, 40))  # head
            draw.rectangle((94, 38, 118, 72), fill=(60, 60, 60))  # body
            draw.rectangle((96, 72, 104, 100), fill=(50, 50, 50))  # legs
            draw.rectangle((108, 72, 116, 100), fill=(50, 50, 50))
        return scene

    def setUp(self) -> None:
        # The person occupies x 88..124, y 16..104; the template shows it at half size.
        self.template = self._scene().crop((88, 16, 124, 104)).resize(
            (18, 44), Image.Resampling.LANCZOS
        )

    def test_finds_an_exact_crop_at_one_scale(self) -> None:
        noise = np.random.default_rng(1).integers(0, 256, (96, 128, 3)).astype(np.uint8)
        scene = Image.fromarray(noise, mode="RGB")
        # Non-square and off the diagonal, so a swapped row/col would miss it.
        template = scene.crop((40, 16, 64, 64))
        match = match_hog_template(
            scene, template, orientations=16, pixels_per_cell=8, cells_per_block=2, step=8
        )
        self.assertEqual(match.euclidean.box, (40, 16, 24, 48))
        self.assertAlmostEqual(match.euclidean.score, 0.0)
        self.assertAlmostEqual(match.euclidean.similarity, 1.0)
        # L2-Hys makes every block unit length, so no window out-scores an exact copy.
        self.assertEqual(match.dot.box, (40, 16, 24, 48))
        self.assertEqual(match.windows, 7 * 14)

    def test_scales_find_a_person_larger_than_the_template(self) -> None:
        single = match_hog_template(self._scene(), self.template, min_similarity=0.65, **self.OPTIONS)
        self.assertFalse(single.euclidean.found)  # one size misses the 2× person

        match = match_hog_template(
            self._scene(), self.template, scales=pyramid_scales(50, 200),
            min_similarity=0.65, **self.OPTIONS,
        )
        for found in (match.euclidean, match.dot):
            self.assertTrue(found.found)
            self.assertAlmostEqual(found.scale, 2.0)
            self.assertEqual(found.box, (88, 16, 36, 88))

    def test_reports_no_person_below_the_threshold(self) -> None:
        match = match_hog_template(
            self._scene(with_person=False), self.template, scales=pyramid_scales(50, 200),
            min_similarity=0.65, **self.OPTIONS,
        )
        self.assertFalse(match.euclidean.found)
        self.assertFalse(match.dot.found)
        self.assertLess(match.dot.similarity, 0.65)

    def test_without_a_minimum_the_best_window_is_always_marked(self) -> None:
        search = parse_search_params("9", "4", "2", "4", "50", "200", "0")
        self.assertEqual(search.min_similarity, 0.0)
        document = ImageDocument()
        document.current = self._scene(with_person=False)
        tool = HOGPersonDetectionTool()
        tool._ask_params = lambda _parent, _scene: DetectionParams(
            Path("person.png"), self.template, search
        )
        result = tool.run(None, document)
        self.assertIn("(no minimum set)", result.details["Euclidean similarity"])
        self.assertIn("(no minimum set)", result.details["Dot product similarity"])
        self.assertTrue(result.message.startswith("Euclidean distance and dot product picked"))
        match = match_hog_template(
            self._scene(with_person=False), self.template, scales=search.scales, **self.OPTIONS
        )
        self.assertTrue(match.euclidean.found and match.dot.found)  # both boxes drawn

    def test_grayscale_scene_matches_too(self) -> None:
        match = match_hog_template(
            self._scene().convert("L"), self.template, scales=(2.0,), **self.OPTIONS
        )
        self.assertEqual(match.euclidean.box, (88, 16, 36, 88))

    def test_default_size_range_runs_from_a_quarter_of_the_image_to_the_largest_fit(self) -> None:
        # 50×130 template, 400 px tall image: 100 / 130 px up to 400 / 130 px.
        self.assertEqual(default_size_range((300, 400), (50, 130)), (77, 307))
        self.assertEqual(default_size_range((512, 512), (128, 128)), (100, 400))
        self.assertEqual(default_size_range((1280, 960), (64, 128)), (188, 750))
        self.assertEqual(default_size_range((1000, 400), (50, 130)), (77, 307))  # height limits
        self.assertEqual(default_size_range((100, 60), (50, 130)), (46, 46))  # template too tall
        self.assertEqual(default_size_range((300, 400), (50, 400)), (50, 100))  # 25% floored at 50%
        self.assertEqual(default_size_range((9000, 9000), (20, 20)), (1000, 1000))  # field limit

    def test_default_smallest_person_rises_until_the_search_fits_the_time_limit(self) -> None:
        # A quarter of 512 px is half this template's height: a 99 s search.
        smallest, largest = default_size_range((512, 512), (128, 256))
        self.assertGreater(smallest, 50)
        levels = plan_search(  # raises if the default search is still too slow
            (512, 512), (128, 256), pixels_per_cell=8, cells_per_block=2, step=8,
            scales=pyramid_scales(smallest, largest),
        )
        self.assertEqual(levels[-1].scale, largest / 100)

    def test_pyramid_scales_are_about_a_quarter_apart(self) -> None:
        scales = pyramid_scales(50, 200)
        self.assertEqual(len(scales), 7)
        self.assertAlmostEqual(scales[0], 0.5)
        self.assertAlmostEqual(scales[3], 1.0)
        self.assertAlmostEqual(scales[-1], 2.0)
        self.assertEqual(pyramid_scales(100, 100), (1.0,))

    def test_plan_counts_windows_and_skips_scales_that_do_not_fit(self) -> None:
        sizes = dict(pixels_per_cell=8, cells_per_block=2, step=8)
        (level,) = plan_search((24, 48), (24, 48), **sizes)
        self.assertEqual(level.windows, 1)  # the last position is included
        levels = plan_search((48, 96), (24, 48), scales=(1.0, 2.0, 4.0), **sizes)
        self.assertEqual([lv.scale for lv in levels], [1.0, 2.0])
        self.assertEqual(levels[1].size, (24, 48))
        self.assertEqual(search_summary(levels, 3, (24, 48)), "2 of 3 scales · 29 windows · under 1 s")

    def test_rejects_searches_that_cannot_run(self) -> None:
        sizes = dict(pixels_per_cell=8, cells_per_block=2)
        with self.assertRaises(ValueError) as caught:
            plan_search((20, 20), (24, 48), step=8, **sizes)
        self.assertIn("at most 42%", str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            plan_search((128, 96), (10, 48), step=8, **sizes)
        self.assertIn("smaller than one HOG block", str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            plan_search((4000, 3000), (24, 48), step=1, **sizes)
        self.assertIn("Step size", str(caught.exception))

    def test_parse_validates_fields_and_the_size_range(self) -> None:
        search = parse_search_params("16", "8", "2", "4", "50", "200", "65")
        self.assertEqual(search.step, 4)
        self.assertEqual(len(search.scales), 7)
        self.assertAlmostEqual(search.min_similarity, 0.65)
        with self.assertRaises(ValueError) as caught:
            parse_search_params("16", "8", "2", "0", "50", "200", "65")
        self.assertIn("Step size", str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            parse_search_params("16", "8", "2", "8", "300", "200", "65")
        self.assertIn("Smallest person", str(caught.exception))

    def test_tool_draws_the_sheet_and_cancel_returns_none(self) -> None:
        document = ImageDocument()
        document.current = self._scene().convert("RGBA")
        tool = HOGPersonDetectionTool()
        tool._ask_params = lambda _parent, _scene: None
        self.assertIsNone(tool.run(None, document))
        search = parse_search_params("9", "4", "2", "4", "50", "200", "65")
        params = DetectionParams(Path("person.png"), self.template, search)
        tool._ask_params = lambda _parent, _scene: params
        result = tool.run(None, document)
        self.assertEqual(result.image.mode, "RGB")
        self.assertEqual(result.details["Euclidean best"], "x=88, y=16, 36×88 (2.00×)")
        self.assertIn("person found", result.details["Dot product similarity"])
        self.assertEqual(result.details["Template"], "person.png (18×44)")
        self.assertEqual(result.message, "Both methods found a person in the same window.")
        self.assertEqual(document.current.mode, "RGBA")

    def test_unavailable_on_its_own_sheet_and_unsupported_modes(self) -> None:
        document = ImageDocument()
        document.current = self._scene()
        tool = HOGPersonDetectionTool()
        self.assertTrue(tool.is_available(document))
        document.apply(document.current, tool.tool_id)
        self.assertFalse(tool.is_available(document))
        document.current = Image.new("CMYK", (40, 40))
        self.assertFalse(tool.is_available(document))


class NCCTests(unittest.TestCase):
    def test_exact_crop_peaks_at_one_with_pad_input(self) -> None:
        # Textured scene: a constant patch makes NCC undefined (zero std).
        noise = np.random.default_rng(0).integers(0, 256, (30, 40), dtype=np.uint8)
        scene = Image.fromarray(noise, mode="L")
        template = scene.crop((12, 8, 20, 16))
        ncc = compute_ncc(scene, template)
        self.assertEqual(ncc.shape, (30, 40))
        peak = float(np.nanmax(ncc))
        self.assertGreater(peak, 0.99)
        row, col = np.unravel_index(int(np.nanargmax(ncc)), ncc.shape)
        # pad_input centres the template, so the peak is at the patch centre.
        self.assertEqual((int(row), int(col)), (8 + 4, 12 + 4))

    def test_matches_skimage_match_template(self) -> None:
        noise = np.random.default_rng(0).integers(0, 256, (48, 64), dtype=np.uint8)
        scene = Image.fromarray(noise, mode="L")
        template = scene.crop((10, 12, 26, 28))
        expected = feature.match_template(
            np.asarray(scene, dtype=np.float64) / 255.0,
            np.asarray(template, dtype=np.float64) / 255.0,
            pad_input=True,
        )
        np.testing.assert_allclose(compute_ncc(scene, template), expected)

    def test_rejects_a_template_larger_than_the_scene(self) -> None:
        with self.assertRaises(ValueError) as caught:
            compute_ncc(Image.new("L", (20, 20)), Image.new("L", (30, 10)))
        self.assertIn("larger", str(caught.exception).lower())

    def test_tool_renders_sheet_and_cancel_returns_none(self) -> None:
        document = ImageDocument()
        noise = np.random.default_rng(3).integers(0, 256, (36, 48, 3), dtype=np.uint8)
        document.current = Image.fromarray(noise, mode="RGB")
        template = document.current.crop((4, 4, 12, 12))
        tool = NCCTool()
        tool._ask_template = lambda _parent: None
        self.assertIsNone(tool.run(None, document))
        tool._ask_template = lambda _parent: template
        result = tool.run(None, document)
        self.assertEqual(result.image.mode, "RGB")
        # Slide: Original | Template | NCC  → three panels wide.
        self.assertGreater(result.image.width, result.image.height * 2)
        self.assertEqual(result.details["Function"], "skimage.feature.match_template")
        self.assertEqual(document.current.mode, "RGB")

    def test_unavailable_on_its_own_sheet(self) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (40, 40))
        tool = NCCTool()
        document.apply(document.current, tool.tool_id)
        self.assertFalse(tool.is_available(document))


class TemplateMatchingTests(unittest.TestCase):
    def test_parse_match_params_reads_angle_and_threshold_percent(self) -> None:
        angle, threshold = parse_match_params("15", "80")
        self.assertEqual(angle, 15)
        self.assertAlmostEqual(threshold, 0.8)

    def test_parse_rejects_threshold_outside_0_100(self) -> None:
        with self.assertRaises(ValueError) as caught:
            parse_match_params("0", "101")
        self.assertIn("Threshold", str(caught.exception))

    def test_rotate_keeps_content_and_changes_size_when_angled(self) -> None:
        template = Image.new("L", (10, 6), 0)
        ImageDraw.Draw(template).rectangle((2, 1, 7, 4), fill=255)
        rotated = rotate_template(template, 90)
        self.assertEqual(rotated.size, (6, 10))

    def test_finds_an_exact_unrotated_match_above_threshold(self) -> None:
        noise = np.random.default_rng(2).integers(0, 256, (40, 50), dtype=np.uint8)
        scene = Image.fromarray(noise, mode="L")
        template = scene.crop((18, 12, 26, 20))
        matches = find_template_matches(scene, template, angle=0, threshold=0.9)
        self.assertGreaterEqual(len(matches), 1)
        best = max(matches, key=lambda m: m.score)
        self.assertAlmostEqual(best.score, 1.0, places=5)
        self.assertEqual(best.box[:2], (18, 12))  # x, y

    def test_superimpose_draws_a_red_marker_on_the_match(self) -> None:
        scene = Image.new("RGB", (40, 30), (200, 200, 200))
        matches = (TemplateMatch(box=(5, 4, 10, 8), score=0.95),)
        out = superimpose_matches(scene, matches)
        pixels = np.asarray(out)
        cx, cy = matches[0].centre
        # Slide marker: hollow red circle around the match centre on a dimmed image.
        self.assertLess(int(pixels[cy, cx].mean()), 120)  # dimmed interior
        ring = pixels[cy - 4 : cy + 5, cx - 4]
        self.assertTrue(np.any(ring[:, 0] > 200))  # red outline nearby

    def test_tool_cancel_returns_none(self) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (40, 30), (50, 50, 50))
        tool = TemplateMatchingTool()
        tool._ask_template = lambda _parent: None
        self.assertIsNone(tool.run(None, document))


class CornerDetectionTests(unittest.TestCase):
    def test_parse_corner_params(self) -> None:
        method, threshold_rel, min_distance = parse_corner_params("harris", "20", "5")
        self.assertEqual(method, "harris")
        self.assertAlmostEqual(threshold_rel, 0.2)
        self.assertEqual(min_distance, 5)

    def test_parse_rejects_unknown_method(self) -> None:
        with self.assertRaises(ValueError) as caught:
            parse_corner_params("blob", "10", "1")
        self.assertIn("harris", str(caught.exception).lower())

    def test_harris_finds_corners_of_a_square(self) -> None:
        image = Image.new("L", (64, 64), 0)
        ImageDraw.Draw(image).rectangle((16, 16, 47, 47), fill=255)
        corners, response = detect_corners(
            image, method="harris", threshold_rel=0.1, min_distance=3
        )
        self.assertGreaterEqual(len(corners), 4)
        self.assertEqual(response.shape, (64, 64))

    def test_fast_finds_corners_of_a_square(self) -> None:
        # FAST needs intensity variation around candidate points; solid shapes
        # often yield an empty response, while letter strokes do not.
        image = Image.new("L", (120, 60), 0)
        ImageDraw.Draw(image).text((8, 12), "CORNER", fill=255)
        corners, response = detect_corners(
            image, method="fast", threshold_rel=0.05, min_distance=2
        )
        self.assertGreaterEqual(len(corners), 1)
        self.assertEqual(response.shape, (60, 120))

    def test_tool_cancel_returns_none(self) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (48, 48), (80, 80, 80))
        tool = CornerDetectionTool()
        tool._ask_params = lambda _parent, _source: None
        self.assertIsNone(tool.run(None, document))

    def test_tool_overlays_corners(self) -> None:
        document = ImageDocument()
        image = Image.new("RGB", (64, 64), (0, 0, 0))
        ImageDraw.Draw(image).rectangle((16, 16, 47, 47), fill=(255, 255, 255))
        document.current = image
        tool = CornerDetectionTool()
        tool._ask_params = lambda _parent, _source: ("harris", 0.1, 3)
        result = tool.run(None, document)
        self.assertEqual(result.image.mode, "RGB")
        # Slide: response map | peaks overlay → wide sheet.
        self.assertGreater(result.image.width, result.image.height)
        self.assertGreater(int(result.details["Corners"]), 0)


class DescriptorTests(unittest.TestCase):
    def test_parse_descriptor_params(self) -> None:
        orb = parse_descriptor_params("orb", "50")
        self.assertEqual(orb.method, "orb")
        self.assertEqual(orb.n_keypoints, 50)
        sift = parse_descriptor_params("sift", "6", "3", "16")
        self.assertEqual(sift.method, "sift")
        self.assertEqual(sift.n_octaves, 6)
        self.assertEqual(sift.n_scales, 3)
        self.assertAlmostEqual(sift.sigma_min, 1.6)

    def test_parse_rejects_unknown_method(self) -> None:
        with self.assertRaises(ValueError) as caught:
            parse_descriptor_params("brief", "10")
        self.assertIn("orb", str(caught.exception).lower())

    def test_orb_returns_keypoints_on_a_textured_image(self) -> None:
        noise = np.random.default_rng(0).integers(0, 256, (80, 80), dtype=np.uint8)
        image = Image.fromarray(noise, mode="L")
        keypoints, descriptors = extract_descriptors(image, method="orb", n_keypoints=20)
        self.assertGreater(len(keypoints), 0)
        self.assertEqual(len(descriptors), len(keypoints))

    def test_sift_returns_keypoints_on_a_textured_image(self) -> None:
        noise = np.random.default_rng(1).integers(0, 256, (96, 96), dtype=np.uint8)
        image = Image.fromarray(noise, mode="L")
        keypoints, descriptors = extract_descriptors(
            image, method="sift", n_octaves=4, n_scales=3, sigma_min=1.6
        )
        self.assertGreater(len(keypoints), 0)
        self.assertEqual(len(descriptors), len(keypoints))

    def test_render_keypoints_keeps_size(self) -> None:
        image = Image.new("RGB", (40, 30), (30, 30, 30))
        keypoints = np.array([[10.0, 12.0], [20.0, 8.0]])
        out = render_keypoints(image, keypoints)
        self.assertEqual(out.size, image.size)
        self.assertEqual(out.mode, "RGB")
        # Hollow red outline at the first keypoint.
        pixels = np.asarray(out)
        self.assertGreater(int(pixels[10, 12 + 4, 0]), 200)

    def test_tool_cancel_returns_none(self) -> None:
        document = ImageDocument()
        document.current = Image.new("RGB", (64, 64), (40, 40, 40))
        tool = DescriptorTool()
        tool._ask_params = lambda _parent, _source: None
        self.assertIsNone(tool.run(None, document))


if __name__ == "__main__":
    unittest.main()
