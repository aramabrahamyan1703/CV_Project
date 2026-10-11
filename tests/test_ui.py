import unittest

from PIL import Image

from forensics_app.ui.image_view import fit_image, format_pixel, minor_step, probe_image, ruler_step
from forensics_app.ui.main_window import step_labels


class FitImageTests(unittest.TestCase):
    def test_small_image_is_centred_at_full_size(self) -> None:
        placement = fit_image((100, 50), (400, 300))
        self.assertEqual((placement.shown_width, placement.shown_height), (100, 50))
        self.assertEqual((placement.left, placement.top), (150, 125))
        self.assertEqual(placement.scale, 1.0)

    def test_large_image_is_scaled_down_inside_the_margin(self) -> None:
        placement = fit_image((4000, 2000), (1000, 800), margin=10)
        self.assertEqual((placement.shown_width, placement.shown_height), (980, 490))
        self.assertAlmostEqual(placement.scale, 0.245)

    def test_canvas_points_map_back_to_source_pixels(self) -> None:
        placement = fit_image((4000, 2000), (1000, 800), margin=10)
        self.assertEqual(placement.to_image(placement.left, placement.top), (0, 0))
        last = placement.to_image(
            placement.left + placement.shown_width - 0.01, placement.top + placement.shown_height - 0.01
        )
        self.assertEqual(last, (3999, 1999))
        self.assertIsNone(placement.to_image(placement.left - 1, placement.top))
        self.assertIsNone(placement.to_image(placement.left, placement.top + placement.shown_height))


class RulerTests(unittest.TestCase):
    def test_step_keeps_numbered_ticks_apart(self) -> None:
        self.assertEqual(ruler_step(1.0, min_gap=64), 100)
        self.assertEqual(ruler_step(0.245, min_gap=64), 500)
        self.assertEqual(ruler_step(8.0, min_gap=64), 10)

    def test_minor_step_divides_the_step(self) -> None:
        self.assertEqual(minor_step(100), 20)
        self.assertEqual(minor_step(200), 40)
        self.assertEqual(minor_step(2), 1)
        self.assertEqual(minor_step(1), 1)


class PixelReadoutTests(unittest.TestCase):
    def test_formats_each_band(self) -> None:
        self.assertEqual(format_pixel(("R", "G", "B"), (92, 61, 40)), "R 92  G 61  B 40")
        self.assertEqual(format_pixel(("L",), 128), "L 128")
        self.assertEqual(format_pixel(("F",), 0.123456), "F 0.1235")

    def test_palette_images_are_read_as_colours(self) -> None:
        image = Image.new("RGB", (2, 2), (10, 20, 30)).convert("P", palette=Image.Palette.ADAPTIVE)
        probe = probe_image(image)
        self.assertEqual(probe.getbands(), ("R", "G", "B"))
        self.assertEqual(probe.getpixel((0, 0)), (10, 20, 30))


class StepLabelTests(unittest.TestCase):
    def test_marks_past_current_and_undone_steps(self) -> None:
        rows = step_labels((None, "blur"), ("edges",), {"blur": "Gaussian blur", "edges": "Canny edges"})
        self.assertEqual(
            rows,
            [("1   Original", "past"), ("2   Gaussian blur", "current"), ("3   Canny edges", "undone")],
        )

    def test_names_a_reset_step(self) -> None:
        rows = step_labels((None, "blur", None), (), {"blur": "Gaussian blur"})
        self.assertEqual(rows[-1], ("3   Reset to original", "current"))


if __name__ == "__main__":
    unittest.main()
