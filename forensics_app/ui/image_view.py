"""Canvas that shows the working image fitted to the window, with pixel rulers.

The rulers count source-image pixels, not screen pixels, and moving the
pointer over the image reports the pixel under it -- position and value,
read from the full-resolution image rather than the scaled preview.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk

from . import theme


MARGIN = 16  # grey border kept around a fitted image
RULER_SIZE = 22  # height of the top ruler
MIN_LABEL_GAP = 64  # screen pixels between numbered ruler ticks
PREVIEW_CACHE_SIZE = 2  # current + original, so "hold to compare" is instant


@dataclass(frozen=True)
class Placement:
    """Where a fitted image lands on the canvas, in canvas pixels."""

    left: int
    top: int
    shown_width: int
    shown_height: int
    image_width: int
    image_height: int

    @property
    def scale(self) -> float:
        return self.shown_width / self.image_width

    def to_image(self, x: float, y: float) -> tuple[int, int] | None:
        """The source pixel under canvas point ``(x, y)``, or ``None`` off the image."""
        dx, dy = x - self.left, y - self.top
        if not (0 <= dx < self.shown_width and 0 <= dy < self.shown_height):
            return None
        return (
            int(dx * self.image_width / self.shown_width),
            int(dy * self.image_height / self.shown_height),
        )

    def canvas_x(self, image_x: float) -> float:
        return self.left + image_x * self.shown_width / self.image_width

    def canvas_y(self, image_y: float) -> float:
        return self.top + image_y * self.shown_height / self.image_height


def fit_image(image_size: tuple[int, int], area_size: tuple[int, int], margin: int = MARGIN) -> Placement:
    """Scale an image down (never up) to fit ``area_size`` less ``margin``, centred."""
    width, height = image_size
    area_width, area_height = area_size
    room_width = max(area_width - 2 * margin, 1)
    room_height = max(area_height - 2 * margin, 1)
    scale = min(1.0, room_width / width, room_height / height)
    shown_width = max(1, round(width * scale))
    shown_height = max(1, round(height * scale))
    return Placement(
        left=(area_width - shown_width) // 2,
        top=(area_height - shown_height) // 2,
        shown_width=shown_width,
        shown_height=shown_height,
        image_width=width,
        image_height=height,
    )


def ruler_step(scale: float, min_gap: float = MIN_LABEL_GAP) -> int:
    """Smallest 1/2/5 × 10ⁿ pixel step whose numbered ticks sit ``min_gap`` apart."""
    magnitude = 1
    while True:
        for base in (1, 2, 5):
            step = base * magnitude
            if step * scale >= min_gap:
                return step
        magnitude *= 10


def minor_step(step: int) -> int:
    """Unnumbered tick spacing between the numbered ones."""
    if step % 5 == 0:
        return max(step // 5, 1)
    return max(step // 2, 1)


def probe_image(image: Image.Image) -> Image.Image:
    """The image to read pixel values from: palette and 1-bit images as colours."""
    if image.mode == "P":
        return image.convert("RGBA" if "transparency" in image.info else "RGB")
    if image.mode == "1":
        return image.convert("L")
    return image


def format_pixel(bands: tuple[str, ...], value: object) -> str:
    """``("R", "G", "B"), (92, 61, 40)`` -> ``"R 92  G 61  B 40"``."""
    values = value if isinstance(value, tuple) else (value,)
    parts = []
    for band, number in zip(bands, values):
        text = f"{number:.4g}" if isinstance(number, float) else str(number)
        parts.append(f"{band} {text}")
    return "  ".join(parts)


class ImageView(ttk.Frame):
    """Fitted image on neutral grey, with rulers and a live pixel probe.

    ``on_probe`` receives a readout such as ``"x 212  y 148   R 92  G 61  B 40"``
    while the pointer is over the image, and ``""`` otherwise.
    """

    def __init__(
        self,
        parent: tk.Misc,
        fonts: theme.Fonts,
        on_probe: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(parent, style="Ruler.TFrame")
        self.fonts = fonts
        self.on_probe = on_probe
        side = max(fonts.mono_small.measure("0000") + 14, RULER_SIZE)

        ttk.Label(self, text="px", style="Ruler.TLabel", anchor="center").grid(
            row=0, column=0, sticky="nsew"
        )
        ruler = {"background": theme.RULER, "borderwidth": 0, "highlightthickness": 0}
        self.top_ruler = tk.Canvas(self, height=RULER_SIZE, **ruler)
        self.top_ruler.grid(row=0, column=1, sticky="ew")
        self.left_ruler = tk.Canvas(self, width=side, **ruler)
        self.left_ruler.grid(row=1, column=0, sticky="ns")
        self.canvas = tk.Canvas(
            self,
            background=theme.CANVAS,
            borderwidth=0,
            highlightthickness=0,
        )
        self.canvas.grid(row=1, column=1, sticky="nsew")
        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)

        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Leave>", lambda _event: self._clear_probe())

        self._source: Image.Image | None = None
        self._probe: Image.Image | None = None
        self._placement: Placement | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._resize_job: str | None = None
        self._previews: list[tuple[Image.Image, tuple[int, int], Image.Image]] = []
        self.show(None)

    @property
    def zoom(self) -> float | None:
        """Displayed size relative to the source, e.g. ``0.5``; ``None`` when empty."""
        return self._placement.scale if self._placement is not None else None

    @property
    def placement(self) -> Placement | None:
        return self._placement

    def show(self, image: Image.Image | None) -> None:
        self._source = image
        self._probe = probe_image(image) if image is not None else None
        self._render()

    def _on_resize(self, _event: tk.Event) -> None:
        if self._resize_job is not None:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(50, self._render)

    def _render(self) -> None:
        self._resize_job = None
        for canvas in (self.canvas, self.top_ruler, self.left_ruler):
            canvas.delete("all")
        width = max(self.canvas.winfo_width(), 1)
        height = max(self.canvas.winfo_height(), 1)

        if self._source is None:
            self._placement = None
            self._photo = None
            self.canvas.configure(cursor="")
            self.canvas.create_text(
                width // 2,
                height // 2 - 12,
                text="Open an image to begin",
                fill=theme.CANVAS_TEXT,
                font=self.fonts.heading,
            )
            self.canvas.create_text(
                width // 2,
                height // 2 + 16,
                text="File › Open image, or Ctrl+O",
                fill=theme.CANVAS_HINT,
                font=self.fonts.hint,
            )
            self._report("")
            return

        placement = fit_image(self._source.size, (width, height))
        self._placement = placement
        self._photo = ImageTk.PhotoImage(self._preview(placement))
        self.canvas.create_image(placement.left, placement.top, image=self._photo, anchor="nw")
        self.canvas.configure(cursor="crosshair")
        self._draw_rulers(placement)
        self._report("")

    def _preview(self, placement: Placement) -> Image.Image:
        """The scaled copy of the source, reused while the size is unchanged."""
        source = self._source
        assert source is not None
        size = (placement.shown_width, placement.shown_height)
        for image, cached_size, preview in self._previews:
            if image is source and cached_size == size:
                return preview
        if size == source.size:
            preview = source
        else:
            preview = source.resize(size, Image.Resampling.LANCZOS, reducing_gap=2.0)
        self._previews = [(source, size, preview), *self._previews][:PREVIEW_CACHE_SIZE]
        return preview

    def _draw_rulers(self, placement: Placement) -> None:
        step = ruler_step(placement.scale)
        minor = minor_step(step)
        draw_minor = minor * placement.scale >= 5
        side = int(self.left_ruler.cget("width"))
        tick = {"fill": theme.MUTED, "width": 1}
        label = {"fill": theme.MUTED, "font": self.fonts.mono_small, "anchor": "nw"}

        top, left = self.top_ruler, self.left_ruler
        top.create_rectangle(
            placement.left, 0, placement.left + placement.shown_width, RULER_SIZE,
            fill=theme.RULER_SPAN, width=0,
        )
        left.create_rectangle(
            0, placement.top, side, placement.top + placement.shown_height,
            fill=theme.RULER_SPAN, width=0,
        )

        for value in range(0, placement.image_width + 1, minor if draw_minor else step):
            x = round(placement.canvas_x(value))
            numbered = value % step == 0
            top.create_line(x, RULER_SIZE - (9 if numbered else 4), x, RULER_SIZE, **tick)
            if numbered:
                top.create_text(x + 3, 3, text=str(value), **label)

        for value in range(0, placement.image_height + 1, minor if draw_minor else step):
            y = round(placement.canvas_y(value))
            numbered = value % step == 0
            left.create_line(side - (9 if numbered else 4), y, side, y, **tick)
            if numbered:
                left.create_text(3, y + 2, text=str(value), **label)

        cursor = {"fill": theme.ACCENT, "width": 2, "state": "hidden", "tags": ("cursor",)}
        top.create_line(0, 0, 0, RULER_SIZE, **cursor)
        left.create_line(0, 0, side, 0, **cursor)

    def _on_motion(self, event: tk.Event) -> None:
        placement, probe = self._placement, self._probe
        if placement is None or probe is None:
            return
        pixel = placement.to_image(event.x, event.y)
        if pixel is None:
            self._clear_probe()
            return
        x, y = pixel
        cx = placement.canvas_x(x + 0.5)
        cy = placement.canvas_y(y + 0.5)
        side = int(self.left_ruler.cget("width"))
        self.top_ruler.coords("cursor", cx, 0, cx, RULER_SIZE)
        self.left_ruler.coords("cursor", 0, cy, side, cy)
        self.top_ruler.itemconfigure("cursor", state="normal")
        self.left_ruler.itemconfigure("cursor", state="normal")
        value = format_pixel(probe.getbands(), probe.getpixel((x, y)))
        self._report(f"x {x}  y {y}    {value}")

    def _clear_probe(self) -> None:
        self.top_ruler.itemconfigure("cursor", state="hidden")
        self.left_ruler.itemconfigure("cursor", state="hidden")
        self._report("")

    def _report(self, text: str) -> None:
        if self.on_probe is not None:
            self.on_probe(text)
