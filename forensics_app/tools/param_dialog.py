"""A single-window parameter form for tools, styled by the app theme.

Layout: a heading and one-line explanation, labelled sections of fields, and a
footer band (like the main window's status bar) whose readout updates as the
user types. While the values are invalid the footer shows why in place and
the action button is disabled, so there are no error pop-ups to dismiss.

Only ttk style names from ``forensics_app.ui.theme`` are used, never the
module itself, so tools keep depending on core alone; without the theme the
dialog falls back to plain ttk.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk
from typing import Generic, TypeVar

from PIL import Image, ImageTk

T = TypeVar("T")

INTRO_WRAP = 380
FOOTER_WRAP = 250

IMAGE_TYPES = [
    ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
    ("All files", "*.*"),
]


@dataclass(frozen=True)
class IntField:
    label: str
    default: int
    limit: int
    hint: str
    minimum: int = 1


def parse_int_fields(fields: tuple[IntField, ...], texts: tuple[str, ...]) -> tuple[int, ...]:
    """Whole numbers within each field's minimum and limit; the error names the field."""
    values = []
    for field, text in zip(fields, texts, strict=True):
        try:
            value = int(text.strip())
        except ValueError:
            raise ValueError(f"{field.label} must be a whole number.") from None
        if not field.minimum <= value <= field.limit:
            raise ValueError(
                f"{field.label} must be between {field.minimum} and {field.limit}."
            )
        values.append(value)
    return tuple(values)


def inline_message(error: Exception) -> str:
    """Fold a multi-paragraph error into one line for the footer."""
    return " ".join(str(error).split())


class ParamDialog(Generic[T]):
    """Build with :meth:`section` / :meth:`int_fields`, then call :meth:`show`."""

    def __init__(self, parent: tk.Misc | None, *, title: str, intro: str, action: str) -> None:
        self.parent = parent
        self.top = tk.Toplevel(parent)
        self.top.withdraw()  # shown once laid out and centred, so it never jumps
        self.top.title(title)
        if parent is not None:
            self.top.transient(parent)
        self.top.resizable(False, False)

        self._action = action
        self._validate: Callable[[], tuple[T, str]] | None = None
        self._value: T | None = None
        self._result: T | None = None
        self._focus: tk.Widget | None = None

        outer = ttk.Frame(self.top)
        outer.pack(fill="both", expand=True)
        self.body = ttk.Frame(outer, padding=(20, 18, 20, 18))
        self.body.pack(fill="both", expand=True)
        self.body.columnconfigure(0, weight=1)

        ttk.Label(self.body, text=title, style="Heading.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            self.body, text=intro, style="Hint.TLabel", wraplength=INTRO_WRAP, justify="left"
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))
        self._next_row = 2

        ttk.Frame(outer, style="Line.TFrame", height=1).pack(fill="x")
        footer = ttk.Frame(outer, style="Surface.TFrame", padding=(20, 10, 14, 10))
        footer.pack(fill="x")
        footer.columnconfigure(0, weight=1)
        self._status = ttk.Label(
            footer, style="Readout.TLabel", wraplength=FOOTER_WRAP, justify="left"
        )
        self._status.grid(row=0, column=0, sticky="w", padx=(0, 16))
        ttk.Button(footer, text="Cancel", command=self._cancel).grid(row=0, column=1)
        self._apply = ttk.Button(
            footer, text=action, style="Primary.TButton", command=self._accept
        )
        self._apply.grid(row=0, column=2, padx=(8, 0))

        self.top.bind("<Return>", lambda _event: self._accept())
        self.top.bind("<KP_Enter>", lambda _event: self._accept())
        self.top.bind("<Escape>", lambda _event: self._cancel())
        self.top.protocol("WM_DELETE_WINDOW", self._cancel)

    def section(self, title: str) -> ttk.Frame:
        """A labelled group; returns the frame to grid its contents into."""
        ttk.Label(self.body, text=title, style="Section.TLabel").grid(
            row=self._next_row, column=0, sticky="w", pady=(18, 6)
        )
        frame = ttk.Frame(self.body)
        frame.grid(row=self._next_row + 1, column=0, sticky="we")
        self._next_row += 2
        return frame

    def int_fields(self, frame: ttk.Frame, fields: tuple[IntField, ...]) -> list[tk.StringVar]:
        """One row per field: label, spin box, hint. Edits refresh the footer."""
        variables = []
        start = frame.grid_size()[1]  # below anything already in the frame
        for offset, field in enumerate(fields):
            row = start + offset
            pady = (0, 0) if row == 0 else (8, 0)
            ttk.Label(frame, text=field.label).grid(row=row, column=0, sticky="w", pady=pady)
            variable = tk.StringVar(value=str(field.default))
            spinbox = ttk.Spinbox(
                frame,
                textvariable=variable,
                from_=field.minimum,
                to=field.limit,
                increment=1,
                width=5,
                justify="right",
            )
            spinbox.grid(row=row, column=1, sticky="w", padx=(14, 0), pady=pady)
            ttk.Label(frame, text=field.hint, style="Hint.TLabel").grid(
                row=row, column=2, sticky="w", padx=(10, 0), pady=pady
            )
            variable.trace_add("write", lambda *_args: self.refresh())
            if self._focus is None:
                self._focus = spinbox
            variables.append(variable)
        return variables

    def checkbox(self, frame: ttk.Frame, text: str, *, default: bool, hint: str) -> tk.BooleanVar:
        """A checkbox row below the frame's existing rows. Toggling refreshes the footer."""
        row = frame.grid_size()[1]
        pady = (0, 0) if row == 0 else (8, 0)
        variable = tk.BooleanVar(value=default)
        ttk.Checkbutton(frame, text=text, variable=variable).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=pady
        )
        ttk.Label(frame, text=hint, style="Hint.TLabel").grid(
            row=row, column=2, sticky="w", padx=(10, 0), pady=pady
        )
        variable.trace_add("write", lambda *_args: self.refresh())
        return variable

    def refresh(self) -> None:
        """Re-run validation: show the readout and enable the action, or the reason not to."""
        if self._validate is None:
            return
        try:
            self._value, readout = self._validate()
        except ValueError as error:
            self._value = None
            self._status.configure(text=inline_message(error), style="Error.TLabel")
            self._apply.state(["disabled"])
        else:
            self._status.configure(text=readout, style="Readout.TLabel")
            self._apply.state(["!disabled"])

    def show(self, validate: Callable[[], tuple[T, str]]) -> T | None:
        """Run modally; ``validate`` returns ``(value, readout)`` or raises ``ValueError``."""
        self._validate = validate
        self.refresh()
        self._centre()
        self.top.deiconify()
        self.top.grab_set()
        if self._focus is not None:
            self._focus.focus_set()
            self._focus.selection_range(0, "end")
        self.top.wait_window()
        return self._result

    def _centre(self) -> None:
        self.top.update_idletasks()
        width, height = self.top.winfo_reqwidth(), self.top.winfo_reqheight()
        if self.parent is not None and self.parent.winfo_viewable():
            x = self.parent.winfo_rootx() + (self.parent.winfo_width() - width) // 2
            y = self.parent.winfo_rooty() + (self.parent.winfo_height() - height) // 3
        else:
            x = (self.top.winfo_screenwidth() - width) // 2
            y = (self.top.winfo_screenheight() - height) // 3
        self.top.geometry(f"+{max(x, 0)}+{max(y, 0)}")

    def _accept(self) -> None:
        if self._value is None:
            return  # the footer already says what to fix
        self._result = self._value
        self.top.destroy()

    def _cancel(self) -> None:
        self._result = None
        self.top.destroy()


class ImageChooser:
    """A thumbnail, file name and size, and a "Choose image…" button in one section."""

    THUMB_W = 76
    THUMB_H = 96

    def __init__(
        self,
        dialog: ParamDialog,
        frame: ttk.Frame,
        *,
        prompt: str,
        empty_hint: str,
        initial: Path | None = None,
    ) -> None:
        self._dialog = dialog
        self._prompt = prompt
        self._empty_hint = empty_hint
        self.path: Path | None = None
        self.image: Image.Image | None = None
        self._error: str | None = None
        self._thumb: ImageTk.PhotoImage | None = None  # kept alive for Tk

        box = ttk.Frame(frame, style="Preview.TFrame", width=self.THUMB_W, height=self.THUMB_H)
        box.grid(row=0, column=0, rowspan=3, sticky="nw")
        box.grid_propagate(False)
        box.columnconfigure(0, weight=1)
        box.rowconfigure(0, weight=1)
        self._preview = ttk.Label(box, style="Preview.TLabel", text="No\nimage")
        self._preview.grid(row=0, column=0)

        self._name = ttk.Label(frame, style="Strong.TLabel", text="No image chosen")
        self._name.grid(row=0, column=1, sticky="sw", padx=(14, 0))
        self._info = ttk.Label(frame, style="Hint.TLabel", text=empty_hint, justify="left")
        self._info.grid(row=1, column=1, sticky="nw", padx=(14, 0), pady=(2, 0))
        frame.rowconfigure(1, weight=1)
        ttk.Button(frame, text="Choose image…", command=self._choose).grid(
            row=2, column=1, sticky="sw", padx=(14, 0)
        )
        if initial is not None and initial.exists():
            self.load(initial)

    def load(self, path: Path) -> None:
        """Open ``path``, show its thumbnail and size, and refresh the dialog."""
        self.path = path
        self._name.configure(text=path.name)
        try:
            with Image.open(path) as opened:
                image = opened.convert("RGB")
        except OSError:
            self.image, self._error = None, f"Cannot open {path.name} as an image."
            self._info.configure(text="Not a readable image file.")
            self._preview.configure(image="", text="No\nimage")
        else:
            self.image, self._error = image, None
            shown = image.copy()
            shown.thumbnail((self.THUMB_W - 8, self.THUMB_H - 8), Image.Resampling.LANCZOS)
            self._thumb = ImageTk.PhotoImage(shown, master=self._dialog.top)
            self._preview.configure(image=self._thumb, text="")
            self._info.configure(text=f"{image.width} × {image.height} px")
        self._dialog.refresh()

    def require(self) -> tuple[Path, Image.Image]:
        """The chosen path and image, or ``ValueError`` saying what is missing."""
        if self._error:
            raise ValueError(self._error)
        if self.path is None or self.image is None:
            raise ValueError(self._prompt)
        return self.path, self.image

    def _choose(self) -> None:
        chosen = filedialog.askopenfilename(
            parent=self._dialog.top,
            title=self._prompt,
            filetypes=IMAGE_TYPES,
            initialdir=str(self.path.parent) if self.path else None,
        )
        if chosen:
            self.load(Path(chosen))


class FolderChooser:
    """A folder name, a line describing its contents, and a "Choose folder…" button."""

    def __init__(
        self,
        dialog: ParamDialog,
        frame: ttk.Frame,
        *,
        prompt: str,
        describe: Callable[[Path], str],
        initial: Path | None = None,
    ) -> None:
        self._dialog = dialog
        self._prompt = prompt
        self._describe = describe
        self.folder: Path | None = None

        self._name = ttk.Label(frame, style="Strong.TLabel", text="No folder chosen")
        self._name.grid(row=0, column=0, sticky="w")
        self._info = ttk.Label(frame, style="Hint.TLabel", text="", justify="left")
        self._info.grid(row=1, column=0, sticky="w", pady=(2, 0))
        ttk.Button(frame, text="Choose folder…", command=self._choose).grid(
            row=0, column=1, rowspan=2, sticky="e", padx=(14, 0)
        )
        frame.columnconfigure(0, weight=1)
        if initial is not None and initial.is_dir():
            self.load(initial)

    def load(self, folder: Path) -> None:
        self.folder = folder
        self._name.configure(text=folder.name or str(folder))
        self._info.configure(text=self._describe(folder))
        self._dialog.refresh()

    def require(self) -> Path:
        if self.folder is None:
            raise ValueError(self._prompt)
        return self.folder

    def _choose(self) -> None:
        chosen = filedialog.askdirectory(
            parent=self._dialog.top,
            title=self._prompt,
            initialdir=str(self.folder) if self.folder else None,
            mustexist=True,
        )
        if chosen:
            self.load(Path(chosen))
