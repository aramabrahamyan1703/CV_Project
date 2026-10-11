"""The application's main window and event coordination."""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import UnidentifiedImageError

from forensics_app.core import ImageDocument
from forensics_app.tools.base import ForensicsTool
from forensics_app.tools.registry import ToolRegistry
from . import theme
from .image_view import ImageView
from .scrollable import ScrollableFrame


OPEN_TYPES = [
    ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
    ("All files", "*.*"),
]
SAVE_TYPES = [("PNG image", "*.png"), ("JPEG image", "*.jpg"), ("TIFF image", "*.tiff")]

SIDEBAR_WIDTH = 236
INSPECTOR_WIDTH = 300


def step_labels(
    history: tuple[str | None, ...],
    redo_history: tuple[str | None, ...],
    titles: dict[str, str],
) -> list[tuple[str, str]]:
    """Rows for the Steps list as ``(text, state)``, state one of past/current/undone."""
    rows: list[tuple[str, str]] = []
    steps = [*history, *redo_history]
    for index, tool_id in enumerate(steps):
        if tool_id is None:
            name = "Original" if index == 0 else "Reset to original"
        else:
            name = titles.get(tool_id, tool_id)
        if index < len(history) - 1:
            state = "past"
        elif index == len(history) - 1:
            state = "current"
        else:
            state = "undone"
        rows.append((f"{index + 1}   {name}", state))
    return rows


class MainWindow:
    def __init__(self, root: tk.Tk, registry: ToolRegistry) -> None:
        self.root = root
        self.registry = registry
        self.document = ImageDocument()
        self.status = tk.StringVar()
        self.readout = tk.StringVar()
        self._message = "Ready. Open an image to begin."
        self._comparing = False

        self._configure_window()
        self._build_menu()
        self._build_layout()
        self._bind_shortcuts()
        self._set_status(self._message)
        self._refresh()

    def _configure_window(self) -> None:
        self.root.title("ForensicsApp")
        self.root.geometry("1220x760")
        self.root.minsize(860, 540)
        self.fonts = theme.apply_theme(self.root)

    def _build_menu(self) -> None:
        menu = tk.Menu(self.root)
        file_menu = tk.Menu(menu, tearoff=False)
        file_menu.add_command(label="Open image…", command=self.open_image, accelerator="Ctrl+O")
        file_menu.add_command(label="Save result as…", command=self.save_image, accelerator="Ctrl+Shift+S")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.destroy)
        menu.add_cascade(label="File", menu=file_menu)

        edit_menu = tk.Menu(menu, tearoff=False)
        edit_menu.add_command(label="Undo", command=self.undo, accelerator="Ctrl+Z")
        edit_menu.add_command(label="Redo", command=self.redo, accelerator="Ctrl+Y")
        edit_menu.add_command(label="Reset to original", command=self.reset)
        menu.add_cascade(label="Edit", menu=edit_menu)

        help_menu = tk.Menu(menu, tearoff=False)
        help_menu.add_command(label="About", command=self.show_about)
        menu.add_cascade(label="Help", menu=help_menu)
        self.root.configure(menu=menu)

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root)
        container.pack(fill="both", expand=True)

        self._build_toolbar(container)
        ttk.Frame(container, style="Line.TFrame", height=1).pack(fill="x")

        status_bar = ttk.Frame(container, style="Surface.TFrame", padding=(14, 6))
        status_bar.pack(side="bottom", fill="x")
        ttk.Label(status_bar, textvariable=self.status, style="Surface.TLabel", anchor="w").pack(
            side="left", fill="x", expand=True
        )
        ttk.Label(status_bar, textvariable=self.readout, style="Readout.TLabel", anchor="e").pack(
            side="right"
        )
        ttk.Frame(container, style="Line.TFrame", height=1).pack(side="bottom", fill="x")

        body = ttk.Panedwindow(container, orient="horizontal")
        body.pack(fill="both", expand=True)

        body.add(self._build_sidebar(body), weight=0)

        self.image_view = ImageView(body, self.fonts, on_probe=self._show_probe)
        body.add(self.image_view, weight=1)

        body.add(self._build_inspector(body), weight=0)

    def _build_toolbar(self, container: ttk.Frame) -> None:
        toolbar = ttk.Frame(container, style="Surface.TFrame", padding=(10, 8))
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="Open image", style="Primary.TButton", command=self.open_image).pack(
            side="left"
        )
        self.save_button = ttk.Button(
            toolbar, text="Save result", style="Toolbar.TButton", command=self.save_image
        )
        self.save_button.pack(side="left", padx=(6, 0))
        ttk.Frame(toolbar, style="Line.TFrame", width=1).pack(side="left", fill="y", padx=12, pady=4)
        self.undo_button = ttk.Button(toolbar, text="Undo", style="Toolbar.TButton", command=self.undo)
        self.undo_button.pack(side="left")
        self.redo_button = ttk.Button(toolbar, text="Redo", style="Toolbar.TButton", command=self.redo)
        self.redo_button.pack(side="left", padx=(2, 0))
        self.reset_button = ttk.Button(toolbar, text="Reset", style="Toolbar.TButton", command=self.reset)
        self.reset_button.pack(side="left", padx=(2, 0))

        # Press and hold: a before/after flick without touching the history.
        self.compare_button = ttk.Button(toolbar, text="Hold to compare", style="Toolbar.TButton")
        self.compare_button.pack(side="right")
        self.compare_button.bind("<ButtonPress-1>", lambda _event: self._start_compare())
        self.compare_button.bind("<ButtonRelease-1>", lambda _event: self._end_compare())

    def _build_sidebar(self, parent: ttk.Panedwindow) -> ttk.Frame:
        sidebar = ttk.Frame(parent, width=SIDEBAR_WIDTH)
        sidebar.pack_propagate(False)
        scroller = ScrollableFrame(sidebar, background=theme.PANEL)
        scroller.pack(fill="both", expand=True)
        column = scroller.body

        ttk.Label(column, text="Tools", style="Heading.TLabel").pack(anchor="w", padx=16, pady=(14, 2))
        self.tool_buttons: dict[str, ttk.Button] = {}
        for category, tools in self.registry.categories():
            ttk.Label(column, text=category, style="Section.TLabel").pack(
                anchor="w", padx=16, pady=(12, 3)
            )
            for tool in tools:
                button = ttk.Button(
                    column,
                    text=tool.title,
                    style="Tool.TButton",
                    command=lambda selected=tool: self.run_tool(selected),
                )
                self.tool_buttons[tool.tool_id] = button
                button.pack(fill="x", padx=6, pady=1)
                button.bind("<ButtonRelease-1>", lambda _event, selected=tool: self._on_tool_click(selected))
                button.bind("<Enter>", lambda _event, selected=tool: self._describe_tool(selected))
                button.bind("<Leave>", lambda _event: self.status.set(self._message))
        ttk.Frame(column, height=12).pack()
        return sidebar

    def _build_inspector(self, parent: ttk.Panedwindow) -> ttk.Frame:
        inspector = ttk.Frame(parent, width=INSPECTOR_WIDTH, padding=(16, 14, 12, 12))
        inspector.pack_propagate(False)

        ttk.Label(inspector, text="Results", style="Heading.TLabel").pack(anchor="w", pady=(0, 6))
        self.results = ttk.Treeview(inspector, columns=("value",), show="tree", selectmode="browse")
        self.results.column("#0", width=120, minwidth=90, stretch=True)
        self.results.column("value", width=150, minwidth=100, stretch=True)
        self.results.pack(fill="both", expand=True)

        ttk.Frame(inspector, style="Line.TFrame", height=1).pack(fill="x", pady=(10, 12))
        ttk.Label(inspector, text="Steps", style="Heading.TLabel").pack(anchor="w", pady=(0, 6))
        self.steps = ttk.Treeview(inspector, show="tree", selectmode="none", height=7)
        self.steps.column("#0", width=260, stretch=True)
        self.steps.tag_configure("past", foreground=theme.MUTED)
        self.steps.tag_configure("current", foreground=theme.ACCENT, font=self.fonts.strong)
        self.steps.tag_configure("undone", foreground=theme.FAINT)
        self.steps.pack(fill="x")
        return inspector

    def _bind_shortcuts(self) -> None:
        self.root.bind_all("<Control-o>", lambda _event: self.open_image())
        self.root.bind_all("<Control-Shift-S>", lambda _event: self.save_image())
        self.root.bind_all("<Control-z>", lambda _event: self.undo())
        self.root.bind_all("<Control-y>", lambda _event: self.redo())

    def _set_status(self, message: str) -> None:
        """Show ``message`` and keep it, so hovering a tool only borrows the bar."""
        self._message = message
        self.status.set(message)

    def open_image(self) -> None:
        filename = filedialog.askopenfilename(title="Open evidence image", filetypes=OPEN_TYPES)
        if not filename:
            return
        try:
            self.document.load(filename)
        except (OSError, UnidentifiedImageError) as error:
            messagebox.showerror("Could not open image", str(error), parent=self.root)
            return
        self._set_status(f"Opened {Path(filename).name}")
        self._show_default_details()
        self._refresh()

    def save_image(self) -> None:
        if not self._require_image():
            return
        source = self.document.path
        initial = f"{source.stem}_result.png" if source else "result.png"
        filename = filedialog.asksaveasfilename(
            title="Save processed image",
            defaultextension=".png",
            initialfile=initial,
            filetypes=SAVE_TYPES,
        )
        if not filename:
            return
        try:
            self.document.save(filename)
        except OSError as error:
            messagebox.showerror("Could not save image", str(error), parent=self.root)
            return
        self._set_status(f"Saved result as {Path(filename).name}")

    def _on_tool_click(self, tool: ForensicsTool) -> None:
        """Explain a greyed-out button, which never reaches :meth:`run_tool`."""
        if tool.is_available(self.document):
            return  # the button's own command runs the tool
        self._report_unavailable(tool)

    def _describe_tool(self, tool: ForensicsTool) -> None:
        if tool.is_available(self.document):
            self.status.set(tool.description)
        else:
            self.status.set(tool.unavailable_message(self.document).partition("\n")[0])

    def _report_unavailable(self, tool: ForensicsTool) -> None:
        message = tool.unavailable_message(self.document)
        messagebox.showinfo(tool.title, message, parent=self.root)
        self._set_status(message.partition("\n")[0])

    def run_tool(self, tool: ForensicsTool) -> None:
        if tool.requires_image and not self._require_image():
            return
        if not tool.is_available(self.document):
            self._report_unavailable(tool)
            return
        try:
            result = tool.run(self.root, self.document)
        except Exception as error:  # keep one student feature from crashing the shell
            messagebox.showerror(f"{tool.title} failed", str(error), parent=self.root)
            self._set_status(f"Error in {tool.title}.")
            return
        if result is None:
            self._set_status(f"Cancelled {tool.title}.")
            return
        if result.image is not None:
            self.document.apply(result.image, tool.tool_id)
        self._show_details(result.details)
        self._set_status(result.message)
        self._refresh()

    def undo(self) -> None:
        if self.document.undo():
            self._set_status("Undid the last image operation.")
            self._show_default_details()
            self._refresh()

    def redo(self) -> None:
        if self.document.redo():
            self._set_status("Redid the image operation.")
            self._show_default_details()
            self._refresh()

    def reset(self) -> None:
        if self.document.reset():
            self._set_status("Restored the original image.")
            self._show_default_details()
            self._refresh()

    def _start_compare(self) -> None:
        if not self.document.is_modified or self.document.original is None:
            return
        self._comparing = True
        self.image_view.show(self.document.original)
        self.status.set("Showing the original image. Release to return to the result.")

    def _end_compare(self) -> None:
        if not self._comparing:
            return
        self._comparing = False
        self.image_view.show(self.document.current)
        self.status.set(self._message)

    def _show_probe(self, text: str) -> None:
        """Pixel under the pointer, or the image size and zoom when off the image."""
        if text:
            self.readout.set(text)
            return
        zoom = self.image_view.zoom if hasattr(self, "image_view") else None
        shown = self.document.original if self._comparing else self.document.current
        if shown is None or zoom is None:
            self.readout.set("")
            return
        self.readout.set(f"{shown.width} × {shown.height} px   {zoom:.0%}")

    def _require_image(self) -> bool:
        if self.document.is_loaded:
            return True
        messagebox.showinfo("No image loaded", "Open an image first.", parent=self.root)
        return False

    def _refresh(self) -> None:
        self._comparing = False
        self.image_view.show(self.document.current)
        loaded = self.document.is_loaded
        self.save_button.configure(state="normal" if loaded else "disabled")
        self.undo_button.configure(state="normal" if self.document.can_undo else "disabled")
        self.redo_button.configure(state="normal" if self.document.can_redo else "disabled")
        self.reset_button.configure(state="normal" if self.document.is_modified else "disabled")
        self.compare_button.configure(state="normal" if self.document.is_modified else "disabled")
        for tool in self.registry.all():
            button = self.tool_buttons.get(tool.tool_id)
            if button is not None:
                button.configure(
                    state="normal" if tool.is_available(self.document) else "disabled",
                    style="Current.Tool.TButton"
                    if tool.tool_id == self.document.last_tool_id
                    else "Tool.TButton",
                )
        self._show_steps()
        title = self.document.path.name if self.document.path else "No image"
        self.root.title(f"ForensicsApp — {title}")

    def _show_steps(self) -> None:
        self.steps.delete(*self.steps.get_children())
        titles = {tool.tool_id: tool.title for tool in self.registry.all()}
        current = None
        for text, state in step_labels(self.document.history, self.document.redo_history, titles):
            item = self.steps.insert("", "end", text=text, tags=(state,))
            if state == "current":
                current = item
        if current is not None:
            self.steps.see(current)

    def _show_default_details(self) -> None:
        image = self.document.current
        if image is None:
            self._show_details({})
            return
        self._show_details(
            {
                "File": self.document.path.name if self.document.path else "—",
                "Size": f"{image.width} × {image.height}",
                "Mode": image.mode,
            }
        )

    def _show_details(self, details: dict[str, object]) -> None:
        for item in self.results.get_children():
            self.results.delete(item)
        for name, value in details.items():
            self.results.insert("", "end", text=str(name), values=(str(value),))

    def show_about(self) -> None:
        messagebox.showinfo(
            "About ForensicsApp",
            "A modular image-forensics application for the Computer Vision course.\n\n"
            "Add each weekly feature as a tool in forensics_app/tools/.",
            parent=self.root,
        )
