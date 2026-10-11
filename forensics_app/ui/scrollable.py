"""A vertically scrolling frame for the tool sidebar."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class ScrollableFrame(ttk.Frame):
    """Put children in :attr:`body`; it scrolls when taller than the frame.

    The scrollbar only appears when there is something to scroll, and the
    mouse wheel or trackpad scrolls whenever the pointer is over the frame.
    """

    def __init__(self, parent: tk.Misc, *, background: str) -> None:
        super().__init__(parent)
        self._canvas = tk.Canvas(self, background=background, borderwidth=0, highlightthickness=0)
        self._scrollbar = ttk.Scrollbar(
            self, orient="vertical", command=self._canvas.yview, style="Slim.Vertical.TScrollbar"
        )
        self._canvas.configure(yscrollcommand=self._scrollbar.set)
        self.body = ttk.Frame(self._canvas)
        self._window = self._canvas.create_window(0, 0, window=self.body, anchor="nw")

        self._canvas.grid(row=0, column=0, sticky="nsew")
        self._scrollbar.grid(row=0, column=1, sticky="ns")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.body.bind("<Configure>", lambda _event: self._update())
        self._canvas.bind("<Configure>", self._on_canvas_resize)
        # bind_all so the wheel works over the buttons too; _on_wheel ignores
        # events that happen anywhere else in the window.
        self.bind_all("<MouseWheel>", self._on_wheel, add="+")
        self.bind_all("<Button-4>", self._on_wheel, add="+")
        self.bind_all("<Button-5>", self._on_wheel, add="+")

    def _on_canvas_resize(self, event: tk.Event) -> None:
        self._canvas.itemconfigure(self._window, width=event.width)
        self._update()

    def _update(self) -> None:
        self._canvas.configure(scrollregion=(0, 0, 0, self.body.winfo_reqheight()))
        if self.body.winfo_reqheight() > self._canvas.winfo_height():
            self._scrollbar.grid()
        else:
            self._scrollbar.grid_remove()
            self._canvas.yview_moveto(0)

    def _contains_pointer(self, event: tk.Event) -> bool:
        widget = self.winfo_containing(event.x_root, event.y_root)
        if widget is None:
            return False
        path, own = str(widget), str(self)
        return path == own or path.startswith(own + ".")

    def _on_wheel(self, event: tk.Event) -> None:
        if not self._contains_pointer(event) or self._canvas.yview() == (0.0, 1.0):
            return
        if event.num == 4:
            units = -1
        elif event.num == 5:
            units = 1
        elif self.tk.call("tk", "windowingsystem") == "win32":
            units = -int(event.delta / 120) or (-1 if event.delta > 0 else 1)
        else:
            units = -event.delta
        self._canvas.yview_scroll(units, "units")
