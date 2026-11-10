"""Ordered collection of tools available to the UI."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable

from forensics_app.core import ImageDocument
from .base import ForensicsTool


class ToolRegistry:
    def __init__(self, tools: Iterable[ForensicsTool] = ()) -> None:
        self._tools: OrderedDict[str, ForensicsTool] = OrderedDict()
        for tool in tools:
            self.register(tool)

    def register(self, tool: ForensicsTool) -> None:
        if tool.tool_id in self._tools:
            raise ValueError(f"Duplicate tool id: {tool.tool_id}")
        self._tools[tool.tool_id] = tool

    def all(self) -> tuple[ForensicsTool, ...]:
        return tuple(self._tools.values())

    def categories(self) -> tuple[tuple[str, tuple[ForensicsTool, ...]], ...]:
        grouped: OrderedDict[str, list[ForensicsTool]] = OrderedDict()
        for tool in self._tools.values():
            grouped.setdefault(tool.category, []).append(tool)
        return tuple((name, tuple(tools)) for name, tools in grouped.items())

    def get(self, tool_id: str | None) -> ForensicsTool | None:
        return self._tools.get(tool_id) if tool_id is not None else None

    def figure_maker(self, document: ImageDocument) -> ForensicsTool | None:
        """The tool whose figure is the working image, or None for a real image."""
        maker = self.get(document.last_tool_id)
        return maker if maker is not None and maker.produces_figure else None

    def blocked_by_figure(self, tool: ForensicsTool, document: ImageDocument) -> str | None:
        """Why ``tool`` cannot run on the working figure, or None if it can."""
        maker = self.figure_maker(document)
        if maker is None or tool.accepts_figures or not tool.requires_image:
            return None
        return (
            f"The working image is a figure made by {maker.title}, not a photo, "
            f"so {tool.title} cannot use it.\n\n"
            "Undo, reset to the original, or open an image to continue."
        )

    def is_available(self, tool: ForensicsTool, document: ImageDocument) -> bool:
        """The tool's own rule, plus: no photo analysis on a figure."""
        return self.blocked_by_figure(tool, document) is None and tool.is_available(document)

    def unavailable_message(self, tool: ForensicsTool, document: ImageDocument) -> str:
        return self.blocked_by_figure(tool, document) or tool.unavailable_message(document)
