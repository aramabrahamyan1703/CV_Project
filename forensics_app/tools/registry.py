"""Ordered collection of tools available to the UI."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable

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
