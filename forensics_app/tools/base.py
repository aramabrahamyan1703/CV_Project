"""Contract implemented by every feature shown in the sidebar."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import tkinter as tk
from typing import Any

from PIL import Image

from forensics_app.core import ImageDocument


@dataclass(frozen=True)
class ToolResult:
    """A tool may produce an image, textual measurements, or both."""

    message: str
    image: Image.Image | None = None
    details: dict[str, Any] = field(default_factory=dict)


class ForensicsTool(ABC):
    """Base class for a functionality/module in the application."""

    tool_id = "tool"
    title = "Unnamed tool"
    category = "Other"
    description = ""
    requires_image = True

    @abstractmethod
    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        """Run the feature; return ``None`` when the user cancels."""
