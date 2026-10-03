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

    
    def is_available(self, document: ImageDocument) -> bool:
        """Whether this tool may run against ``document`` right now.

        The sidebar greys out a tool whose answer is ``False``. Override it when
        a feature is meaningless on its own output -- see
        :class:`~forensics_app.tools.channel_split.ChannelSplitTool`. Deliberately
        document-only (no Tkinter) so the rule stays testable without a display.
        """
        return True

    def unavailable_message(self, document: ImageDocument) -> str:
        """Explain to the user why :meth:`is_available` said no."""
        return f"{self.title} cannot run on the current image."

    @abstractmethod
    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        """Run the feature; return ``None`` when the user cancels."""
