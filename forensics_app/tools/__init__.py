"""Register course functionality here so it appears in the sidebar."""

from .grayscale import GrayscaleTool
from .image_info import ImageInfoTool
from .registry import ToolRegistry
from .channel_split import ChannelSplitTool
from .channel_swap import ChannelSwapTool
from .masking import MaskingTool
from .contrast_stretch import ContrastStretchTool
from .histogram import HistogramTool
from .convolution import ConvolutionTool
from .skimage_filters import (
    GaussianFilterTool,
    MedianFilterTool,
    SobelFilterTool,
    UnsharpMaskTool,
)
from .canny import CannyEdgeTool


def build_tool_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            ImageInfoTool(),
            GrayscaleTool(),
            ChannelSplitTool(),
            ChannelSwapTool(),
            MaskingTool(),
            HistogramTool(),
            ContrastStretchTool(),
            ConvolutionTool(),
            GaussianFilterTool(),
            MedianFilterTool(),
            SobelFilterTool(),
            UnsharpMaskTool(),
            CannyEdgeTool(),
        ]
    )


__all__ = ["ToolRegistry", "build_tool_registry"]
