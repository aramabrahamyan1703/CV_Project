"""Register course functionality here so it appears in the sidebar."""

from .channel_split import ChannelSplitTool
from .channel_swap import ChannelSwapTool
from .contrast_stretch import ContrastStretchTool
from .edge_visualization import EdgeVisualizationTool
from .grayscale import GrayscaleTool
from .histogram import HistogramTool
from .image_info import ImageInfoTool
from .masking import MaskingTool
from .registry import ToolRegistry
from .sharpen import SharpenTool
from .convolution import ConvolutionTool
from .skimage_filters import (
    GaussianFilterTool,
    MedianFilterTool,
    SobelFilterTool,
    UnsharpMaskTool,
)
from .canny import CannyEdgeTool
from .ncc import NCCTool
from .template_matching import TemplateMatchingTool
from .hog import HOGTool
from .hog_detection import HOGPersonDetectionTool
from .corners import CornerDetectionTool
from .descriptors import DescriptorTool


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
            EdgeVisualizationTool(),
            SharpenTool(),
            ConvolutionTool(),
            GaussianFilterTool(),
            MedianFilterTool(),
            SobelFilterTool(),
            UnsharpMaskTool(),
            CannyEdgeTool(),
            NCCTool(),
            TemplateMatchingTool(),
            HOGTool(),
            HOGPersonDetectionTool(),
            CornerDetectionTool(),
            DescriptorTool(),
        ]
    )


__all__ = ["ToolRegistry", "build_tool_registry"]
