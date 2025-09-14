"""
Video-Context Image Captioning with Liquid AI

A two-step project that leverages Liquid AI's LFM2-VL models to generate 
personalized image captions using video context.
"""

__version__ = "1.0.0"
__author__ = "Video Context Captioning Team"

from .pipeline import VideoContextCaptionPipeline
from .video_processor import VideoProcessor
from .context_extractor import ContextExtractor
from .image_captioner import ImageCaptioner
from .liquid_integration import LiquidAIIntegration

__all__ = [
    "VideoContextCaptionPipeline",
    "VideoProcessor", 
    "ContextExtractor",
    "ImageCaptioner",
    "LiquidAIIntegration"
]
