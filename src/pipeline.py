"""
Main pipeline for video-context image captioning.

This module orchestrates the two-step process of extracting video context
and generating image captions using Liquid AI's LFM2-VL models.
"""

import torch
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging
import time

from .video_processor import VideoProcessor
from .context_extractor import ContextExtractor
from .image_captioner import ImageCaptioner
from .liquid_integration import LiquidAIIntegration

class VideoContextCaptionPipeline:
    """
    Main pipeline for generating image captions with video context.
    
    This pipeline leverages Liquid AI's LFM2-VL models and their continuous-time
    processing capabilities to extract context from videos and apply it to image captioning.
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        frame_extraction_strategy: str = "key_frames",
        max_frames: int = 10,
        context_aggregation: str = "weighted_average",
        context_weight: float = 0.7
    ):
        """
        Initialize the video-context captioning pipeline.
        
        Args:
            model_path: Path to the LFM2-VL model
            device: Device to run inference on
            frame_extraction_strategy: Strategy for extracting frames ("key_frames", "uniform", "adaptive")
            max_frames: Maximum number of frames to extract
            context_aggregation: Method for aggregating frame contexts
            context_weight: Weight of video context in final caption (0.0-1.0)
        """
        self.device = device
        self.context_weight = context_weight
        self.max_frames = max_frames
        
        # Initialize components
        self.video_processor = VideoProcessor(
            strategy=frame_extraction_strategy,
            max_frames=max_frames
        )
        
        self.context_extractor = ContextExtractor(
            aggregation_method=context_aggregation
        )
        
        self.liquid_integration = LiquidAIIntegration(
            model_path=model_path,
            device=device
        )
        
        self.image_captioner = ImageCaptioner(
            liquid_integration=self.liquid_integration,
            context_weight=context_weight
        )
        
        logging.info(f"Pipeline initialized with {frame_extraction_strategy} frame extraction")
    
    def generate_caption(
        self,
        image_path: str,
        video_path: str,
        context_weight: Optional[float] = None,
        max_context_length: int = 512
    ) -> Dict[str, any]:
        """
        Generate image caption using video context.
        
        Args:
            image_path: Path to the input image
            video_path: Path to the context video
            context_weight: Override default context weight
            max_context_length: Maximum length of context text
            
        Returns:
            Dictionary containing caption and metadata
        """
        start_time = time.time()
        
        try:
            # Step 1: Extract video context
            logging.info("Step 1: Extracting video context...")
            video_context = self._extract_video_context(
                video_path, 
                max_context_length
            )
            
            # Step 2: Generate image caption with context
            logging.info("Step 2: Generating image caption with context...")
            caption_result = self.image_captioner.generate_caption(
                image_path=image_path,
                video_context=video_context,
                context_weight=context_weight or self.context_weight
            )
            
            total_time = time.time() - start_time
            
            return {
                "caption": caption_result["caption"],
                "video_context": video_context,
                "confidence": caption_result["confidence"],
                "processing_time": total_time,
                "frames_processed": video_context["frames_processed"],
                "context_relevance": caption_result["context_relevance"]
            }
            
        except Exception as e:
            logging.error(f"Error in caption generation: {str(e)}")
            raise
    
    def _extract_video_context(
        self, 
        video_path: str, 
        max_context_length: int
    ) -> Dict[str, any]:
        """
        Extract context from video using Liquid AI's continuous-time processing.
        
        This leverages the LNN architecture's ability to process sequential data
        and maintain adaptive state across video frames.
        """
        # Extract frames from video
        frames = self.video_processor.extract_frames(video_path)
        
        # Process frames through LFM2-VL to get preliminary captions
        frame_captions = []
        frame_features = []
        
        for i, frame in enumerate(frames):
            logging.info(f"Processing frame {i+1}/{len(frames)}")
            
            # Use LNN's continuous-time processing for each frame
            caption_result = self.liquid_integration.generate_caption(
                image=frame,
                return_features=True
            )
            
            frame_captions.append(caption_result["caption"])
            frame_features.append(caption_result["features"])
        
        # Aggregate context using LNN's state space capabilities
        aggregated_context = self.context_extractor.aggregate_context(
            frame_captions=frame_captions,
            frame_features=frame_features,
            max_length=max_context_length
        )
        
        return {
            "context_text": aggregated_context["text"],
            "context_features": aggregated_context["features"],
            "frames_processed": len(frames),
            "frame_captions": frame_captions,
            "temporal_consistency": aggregated_context["temporal_consistency"]
        }
    
    def batch_process(
        self,
        image_video_pairs: List[Tuple[str, str]],
        output_file: Optional[str] = None
    ) -> List[Dict[str, any]]:
        """
        Process multiple image-video pairs in batch.
        
        Args:
            image_video_pairs: List of (image_path, video_path) tuples
            output_file: Optional file to save results
            
        Returns:
            List of caption results
        """
        results = []
        
        for i, (image_path, video_path) in enumerate(image_video_pairs):
            logging.info(f"Processing pair {i+1}/{len(image_video_pairs)}")
            
            try:
                result = self.generate_caption(image_path, video_path)
                result["image_path"] = image_path
                result["video_path"] = video_path
                results.append(result)
                
            except Exception as e:
                logging.error(f"Error processing pair {i+1}: {str(e)}")
                results.append({
                    "image_path": image_path,
                    "video_path": video_path,
                    "error": str(e)
                })
        
        if output_file:
            self._save_results(results, output_file)
        
        return results
    
    def _save_results(self, results: List[Dict], output_file: str):
        """Save results to file."""
        import json
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logging.info(f"Results saved to {output_file}")
    
    def get_pipeline_info(self) -> Dict[str, any]:
        """Get information about the pipeline configuration."""
        return {
            "device": self.device,
            "max_frames": self.max_frames,
            "context_weight": self.context_weight,
            "frame_extraction_strategy": self.video_processor.strategy,
            "context_aggregation": self.context_extractor.aggregation_method,
            "model_path": self.liquid_integration.model_path
        }
