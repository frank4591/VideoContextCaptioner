"""
Example usage of the video-context image captioning pipeline.

This script demonstrates how to use the VideoContextCaptionPipeline
to generate image captions with video context.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import VideoContextCaptionPipeline
from src.video_processor import VideoProcessor
from src.context_extractor import ContextExtractor
from src.image_captioner import ImageCaptioner
from src.liquid_integration import LiquidAIIntegration
import logging
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_files():
    """Create sample image and video files for testing."""
    import cv2
    import numpy as np
    
    # Create examples directory
    examples_dir = Path(__file__).parent
    examples_dir.mkdir(exist_ok=True)
    
    # Create a sample image
    sample_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    cv2.imwrite(str(examples_dir / "sample_image.jpg"), sample_image)
    
    # Create a sample video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(examples_dir / "sample_video.mp4"), fourcc, 30.0, (224, 224))
    
    for i in range(90):  # 3 seconds at 30 fps
        frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        out.write(frame)
    
    out.release()
    
    logger.info("Sample files created successfully")

def example_basic_usage():
    """Example 1: Basic usage with single image-video pair."""
    logger.info("=" * 50)
    logger.info("Example 1: Basic Usage")
    logger.info("=" * 50)
    
    try:
        # Initialize pipeline
        pipeline = VideoContextCaptionPipeline(
            model_path="./models/lfm2-vl-450m",  # Update with actual model path
            device="cuda",
            frame_extraction_strategy="key_frames",
            max_frames=8,
            context_aggregation="weighted_average",
            context_weight=0.7
        )
        
        # Generate caption with video context
        result = pipeline.generate_caption(
            image_path="examples/sample_image.jpg",
            video_path="examples/sample_video.mp4",
            context_weight=0.8
        )
        
        print(f"Generated Caption: {result['caption']}")
        print(f"Video Context: {result['video_context']['context_text']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Context Relevance: {result['context_relevance']:.2f}")
        print(f"Frames Processed: {result['frames_processed']}")
        print(f"Processing Time: {result['processing_time']:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error in basic usage example: {str(e)}")

def example_advanced_usage():
    """Example 2: Advanced usage with different strategies."""
    logger.info("=" * 50)
    logger.info("Example 2: Advanced Usage")
    logger.info("=" * 50)
    
    try:
        # Test different frame extraction strategies
        strategies = ["key_frames", "uniform", "adaptive"]
        
        for strategy in strategies:
            logger.info(f"Testing strategy: {strategy}")
            
            pipeline = VideoContextCaptionPipeline(
                model_path="./models/lfm2-vl-450m",
                device="cuda",
                frame_extraction_strategy=strategy,
                max_frames=5,
                context_aggregation="weighted_average"
            )
            
            result = pipeline.generate_caption(
                image_path="examples/sample_image.jpg",
                video_path="examples/sample_video.mp4"
            )
            
            print(f"Strategy: {strategy}")
            print(f"Caption: {result['caption']}")
            print(f"Frames Processed: {result['frames_processed']}")
            print("-" * 30)
        
    except Exception as e:
        logger.error(f"Error in advanced usage example: {str(e)}")

def example_batch_processing():
    """Example 3: Batch processing multiple image-video pairs."""
    logger.info("=" * 50)
    logger.info("Example 3: Batch Processing")
    logger.info("=" * 50)
    
    try:
        # Create multiple sample files
        create_sample_files()
        
        # Initialize pipeline
        pipeline = VideoContextCaptionPipeline(
            model_path="./models/lfm2-vl-450m",
            device="cuda",
            frame_extraction_strategy="key_frames",
            max_frames=6
        )
        
        # Prepare batch data
        image_video_pairs = [
            ("examples/sample_image.jpg", "examples/sample_video.mp4"),
            ("examples/sample_image.jpg", "examples/sample_video.mp4"),  # Same files for demo
            ("examples/sample_image.jpg", "examples/sample_video.mp4")
        ]
        
        # Process batch
        results = pipeline.batch_process(
            image_video_pairs=image_video_pairs,
            output_file="batch_results.json"
        )
        
        print(f"Processed {len(results)} image-video pairs")
        for i, result in enumerate(results):
            if "error" not in result:
                print(f"Pair {i+1}: {result['caption'][:100]}...")
                print(f"  Confidence: {result['confidence']:.2f}")
                print(f"  Context Relevance: {result['context_relevance']:.2f}")
            else:
                print(f"Pair {i+1}: Error - {result['error']}")
        
    except Exception as e:
        logger.error(f"Error in batch processing example: {str(e)}")

def example_context_comparison():
    """Example 4: Compare captions with and without context."""
    logger.info("=" * 50)
    logger.info("Example 4: Context Comparison")
    logger.info("=" * 50)
    
    try:
        # Initialize pipeline
        pipeline = VideoContextCaptionPipeline(
            model_path="./models/lfm2-vl-450m",
            device="cuda",
            frame_extraction_strategy="key_frames",
            max_frames=8
        )
        
        # Generate caption without context
        result_without_context = pipeline.image_captioner.generate_caption_without_context(
            image_path="examples/sample_image.jpg"
        )
        
        # Generate caption with context
        result_with_context = pipeline.generate_caption(
            image_path="examples/sample_image.jpg",
            video_path="examples/sample_video.mp4",
            context_weight=0.7
        )
        
        print("Caption without context:")
        print(f"  {result_without_context['caption']}")
        print(f"  Confidence: {result_without_context['confidence']:.2f}")
        print()
        
        print("Caption with context:")
        print(f"  {result_with_context['caption']}")
        print(f"  Confidence: {result_with_context['confidence']:.2f}")
        print(f"  Context Relevance: {result_with_context['context_relevance']:.2f}")
        print()
        
        print("Video Context:")
        print(f"  {result_with_context['video_context']['context_text']}")
        
    except Exception as e:
        logger.error(f"Error in context comparison example: {str(e)}")

def example_quality_analysis():
    """Example 5: Analyze caption quality."""
    logger.info("=" * 50)
    logger.info("Example 5: Quality Analysis")
    logger.info("=" * 50)
    
    try:
        # Initialize pipeline
        pipeline = VideoContextCaptionPipeline(
            model_path="./models/lfm2-vl-450m",
            device="cuda"
        )
        
        # Generate caption
        result = pipeline.generate_caption(
            image_path="examples/sample_image.jpg",
            video_path="examples/sample_video.mp4"
        )
        
        # Analyze quality
        quality_analysis = pipeline.image_captioner.analyze_caption_quality(
            result['caption'],
            result['video_context']
        )
        
        print("Caption Quality Analysis:")
        print(f"  Word Count: {quality_analysis['word_count']}")
        print(f"  Unique Words: {quality_analysis['unique_words']}")
        print(f"  Diversity Ratio: {quality_analysis['diversity_ratio']:.2f}")
        print(f"  Context Relevance: {quality_analysis['context_relevance']:.2f}")
        print(f"  Has Objects: {quality_analysis['has_objects']}")
        print(f"  Has Actions: {quality_analysis['has_actions']}")
        print(f"  Has Scenes: {quality_analysis['has_scenes']}")
        print(f"  Overall Quality Score: {quality_analysis['quality_score']:.2f}")
        
    except Exception as e:
        logger.error(f"Error in quality analysis example: {str(e)}")

def example_pipeline_info():
    """Example 6: Get pipeline information."""
    logger.info("=" * 50)
    logger.info("Example 6: Pipeline Information")
    logger.info("=" * 50)
    
    try:
        # Initialize pipeline
        pipeline = VideoContextCaptionPipeline(
            model_path="./models/lfm2-vl-450m",
            device="cuda",
            frame_extraction_strategy="adaptive",
            max_frames=10,
            context_aggregation="attention"
        )
        
        # Get pipeline info
        info = pipeline.get_pipeline_info()
        
        print("Pipeline Configuration:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Get model info
        model_info = pipeline.liquid_integration.get_model_info()
        
        print("\nModel Information:")
        for key, value in model_info.items():
            print(f"  {key}: {value}")
        
    except Exception as e:
        logger.error(f"Error in pipeline info example: {str(e)}")

def main():
    """Run all examples."""
    logger.info("Starting Video-Context Image Captioning Examples")
    logger.info("=" * 60)
    
    # Create sample files first
    create_sample_files()
    
    # Run examples
    try:
        example_basic_usage()
        example_advanced_usage()
        example_batch_processing()
        example_context_comparison()
        example_quality_analysis()
        example_pipeline_info()
        
        logger.info("=" * 60)
        logger.info("All examples completed successfully!")
        
    except Exception as e:
        logger.error(f"Error running examples: {str(e)}")

if __name__ == "__main__":
    main()
