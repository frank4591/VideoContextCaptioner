#!/usr/bin/env python3
"""
Test script for the fixed LFM2-VL model integration.
"""

import sys
import os
import logging
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

# Add src to path
sys.path.append('src')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_image():
    """Create a proper test image."""
    # Create a simple test image with PIL
    img = Image.new('RGB', (224, 224), color='red')
    
    # Add some simple shapes
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    
    # Draw a blue rectangle
    draw.rectangle([50, 50, 150, 150], fill='blue')
    
    # Draw a green circle
    draw.ellipse([100, 100, 200, 200], fill='green')
    
    # Save the image
    examples_dir = Path('examples')
    examples_dir.mkdir(exist_ok=True)
    img.save('examples/test_image.jpg')
    
    return 'examples/test_image.jpg'

def test_simple_caption():
    """Test simple image captioning."""
    logger.info("Testing simple image captioning...")
    
    # Create test image
    image_path = create_test_image()
    
    # Test Liquid AI integration directly
    from src.liquid_integration import LiquidAIIntegration
    
    integration = LiquidAIIntegration(
        model_path='/home/frank/BrandInfluencerDatasetTraining/LFM2-VL-450M',
        device='cuda'
    )
    
    # Test caption generation
    result = integration.generate_caption(
        image=image_path,
        text_prompt="Describe this image in detail",
        return_features=True
    )
    
    logger.info(f"Generated caption: {result['caption']}")
    logger.info(f"Confidence: {result['confidence']:.2f}")
    
    return result

def test_pipeline():
    """Test the complete pipeline."""
    logger.info("Testing complete pipeline...")
    
    from src.pipeline import VideoContextCaptionPipeline
    
    # Create test files
    image_path = create_test_image()
    
    # Create a simple test video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('examples/test_video.mp4', fourcc, 30.0, (224, 224))
    
    for i in range(30):  # 1 second at 30 fps
        # Create frames with different colors
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        frame[:, :, i % 3] = 255  # Different color channel each frame
        out.write(frame)
    
    out.release()
    
    # Initialize pipeline
    pipeline = VideoContextCaptionPipeline(
        model_path='/home/frank/BrandInfluencerDatasetTraining/LFM2-VL-450M',
        device='cuda',
        max_frames=3  # Small number for testing
    )
    
    # Test image captioning without context
    logger.info("Testing image captioning without context...")
    result_no_context = pipeline.image_captioner.generate_caption_without_context(
        image_path=image_path
    )
    
    logger.info(f"Caption without context: {result_no_context['caption']}")
    logger.info(f"Confidence: {result_no_context['confidence']:.2f}")
    
    # Test image captioning with video context
    logger.info("Testing image captioning with video context...")
    result_with_context = pipeline.generate_caption(
        image_path=image_path,
        video_path='examples/test_video.mp4',
        context_weight=0.7
    )
    
    logger.info(f"Caption with context: {result_with_context['caption']}")
    logger.info(f"Video context: {result_with_context['video_context']['context_text']}")
    logger.info(f"Confidence: {result_with_context['confidence']:.2f}")
    logger.info(f"Context relevance: {result_with_context['context_relevance']:.2f}")
    
    return result_with_context

def main():
    """Run all tests."""
    logger.info("🧪 Testing Fixed LFM2-VL Integration")
    logger.info("=" * 50)
    
    try:
        # Test 1: Simple caption generation
        logger.info("\n1. Testing simple caption generation...")
        result1 = test_simple_caption()
        
        if result1['caption'] and len(result1['caption'].strip()) > 10:
            logger.info("✅ Simple caption generation working!")
        else:
            logger.warning("⚠️ Simple caption generation needs improvement")
        
        # Test 2: Complete pipeline
        logger.info("\n2. Testing complete pipeline...")
        result2 = test_pipeline()
        
        if result2['caption'] and len(result2['caption'].strip()) > 10:
            logger.info("✅ Complete pipeline working!")
        else:
            logger.warning("⚠️ Complete pipeline needs improvement")
        
        logger.info("\n🎉 Testing completed!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
