#!/usr/bin/env python3
"""
Test script for the real LFM2-VL model integration.

This script tests the video-context image captioning pipeline
with the actual LFM2-VL-450M model.
"""

import sys
import os
import logging
import numpy as np
import cv2
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.pipeline import VideoContextCaptionPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_files():
    """Create test image and video files."""
    logger.info("Creating test files...")
    
    # Create examples directory
    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True)
    
    # Create a sample image
    sample_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    cv2.imwrite(str(examples_dir / "test_image.jpg"), sample_image)
    
    # Create a sample video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(examples_dir / "test_video.mp4"), fourcc, 30.0, (224, 224))
    
    for i in range(60):  # 2 seconds at 30 fps
        frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        out.write(frame)
    
    out.release()
    
    logger.info("Test files created successfully")

def test_model_loading():
    """Test loading the real LFM2-VL model."""
    logger.info("Testing model loading...")
    
    model_path = "/home/frank/BrandInfluencerDatasetTraining/LFM2-VL-450M"
    
    try:
        # Test Liquid AI integration directly
        from src.liquid_integration import LiquidAIIntegration
        
        integration = LiquidAIIntegration(
            model_path=model_path,
            device="cuda"
        )
        
        model_info = integration.get_model_info()
        logger.info(f"Model loaded successfully: {model_info}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def test_pipeline_initialization():
    """Test pipeline initialization with real model."""
    logger.info("Testing pipeline initialization...")
    
    try:
        pipeline = VideoContextCaptionPipeline(
            model_path="/home/frank/BrandInfluencerDatasetTraining/LFM2-VL-450M",
            device="cuda",
            max_frames=5
        )
        
        info = pipeline.get_pipeline_info()
        logger.info(f"Pipeline initialized successfully: {info}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error initializing pipeline: {str(e)}")
        return False

def test_caption_generation():
    """Test caption generation with real model."""
    logger.info("Testing caption generation...")
    
    try:
        # Create test files
        create_test_files()
        
        # Initialize pipeline
        pipeline = VideoContextCaptionPipeline(
            model_path="/home/frank/BrandInfluencerDatasetTraining/LFM2-VL-450M",
            device="cuda",
            max_frames=3  # Small number for testing
        )
        
        # Test image captioning without context
        logger.info("Testing image captioning without context...")
        result_no_context = pipeline.image_captioner.generate_caption_without_context(
            image_path="examples/test_image.jpg"
        )
        
        logger.info(f"Caption without context: {result_no_context['caption']}")
        logger.info(f"Confidence: {result_no_context['confidence']:.2f}")
        
        # Test image captioning with video context
        logger.info("Testing image captioning with video context...")
        result_with_context = pipeline.generate_caption(
            image_path="examples/test_image.jpg",
            video_path="examples/test_video.mp4",
            context_weight=0.7
        )
        
        logger.info(f"Caption with context: {result_with_context['caption']}")
        logger.info(f"Video context: {result_with_context['video_context']['context_text']}")
        logger.info(f"Confidence: {result_with_context['confidence']:.2f}")
        logger.info(f"Context relevance: {result_with_context['context_relevance']:.2f}")
        logger.info(f"Frames processed: {result_with_context['frames_processed']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in caption generation: {str(e)}")
        return False

def main():
    """Run all tests."""
    logger.info("Starting LFM2-VL model integration tests...")
    logger.info("=" * 60)
    
    tests = [
        ("Model Loading", test_model_loading),
        ("Pipeline Initialization", test_pipeline_initialization),
        ("Caption Generation", test_caption_generation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\nRunning test: {test_name}")
        logger.info("-" * 40)
        
        try:
            success = test_func()
            results.append((test_name, success))
            
            if success:
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")
                
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED with exception: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! The LFM2-VL integration is working correctly.")
    else:
        logger.warning("⚠️  Some tests failed. Check the logs for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
