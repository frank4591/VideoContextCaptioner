#!/usr/bin/env python3
"""
Debug script to test Instagram caption extraction.
"""

import sys
import os
import logging
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append('src')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_caption_extraction():
    """Test Instagram caption extraction with debug output."""
    logger.info("🧪 Testing Instagram caption extraction...")
    
    try:
        from src.pipeline import VideoContextCaptionPipeline
        
        # Initialize pipeline
        logger.info("Loading LFM2-VL-450M model...")
        pipeline = VideoContextCaptionPipeline(
            model_path='/home/frank/BrandInfluencerDatasetTraining/LFM2-VL-450M',
            device='cuda',
            max_frames=2,  # Use fewer frames for faster testing
            frame_extraction_strategy='key_frames'
        )
        
        # Check if we're using the actual model
        if pipeline.liquid_integration.model.get('is_mock', True):
            logger.warning("⚠️ Using mock model instead of actual LFM model!")
            return None
        
        logger.info("✅ Actual LFM2-VL model loaded successfully")
        
        # Test files
        sample_image_path = 'examples/sample_image.jpg'
        sample_video_path = 'examples/sample_video.mp4'
        
        if not Path(sample_image_path).exists():
            logger.error(f"Sample image not found: {sample_image_path}")
            return None
            
        if not Path(sample_video_path).exists():
            logger.error(f"Sample video not found: {sample_video_path}")
            return None
        
        print("\n" + "="*70)
        print("🔍 DEBUGGING CAPTION EXTRACTION")
        print("="*70)
        
        # Test Instagram caption with context
        print("\n1. Testing Instagram Caption Extraction:")
        result = pipeline.generate_caption(
            image_path=sample_image_path,
            video_path=sample_video_path,
            context_weight=0.7,
            text_prompt="Create an Instagram-style caption for this image."
        )
        
        print(f"\n📊 Results:")
        print(f"   Raw Output: {result.get('raw_output', 'N/A')}")
        print(f"   Context: {result.get('context', 'N/A')[:100]}...")
        print(f"   Prompt: {result.get('prompt', 'N/A')}")
        print(f"   Instagram Caption: {result.get('instagram_caption', 'N/A')}")
        print(f"   Caption (backward compat): {result.get('caption', 'N/A')}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Error in caption extraction test: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main test function."""
    print("🧪 Caption Extraction Debug Test")
    print("=" * 70)
    print("Testing Instagram caption extraction with debug output")
    print("=" * 70)
    
    results = test_caption_extraction()
    
    if results:
        print("\n" + "="*70)
        print("📊 DEBUG SUMMARY")
        print("="*70)
        
        print("✅ Caption Extraction: COMPLETED")
        print(f"   - Raw output length: {len(results.get('raw_output', ''))}")
        print(f"   - Instagram caption length: {len(results.get('instagram_caption', ''))}")
        print(f"   - Context length: {len(results.get('context', ''))}")
        
        print("\n🎯 Analysis:")
        print(f"   - Raw output contains: {results.get('raw_output', 'N/A')[:200]}...")
        print(f"   - Extracted caption: {results.get('instagram_caption', 'N/A')}")
        
        print("\n✅ Debug test completed!")
    else:
        print("\n❌ Test failed - check logs for details")
    
    print("\n🎉 Caption extraction debug completed!")

if __name__ == "__main__":
    main()


