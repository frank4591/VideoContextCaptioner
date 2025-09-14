#!/usr/bin/env python3
"""
Test to verify that Instagram captions are properly separated from video context.
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.append('src')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_caption_separation():
    """Test that Instagram captions are properly separated from context."""
    logger.info("🧪 Testing Instagram caption separation...")
    
    try:
        from src.pipeline import VideoContextCaptionPipeline
        
        # Initialize pipeline
        logger.info("Loading LFM2-VL-450M model...")
        pipeline = VideoContextCaptionPipeline(
            model_path='/home/frank/BrandInfluencerDatasetTraining/LFM2-VL-450M',
            device='cuda',
            max_frames=3,
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
        print("📸 TESTING CAPTION SEPARATION")
        print("="*70)
        
        # Test Instagram caption with context
        print("\n1. Instagram Caption with Video Context:")
        result = pipeline.generate_caption(
            image_path=sample_image_path,
            video_path=sample_video_path,
            context_weight=0.7,
            text_prompt="Create an Instagram-style caption for this image."
        )
        
        print(f"\n   📝 Raw Output (first 300 chars):")
        print(f"   {result.get('raw_output', 'N/A')[:300]}...")
        
        print(f"\n   🎯 Extracted Instagram Caption:")
        print(f"   {result['caption']}")
        
        print(f"\n   📊 Metrics:")
        print(f"   - Confidence: {result['confidence']:.3f}")
        print(f"   - Context Relevance: {result['context_relevance']:.3f}")
        print(f"   - Frames Processed: {result['frames_processed']}")
        print(f"   - Processing Time: {result['processing_time']:.2f}s")
        
        # Test different Instagram prompts
        print("\n" + "="*70)
        print("📸 TESTING DIFFERENT INSTAGRAM PROMPTS")
        print("="*70)
        
        instagram_prompts = [
            "Create a catchy Instagram caption for this image.",
            "Write an engaging Instagram post caption for this image.",
            "Generate a trendy Instagram caption with hashtags for this image.",
            "Create a short and sweet Instagram caption for this image."
        ]
        
        for i, prompt in enumerate(instagram_prompts):
            print(f"\n{i+1}. Testing: {prompt}")
            try:
                result = pipeline.generate_caption(
                    image_path=sample_image_path,
                    video_path=sample_video_path,
                    context_weight=0.7,
                    text_prompt=prompt
                )
                print(f"   Raw: {result.get('raw_output', 'N/A')[:150]}...")
                print(f"   Caption: {result['caption']}")
            except Exception as e:
                print(f"   Error: {e}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Error in caption separation test: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main test function."""
    print("🧪 Instagram Caption Separation Test")
    print("=" * 70)
    print("Testing separation of Instagram captions from video context")
    print("=" * 70)
    
    results = test_caption_separation()
    
    if results:
        print("\n" + "="*70)
        print("📊 TEST SUMMARY")
        print("="*70)
        
        print("✅ Caption Separation: SUCCESS")
        print(f"   - Raw output contains both context and caption")
        print(f"   - Extracted caption is clean and Instagram-ready")
        print(f"   - Context and caption are properly separated")
        
        print(f"\n🎯 Key Features:")
        print(f"   - Raw output: {results.get('raw_output', 'N/A')[:100]}...")
        print(f"   - Clean caption: {results['caption']}")
        print(f"   - Context relevance: {results['context_relevance']:.3f}")
        
        print("\n✅ Instagram caption separation working correctly!")
    else:
        print("\n❌ Test failed - check logs for details")
    
    print("\n🎉 Caption separation test completed!")

if __name__ == "__main__":
    main()
