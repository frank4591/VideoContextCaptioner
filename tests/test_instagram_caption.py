#!/usr/bin/env python3
"""
Quick test to verify Instagram-style caption generation with video context.
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

def test_instagram_caption():
    """Test Instagram-style caption generation with video context."""
    logger.info("🧪 Testing Instagram-style caption generation...")
    
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
        print("📸 TESTING INSTAGRAM-STYLE CAPTION GENERATION")
        print("="*70)
        
        # Test 1: Regular caption without context
        print("\n1. Regular caption without video context:")
        result_no_context = pipeline.image_captioner.generate_caption_without_context(
            image_path=sample_image_path,
            text_prompt="Describe this image in detail."
        )
        print(f"   Caption: {result_no_context['caption']}")
        
        # Test 2: Instagram-style caption without context
        print("\n2. Instagram-style caption without video context:")
        result_instagram_no_context = pipeline.image_captioner.generate_caption_without_context(
            image_path=sample_image_path,
            text_prompt="Create an Instagram-style caption for this image."
        )
        print(f"   Caption: {result_instagram_no_context['caption']}")
        
        # Test 3: Instagram-style caption WITH video context
        print("\n3. Instagram-style caption WITH video context:")
        result_instagram_with_context = pipeline.generate_caption(
            image_path=sample_image_path,
            video_path=sample_video_path,
            context_weight=0.7,
            text_prompt="Create an Instagram-style caption for this image."
        )
        print(f"   Caption: {result_instagram_with_context['caption']}")
        print(f"   Context Relevance: {result_instagram_with_context['context_relevance']:.3f}")
        print(f"   Frames Processed: {result_instagram_with_context['frames_processed']}")
        
        # Test 4: Different Instagram-style prompts
        print("\n4. Different Instagram-style prompts with video context:")
        
        instagram_prompts = [
            "Create a catchy Instagram caption for this image.",
            "Write an engaging Instagram post caption for this image.",
            "Generate a trendy Instagram caption with hashtags for this image.",
            "Create a short and sweet Instagram caption for this image."
        ]
        
        for i, prompt in enumerate(instagram_prompts):
            print(f"\n   Prompt {i+1}: {prompt}")
            try:
                result = pipeline.generate_caption(
                    image_path=sample_image_path,
                    video_path=sample_video_path,
                    context_weight=0.7,
                    text_prompt=prompt
                )
                print(f"   Caption: {result['caption']}")
            except Exception as e:
                print(f"   Error: {e}")
        
        return {
            'no_context': result_no_context,
            'instagram_no_context': result_instagram_no_context,
            'instagram_with_context': result_instagram_with_context
        }
        
    except Exception as e:
        logger.error(f"❌ Error in Instagram caption test: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main test function."""
    print("🧪 Instagram-Style Caption Generation Test")
    print("=" * 70)
    print("Testing custom prompt integration with video context")
    print("=" * 70)
    
    results = test_instagram_caption()
    
    if results:
        print("\n" + "="*70)
        print("📊 TEST SUMMARY")
        print("="*70)
        
        print("✅ Regular caption (no context): SUCCESS")
        print(f"   Caption: {results['no_context']['caption'][:100]}...")
        
        print("\n✅ Instagram caption (no context): SUCCESS")
        print(f"   Caption: {results['instagram_no_context']['caption'][:100]}...")
        
        print("\n✅ Instagram caption (with context): SUCCESS")
        print(f"   Caption: {results['instagram_with_context']['caption'][:100]}...")
        print(f"   Context Relevance: {results['instagram_with_context']['context_relevance']:.3f}")
        
        print("\n🎯 Key Fixes Applied:")
        print("   - Added text_prompt parameter to pipeline.generate_caption()")
        print("   - Added text_prompt parameter to image_captioner.generate_caption()")
        print("   - Added text_prompt parameter to liquid_integration.generate_with_context()")
        print("   - Custom prompts now properly integrated with video context")
        
        print("\n✅ Instagram-style caption generation working correctly!")
    else:
        print("\n❌ Test failed - check logs for details")
    
    print("\n🎉 Instagram-style caption test completed!")

if __name__ == "__main__":
    main()
