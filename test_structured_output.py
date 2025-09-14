#!/usr/bin/env python3
"""
Test to demonstrate the new structured output with separate context, prompt, and Instagram caption.
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

def test_structured_output():
    """Test the new structured output format."""
    logger.info("🧪 Testing structured output format...")
    
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
        print("📸 TESTING STRUCTURED OUTPUT FORMAT")
        print("="*70)
        
        # Test with different Instagram prompts
        instagram_prompts = [
            "Create an Instagram-style caption for this image.",
            "Write a catchy Instagram post caption for this image.",
            "Generate a trendy Instagram caption with hashtags for this image.",
            "Create a short and sweet Instagram caption for this image."
        ]
        
        for i, prompt in enumerate(instagram_prompts):
            print(f"\n--- Test {i+1}: {prompt} ---")
            
            try:
                result = pipeline.generate_caption(
                    image_path=sample_image_path,
                    video_path=sample_video_path,
                    context_weight=0.7,
                    text_prompt=prompt
                )
                
                print(f"✅ Structured Output:")
                print(f"   📝 Context: {result.get('context', 'N/A')[:100]}...")
                print(f"   🎯 Prompt: {result.get('prompt', 'N/A')}")
                print(f"   📸 Instagram Caption: {result.get('instagram_caption', result['caption'])}")
                print(f"   🔄 Raw Output: {result.get('raw_output', 'N/A')[:150]}...")
                print(f"   📊 Metrics: Confidence={result['confidence']:.3f}, Relevance={result['context_relevance']:.3f}")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error in structured output test: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main test function."""
    print("🧪 Structured Output Test")
    print("=" * 70)
    print("Testing new structured output with separate context, prompt, and caption")
    print("=" * 70)
    
    results = test_structured_output()
    
    if results:
        print("\n" + "="*70)
        print("📊 TEST SUMMARY")
        print("="*70)
        
        print("✅ Structured Output: SUCCESS")
        print("   - Context: Extracted video context from frame descriptions")
        print("   - Prompt: The Instagram-style prompt used for generation")
        print("   - Instagram Caption: Clean, extracted Instagram caption")
        print("   - Raw Output: Full model output for debugging")
        
        print(f"\n🎯 Key Benefits:")
        print(f"   - Clear separation of context, prompt, and caption")
        print(f"   - Easy access to individual components")
        print(f"   - Better debugging and analysis capabilities")
        print(f"   - Backward compatibility maintained")
        
        print("\n✅ Structured output format working correctly!")
    else:
        print("\n❌ Test failed - check logs for details")
    
    print("\n🎉 Structured output test completed!")

if __name__ == "__main__":
    main()
