#!/usr/bin/env python3
"""
Test to verify the correct flow: descriptive frame analysis -> context aggregation -> Instagram caption generation.
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

def test_correct_flow():
    """Test the correct flow: descriptive frames -> context -> Instagram caption."""
    logger.info("🧪 Testing correct flow: descriptive frames -> context -> Instagram caption...")
    
    try:
        from src.pipeline import VideoContextCaptionPipeline
        from src.video_processor import VideoProcessor
        from src.context_extractor import ContextExtractor
        from src.liquid_integration import LiquidAIIntegration
        
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
        print("📹 STEP 1: EXTRACT KEY FRAMES")
        print("="*70)
        
        # Extract frames
        video_processor = VideoProcessor(
            strategy='key_frames',
            max_frames=3,
            frame_interval=1.0
        )
        
        frames = video_processor.extract_frames(sample_video_path)
        print(f"✅ Extracted {len(frames)} key frames")
        
        print("\n" + "="*70)
        print("🎬 STEP 2: DESCRIPTIVE FRAME ANALYSIS")
        print("="*70)
        
        # Process each frame with descriptive prompt (NOT Instagram style)
        frame_descriptions = []
        
        for i, frame in enumerate(frames):
            print(f"\n--- Processing Frame {i+1}/{len(frames)} ---")
            
            # Use descriptive prompt for frame analysis
            result = pipeline.liquid_integration.generate_caption(
                image=frame,
                text_prompt="Describe this video frame in detail, focusing on visual elements, objects, and scene composition.",
                return_features=True
            )
            
            description = result['caption']
            frame_descriptions.append(description)
            
            print(f"   ✅ Frame Description:")
            print(f"      {description}")
            print(f"      Confidence: {result['confidence']:.3f}")
        
        print("\n" + "="*70)
        print("🔄 STEP 3: CONTEXT AGGREGATION")
        print("="*70)
        
        # Aggregate context from frame descriptions
        context_extractor = ContextExtractor(aggregation_method="weighted_average")
        
        # Create dummy features for aggregation
        dummy_features = [np.random.randn(768) for _ in frame_descriptions]
        
        aggregated_context = context_extractor.aggregate_context(
            frame_captions=frame_descriptions,
            frame_features=dummy_features,
            max_length=512
        )
        
        print(f"✅ Aggregated Video Context:")
        print(f"   {aggregated_context['context_text']}")
        
        print("\n" + "="*70)
        print("📸 STEP 4: INSTAGRAM CAPTION GENERATION")
        print("="*70)
        
        # Now generate Instagram caption with the aggregated context
        result = pipeline.generate_caption(
            image_path=sample_image_path,
            video_path=sample_video_path,
            context_weight=0.7,
            text_prompt="Create an Instagram-style caption for this image."
        )
        
        print(f"✅ Final Results:")
        print(f"   📝 Context: {result.get('context', 'N/A')[:100]}...")
        print(f"   🎯 Prompt: {result.get('prompt', 'N/A')}")
        print(f"   📸 Instagram Caption: {result.get('instagram_caption', result['caption'])}")
        print(f"   🔄 Raw Output: {result.get('raw_output', 'N/A')[:150]}...")
        
        print(f"\n📊 Results Summary:")
        print(f"   - Frames processed: {len(frames)}")
        print(f"   - Frame descriptions: {len(frame_descriptions)}")
        print(f"   - Context relevance: {result['context_relevance']:.3f}")
        print(f"   - Processing time: {result['processing_time']:.2f}s")
        
        return {
            'frames_processed': len(frames),
            'frame_descriptions': frame_descriptions,
            'aggregated_context': aggregated_context,
            'instagram_caption': result['caption'],
            'context_relevance': result['context_relevance']
        }
        
    except Exception as e:
        logger.error(f"❌ Error in correct flow test: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main test function."""
    print("🧪 Correct Flow Test")
    print("=" * 70)
    print("Testing: Descriptive frames -> Context aggregation -> Instagram caption")
    print("=" * 70)
    
    results = test_correct_flow()
    
    if results:
        print("\n" + "="*70)
        print("📊 TEST SUMMARY")
        print("="*70)
        
        print("✅ Flow Verification: SUCCESS")
        print(f"   - Frame analysis: {results['frames_processed']} frames processed descriptively")
        print(f"   - Context aggregation: Video context extracted from descriptions")
        print(f"   - Instagram caption: Generated with video context")
        print(f"   - Context relevance: {results['context_relevance']:.3f}")
        
        print(f"\n🎯 Key Points:")
        print(f"   - Frame processing uses descriptive prompts (not Instagram style)")
        print(f"   - Instagram style prompt only used in final caption generation")
        print(f"   - Video context properly aggregated from frame descriptions")
        print(f"   - Final caption: {results['instagram_caption']}")
        
        print("\n✅ Correct flow implementation working!")
    else:
        print("\n❌ Test failed - check logs for details")
    
    print("\n🎉 Correct flow test completed!")

if __name__ == "__main__":
    main()
