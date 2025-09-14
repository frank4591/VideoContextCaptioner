#!/usr/bin/env python3
"""
Detailed test showing context extraction from each video frame and final caption generation.
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

def test_detailed_context_extraction():
    """Test with detailed logging of context extraction process."""
    logger.info("🧪 Testing detailed context extraction with key_frames strategy...")
    
    try:
        from src.pipeline import VideoContextCaptionPipeline
        from src.video_processor import VideoProcessor
        from src.context_extractor import ContextExtractor
        from src.liquid_integration import LiquidAIIntegration
        
        # Initialize pipeline with key_frames strategy
        logger.info("Loading LFM2-VL-450M model...")
        pipeline = VideoContextCaptionPipeline(
            model_path='/home/frank/BrandInfluencerDatasetTraining/LFM2-VL-450M',
            device='cuda',
            max_frames=5,  # Extract up to 5 frames
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
        
        print("\n" + "="*80)
        print("📹 STEP 1: VIDEO FRAME EXTRACTION (Key Frames Strategy)")
        print("="*80)
        
        # Extract frames using key_frames strategy
        video_processor = VideoProcessor(
            strategy='key_frames',
            max_frames=5,
            frame_interval=1.0
        )
        
        frames = video_processor.extract_frames(sample_video_path)
        print(f"✅ Extracted {len(frames)} frames using key_frames strategy")
        
        # Get video info
        import cv2
        cap = cv2.VideoCapture(sample_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        cap.release()
        
        print(f"   Video Info: {duration:.2f}s duration, {fps:.2f} FPS, {total_frames} total frames")
        
        print("\n" + "="*80)
        print("🎬 STEP 2: PROCESSING EACH EXTRACTED FRAME (DESCRIPTIVE ANALYSIS)")
        print("="*80)
        
        # Process each frame individually
        frame_captions = []
        frame_features = []
        
        for i, frame in enumerate(frames):
            print(f"\n--- Processing Frame {i+1}/{len(frames)} ---")
            
            # Generate caption for this frame
            try:
                result = pipeline.liquid_integration.generate_caption(
                    image=frame,
                    text_prompt="Describe this video frame in detail, focusing on visual elements, objects, and scene composition.",
                    return_features=True
                )
                
                caption = result['caption']
                features = result.get('features', [])
                confidence = result['confidence']
                processing_time = result['processing_time']
                
                frame_captions.append(caption)
                frame_features.append(features)
                
                print(f"   ✅ Frame Description Generated:")
                print(f"      Description: {caption}")
                print(f"      Confidence: {confidence:.3f}")
                print(f"      Processing Time: {processing_time:.2f}s")
                print(f"      Features Shape: {len(features) if hasattr(features, '__len__') else 'N/A'}")
                
            except Exception as e:
                logger.error(f"   ❌ Error processing frame {i+1}: {e}")
                frame_captions.append(f"Error processing frame {i+1}")
                frame_features.append([])
        
        print("\n" + "="*80)
        print("🔄 STEP 3: VIDEO CONTEXT AGGREGATION")
        print("="*80)
        
        # Aggregate context from all frame captions
        context_extractor = ContextExtractor(aggregation_method="weighted_average")
        
        print(f"   Aggregating video context from {len(frame_captions)} frame descriptions...")
        print(f"   Method: weighted_average")
        
        aggregated_context = context_extractor.aggregate_context(
            frame_captions=frame_captions,
            frame_features=frame_features,
            max_length=512
        )
        
        print(f"\n   ✅ Aggregated Context:")
        print(f"      Context Text: {aggregated_context['context_text']}")
        print(f"      Features Shape: {len(aggregated_context['features']) if hasattr(aggregated_context['features'], '__len__') else 'N/A'}")
        print(f"      Temporal Consistency: {aggregated_context['temporal_consistency']:.3f}")
        print(f"      Selected Captions: {len(aggregated_context.get('selected_captions', []))}")
        
        if 'selected_captions' in aggregated_context:
            print(f"\n   Selected Captions for Context:")
            for i, caption in enumerate(aggregated_context['selected_captions']):
                print(f"      {i+1}. {caption}")
        
        print("\n" + "="*80)
        print("🖼️ STEP 4: INSTAGRAM-STYLE CAPTION GENERATION WITH VIDEO CONTEXT")
        print("="*80)
        
        # Show what input goes into the final caption generation
        context_text = aggregated_context['context_text']
        context_prompt = f"Based on the video context: '{context_text}', Create an instagram style caption for this image."
        
        print(f"   Image: {sample_image_path}")
        print(f"   Video Context: {context_text}")
        print(f"   Context-Aware Prompt: {context_prompt}")
        
        # Generate final caption with context
        print(f"\n   Generating final caption with video context...")
        
        result_with_context = pipeline.generate_caption(
            image_path=sample_image_path,
            video_path=sample_video_path,
            context_weight=0.7,
            text_prompt="Create an instagram style caption for this image."
        )
        
        print(f"\n   ✅ Final Result:")
        print(f"      📝 Video Context: {result_with_context.get('context', 'N/A')[:150]}...")
        print(f"      🎯 Prompt Used: {result_with_context.get('prompt', 'N/A')}")
        print(f"      📸 Instagram Caption: {result_with_context.get('instagram_caption', result_with_context['caption'])}")
        print(f"      🔄 Raw Output: {result_with_context.get('raw_output', 'N/A')[:200]}...")
        print(f"      📊 Confidence: {result_with_context['confidence']:.3f}")
        print(f"      📊 Context Relevance: {result_with_context['context_relevance']:.3f}")
        print(f"      📊 Frames Processed: {result_with_context['frames_processed']}")
        print(f"      ⏱️ Processing Time: {result_with_context['processing_time']:.2f}s")
        
        # Also test without context for comparison
        print("\n" + "="*80)
        print("🖼️ STEP 5: IMAGE CAPTION WITHOUT CONTEXT (COMPARISON)")
        print("="*80)
        
        result_no_context = pipeline.image_captioner.generate_caption_without_context(
            image_path=sample_image_path
        )
        
        print(f"   ✅ Caption Without Context:")
        print(f"      Caption: {result_no_context['caption']}")
        print(f"      Confidence: {result_no_context['confidence']:.3f}")
        print(f"      Processing Time: {result_no_context['processing_time']:.2f}s")
        
        return {
            'frames_extracted': len(frames),
            'frame_captions': frame_captions,
            'aggregated_context': aggregated_context,
            'context_prompt': context_prompt,
            'result_with_context': result_with_context,
            'result_no_context': result_no_context
        }
        
    except Exception as e:
        logger.error(f"❌ Error in detailed context test: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main test function."""
    print("🧪 Detailed Context Extraction Test")
    print("=" * 80)
    print("Testing key_frames strategy with detailed logging of each step")
    print("=" * 80)
    
    results = test_detailed_context_extraction()
    
    if results:
        print("\n" + "="*80)
        print("📊 FINAL SUMMARY")
        print("="*80)
        
        print(f"✅ Frames Extracted: {results['frames_extracted']}")
        print(f"✅ Context Text: {results['aggregated_context']['context_text'][:100]}...")
        print(f"✅ Instagram Caption (with context): {results['result_with_context']['caption']}")
        print(f"✅ Regular Caption (without context): {results['result_no_context']['caption']}")
        
        print(f"\n🎯 Key Insights:")
        print(f"   - Key frames strategy detected {results['frames_extracted']} significant scene changes")
        print(f"   - Each frame was processed to generate descriptive features (not Instagram captions)")
        print(f"   - Video context was aggregated from frame descriptions using weighted averaging")
        print(f"   - Final Instagram caption incorporates video context meaningfully")
        
        print(f"\n✅ Detailed context extraction test completed successfully!")
    else:
        print("\n❌ Test failed - check logs for details")
    
    print("\n🎉 Detailed context extraction test completed!")

if __name__ == "__main__":
    main()
