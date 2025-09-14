#!/usr/bin/env python3
"""
Quick test script for the video-context image captioning pipeline.
"""

import sys
import os
import numpy as np
import cv2
from pathlib import Path

# Add src to path
sys.path.append('src')

def main():
    print("🎥 Video-Context Image Captioning with Liquid AI")
    print("=" * 50)
    
    # Create test files
    print("Creating test files...")
    examples_dir = Path('examples')
    examples_dir.mkdir(exist_ok=True)
    
    # Create test image
    sample_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    cv2.imwrite('examples/test_image.jpg', sample_image)
    
    # Create test video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('examples/test_video.mp4', fourcc, 30.0, (224, 224))
    for i in range(30):  # 1 second at 30 fps
        frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        out.write(frame)
    out.release()
    
    print("✅ Test files created")
    
    # Initialize pipeline
    print("Initializing pipeline with LFM2-VL model...")
    from src.pipeline import VideoContextCaptionPipeline
    
    pipeline = VideoContextCaptionPipeline(
        model_path='/home/frank/BrandInfluencerDatasetTraining/LFM2-VL-450M',
        device='cuda',
        max_frames=3
    )
    
    print("✅ Pipeline initialized successfully!")
    
    # Test image captioning without context
    print("\n📝 Testing image captioning without context...")
    result_no_context = pipeline.image_captioner.generate_caption_without_context(
        image_path='examples/test_image.jpg'
    )
    
    print(f"Caption: {result_no_context['caption']}")
    print(f"Confidence: {result_no_context['confidence']:.2f}")
    
    # Test image captioning with video context
    print("\n🎬 Testing image captioning with video context...")
    result_with_context = pipeline.generate_caption(
        image_path='examples/test_image.jpg',
        video_path='examples/test_video.mp4',
        context_weight=0.7
    )
    
    print(f"Caption with context: {result_with_context['caption']}")
    print(f"Video context: {result_with_context['video_context']['context_text']}")
    print(f"Confidence: {result_with_context['confidence']:.2f}")
    print(f"Context relevance: {result_with_context['context_relevance']:.2f}")
    print(f"Frames processed: {result_with_context['frames_processed']}")
    
    print("\n🎉 All tests completed successfully!")
    print("\nThe video-context image captioning system is working with the real LFM2-VL model!")

if __name__ == "__main__":
    main()
