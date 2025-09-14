#!/usr/bin/env python3
"""
Simple analysis of context extraction process without heavy model loading.
"""

import sys
import os
import logging
from pathlib import Path
import cv2
import numpy as np

# Add src to path
sys.path.append('src')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_video_frames():
    """Analyze video frame extraction without model loading."""
    logger.info("🔍 Analyzing video frame extraction process...")
    
    try:
        from src.video_processor import VideoProcessor
        
        sample_video_path = 'examples/sample_video.mp4'
        
        if not Path(sample_video_path).exists():
            logger.error(f"Sample video not found: {sample_video_path}")
            return None
        
        print("\n" + "="*80)
        print("📹 VIDEO ANALYSIS")
        print("="*80)
        
        # Get video info
        cap = cv2.VideoCapture(sample_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        print(f"   File: {sample_video_path}")
        print(f"   Duration: {duration:.2f} seconds")
        print(f"   FPS: {fps:.2f}")
        print(f"   Total Frames: {total_frames}")
        print(f"   Resolution: {width}x{height}")
        
        print("\n" + "="*80)
        print("🎯 KEY FRAMES EXTRACTION ANALYSIS")
        print("="*80)
        
        # Analyze key frames extraction
        processor = VideoProcessor(
            strategy='key_frames',
            max_frames=5,
            frame_interval=1.0
        )
        
        print(f"   Strategy: key_frames")
        print(f"   Max Frames: 5")
        print(f"   Scene Change Threshold: 30")
        
        # Simulate the key frames extraction process
        cap = cv2.VideoCapture(sample_video_path)
        frames = []
        prev_frame = None
        frame_count = 0
        scene_changes = []
        
        print(f"\n   Analyzing scene changes...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            current_time = current_frame_num / fps
            
            # Convert to grayscale for comparison
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_frame is not None:
                # Calculate frame difference
                diff = cv2.absdiff(prev_frame, gray)
                mean_diff = np.mean(diff)
                
                # If significant change detected, add frame
                if mean_diff > 30:  # Threshold for scene change
                    frames.append(frame)
                    scene_changes.append({
                        'frame_number': current_frame_num,
                        'timestamp': current_time,
                        'difference': mean_diff
                    })
                    frame_count += 1
                    
                    print(f"   ✅ Scene Change {frame_count}: Frame {current_frame_num} at {current_time:.2f}s (diff: {mean_diff:.2f})")
                    
                    if frame_count >= 5:
                        break
            
            prev_frame = gray.copy()
        
        cap.release()
        
        print(f"\n   Total Scene Changes Detected: {len(scene_changes)}")
        print(f"   Frames Extracted: {len(frames)}")
        
        print("\n" + "="*80)
        print("📝 CONTEXT EXTRACTION PROCESS")
        print("="*80)
        
        print(f"   Step 1: Extract {len(frames)} key frames from video")
        print(f"   Step 2: Process each frame through LFM2-VL model")
        print(f"   Step 3: Generate caption for each frame")
        print(f"   Step 4: Aggregate captions using weighted averaging")
        print(f"   Step 5: Use aggregated context for final image caption")
        
        print(f"\n   Frame Timestamps:")
        for i, change in enumerate(scene_changes):
            print(f"      Frame {i+1}: {change['timestamp']:.2f}s (Frame {change['frame_number']})")
        
        print(f"\n   Context Creation Process:")
        print(f"      - Each frame → LFM2-VL model → Caption")
        print(f"      - Captions → Context Extractor → Aggregated Context")
        print(f"      - Aggregated Context → Final Image Caption")
        
        print("\n" + "="*80)
        print("🖼️ FINAL CAPTION GENERATION INPUT")
        print("="*80)
        
        print(f"   Image: examples/sample_image.jpg")
        print(f"   Video Context: [Aggregated from {len(frames)} frame captions]")
        print(f"   Context-Aware Prompt: 'Based on the video context: [context], describe this image in detail.'")
        print(f"   Model: LFM2-VL-450M with chat template format")
        
        return {
            'video_info': {
                'duration': duration,
                'fps': fps,
                'total_frames': total_frames,
                'resolution': f"{width}x{height}"
            },
            'scene_changes': scene_changes,
            'frames_extracted': len(frames)
        }
        
    except Exception as e:
        logger.error(f"❌ Error in video analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main analysis function."""
    print("🔍 Video Context Extraction Analysis")
    print("=" * 80)
    print("Analyzing key_frames strategy and context extraction process")
    print("=" * 80)
    
    results = analyze_video_frames()
    
    if results:
        print("\n" + "="*80)
        print("📊 ANALYSIS SUMMARY")
        print("="*80)
        
        video_info = results['video_info']
        print(f"✅ Video Duration: {video_info['duration']:.2f} seconds")
        print(f"✅ Video FPS: {video_info['fps']:.2f}")
        print(f"✅ Total Frames: {video_info['total_frames']}")
        print(f"✅ Resolution: {video_info['resolution']}")
        
        print(f"\n✅ Scene Changes Detected: {len(results['scene_changes'])}")
        print(f"✅ Frames Extracted: {results['frames_extracted']}")
        
        print(f"\n🎯 Key Points:")
        print(f"   - Key frames strategy detects significant visual changes")
        print(f"   - Frames are extracted at irregular intervals (when changes occur)")
        print(f"   - Each frame is processed individually through LFM2-VL")
        print(f"   - Context is aggregated from all frame captions")
        print(f"   - Final image caption incorporates the video context")
        
        print(f"\n✅ Video context extraction analysis completed!")
    else:
        print("\n❌ Analysis failed - check logs for details")
    
    print("\n🎉 Video context extraction analysis completed!")

if __name__ == "__main__":
    main()
