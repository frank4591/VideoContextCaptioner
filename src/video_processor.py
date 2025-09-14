"""
Video frame extraction and processing module.

Handles video frame extraction using various strategies and leverages
OpenCV for efficient video processing.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict
import logging
from pathlib import Path

class VideoProcessor:
    """
    Handles video frame extraction using various strategies.
    
    Leverages OpenCV for efficient video processing and implements
    different strategies for selecting representative frames.
    """
    
    def __init__(
        self,
        strategy: str = "key_frames",
        max_frames: int = 10,
        frame_interval: float = 1.0
    ):
        """
        Initialize video processor.
        
        Args:
            strategy: Frame extraction strategy ("key_frames", "uniform", "adaptive")
            max_frames: Maximum number of frames to extract
            frame_interval: Interval between frames for uniform sampling (seconds)
        """
        self.strategy = strategy
        self.max_frames = max_frames
        self.frame_interval = frame_interval
        
        logging.info(f"VideoProcessor initialized with {strategy} strategy")
    
    def extract_frames(self, video_path: str) -> List[np.ndarray]:
        """
        Extract frames from video using the specified strategy.
        
        Args:
            video_path: Path to the input video
            
        Returns:
            List of extracted frames as numpy arrays
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        try:
            if self.strategy == "key_frames":
                frames = self._extract_key_frames(cap)
            elif self.strategy == "uniform":
                frames = self._extract_uniform_frames(cap)
            elif self.strategy == "adaptive":
                frames = self._extract_adaptive_frames(cap)
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")
            
            # Ensure we don't exceed max_frames
            if len(frames) > self.max_frames:
                frames = self._select_representative_frames(frames, self.max_frames)
            
            logging.info(f"Extracted {len(frames)} frames from {video_path}")
            return frames
            
        finally:
            cap.release()
    
    def _extract_key_frames(self, cap: cv2.VideoCapture) -> List[np.ndarray]:
        """Extract key frames using scene change detection."""
        frames = []
        prev_frame = None
        frame_count = 0
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale for comparison
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_frame is not None:
                # Calculate frame difference
                diff = cv2.absdiff(prev_frame, gray)
                mean_diff = np.mean(diff)
                
                # If significant change detected, add frame
                if mean_diff > 30:  # Threshold for scene change
                    frames.append(frame)
                    frame_count += 1
                    
                    if frame_count >= self.max_frames:
                        break
            
            prev_frame = gray.copy()
        
        # If no key frames detected, sample uniformly
        if not frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frames = self._extract_uniform_frames(cap)
        
        return frames
    
    def _extract_uniform_frames(self, cap: cv2.VideoCapture) -> List[np.ndarray]:
        """Extract frames at uniform intervals."""
        frames = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval_frames = int(fps * self.frame_interval)
        
        frame_count = 0
        current_frame = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if current_frame % frame_interval_frames == 0:
                frames.append(frame)
                frame_count += 1
                
                if frame_count >= self.max_frames:
                    break
            
            current_frame += 1
        
        return frames
    
    def _extract_adaptive_frames(self, cap: cv2.VideoCapture) -> List[np.ndarray]:
        """Extract frames adaptively based on content complexity."""
        frames = []
        frame_scores = []
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calculate frame complexity score
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            frame_scores.append((frame_count, frame, laplacian_var))
            frame_count += 1
        
        # Sort by complexity and select top frames
        frame_scores.sort(key=lambda x: x[2], reverse=True)
        selected_frames = frame_scores[:self.max_frames]
        selected_frames.sort(key=lambda x: x[0])  # Sort by original order
        
        return [frame for _, frame, _ in selected_frames]
    
    def _select_representative_frames(
        self, 
        frames: List[np.ndarray], 
        max_frames: int
    ) -> List[np.ndarray]:
        """Select most representative frames from a larger set."""
        if len(frames) <= max_frames:
            return frames
        
        # Calculate frame differences
        frame_diffs = []
        for i in range(1, len(frames)):
            diff = cv2.absdiff(frames[i-1], frames[i])
            mean_diff = np.mean(diff)
            frame_diffs.append((i, mean_diff))
        
        # Select frames with highest differences (most distinct)
        frame_diffs.sort(key=lambda x: x[1], reverse=True)
        selected_indices = [0] + [idx for idx, _ in frame_diffs[:max_frames-1]]
        selected_indices.sort()
        
        return [frames[i] for i in selected_indices]
    
    def get_video_info(self, video_path: str) -> Dict[str, any]:
        """
        Get information about the video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary containing video information
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            return {
                "fps": fps,
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "duration": duration
            }
        finally:
            cap.release()
    
    def extract_frames_with_timestamps(self, video_path: str) -> List[Tuple[np.ndarray, float]]:
        """
        Extract frames with their timestamps.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            List of (frame, timestamp) tuples
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        frames_with_timestamps = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        try:
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                timestamp = frame_count / fps if fps > 0 else 0
                frames_with_timestamps.append((frame, timestamp))
                frame_count += 1
                
                if len(frames_with_timestamps) >= self.max_frames:
                    break
        
        finally:
            cap.release()
        
        return frames_with_timestamps
