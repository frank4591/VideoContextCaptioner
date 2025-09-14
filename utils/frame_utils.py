"""
Frame processing utilities.

This module provides utility functions for video frame processing,
image preprocessing, and frame analysis.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
from pathlib import Path

class FrameUtils:
    """
    Utility class for frame processing operations.
    """
    
    @staticmethod
    def resize_frame(frame: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Resize frame to target size while maintaining aspect ratio.
        
        Args:
            frame: Input frame as numpy array
            target_size: Target size (width, height)
            
        Returns:
            Resized frame
        """
        return cv2.resize(frame, target_size)
    
    @staticmethod
    def normalize_frame(frame: np.ndarray, mean: List[float] = [0.485, 0.456, 0.406], 
                       std: List[float] = [0.229, 0.224, 0.225]) -> np.ndarray:
        """
        Normalize frame using ImageNet statistics.
        
        Args:
            frame: Input frame as numpy array
            mean: Mean values for normalization
            std: Standard deviation values for normalization
            
        Returns:
            Normalized frame
        """
        frame = frame.astype(np.float32) / 255.0
        
        # Normalize each channel
        for i in range(3):
            frame[:, :, i] = (frame[:, :, i] - mean[i]) / std[i]
        
        return frame
    
    @staticmethod
    def preprocess_frame(frame: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Complete preprocessing pipeline for a frame.
        
        Args:
            frame: Input frame as numpy array
            target_size: Target size for resizing
            
        Returns:
            Preprocessed frame ready for model input
        """
        # Convert BGR to RGB if needed
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize frame
        frame = FrameUtils.resize_frame(frame, target_size)
        
        # Normalize frame
        frame = FrameUtils.normalize_frame(frame)
        
        return frame
    
    @staticmethod
    def calculate_frame_difference(frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Calculate the difference between two frames.
        
        Args:
            frame1: First frame
            frame2: Second frame
            
        Returns:
            Mean absolute difference
        """
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY) if len(frame1.shape) == 3 else frame1
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY) if len(frame2.shape) == 3 else frame2
        
        # Calculate absolute difference
        diff = cv2.absdiff(gray1, gray2)
        
        return np.mean(diff)
    
    @staticmethod
    def detect_scene_changes(frames: List[np.ndarray], threshold: float = 30.0) -> List[int]:
        """
        Detect scene changes in a sequence of frames.
        
        Args:
            frames: List of frames
            threshold: Threshold for scene change detection
            
        Returns:
            List of frame indices where scene changes occur
        """
        if len(frames) < 2:
            return []
        
        scene_changes = []
        
        for i in range(1, len(frames)):
            diff = FrameUtils.calculate_frame_difference(frames[i-1], frames[i])
            if diff > threshold:
                scene_changes.append(i)
        
        return scene_changes
    
    @staticmethod
    def calculate_frame_complexity(frame: np.ndarray) -> float:
        """
        Calculate the complexity of a frame using Laplacian variance.
        
        Args:
            frame: Input frame
            
        Returns:
            Complexity score
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        # Calculate Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        return laplacian_var
    
    @staticmethod
    def extract_optical_flow(frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """
        Extract optical flow between two frames.
        
        Args:
            frame1: First frame
            frame2: Second frame
            
        Returns:
            Optical flow array
        """
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY) if len(frame1.shape) == 3 else frame1
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY) if len(frame2.shape) == 3 else frame2
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowPyrLK(gray1, gray2, None, None)
        
        return flow
    
    @staticmethod
    def create_frame_thumbnail(frame: np.ndarray, max_size: Tuple[int, int] = (150, 150)) -> np.ndarray:
        """
        Create a thumbnail of a frame.
        
        Args:
            frame: Input frame
            max_size: Maximum size for thumbnail
            
        Returns:
            Thumbnail frame
        """
        h, w = frame.shape[:2]
        max_h, max_w = max_size
        
        # Calculate scaling factor
        scale = min(max_h / h, max_w / w)
        
        # Calculate new dimensions
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        # Resize frame
        thumbnail = cv2.resize(frame, (new_w, new_h))
        
        return thumbnail
    
    @staticmethod
    def save_frame(frame: np.ndarray, output_path: str) -> bool:
        """
        Save a frame to disk.
        
        Args:
            frame: Frame to save
            output_path: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save frame
            success = cv2.imwrite(output_path, frame)
            
            if success:
                logging.info(f"Frame saved to {output_path}")
            else:
                logging.error(f"Failed to save frame to {output_path}")
            
            return success
            
        except Exception as e:
            logging.error(f"Error saving frame: {str(e)}")
            return False
    
    @staticmethod
    def load_frame(frame_path: str) -> Optional[np.ndarray]:
        """
        Load a frame from disk.
        
        Args:
            frame_path: Path to frame file
            
        Returns:
            Loaded frame or None if failed
        """
        try:
            if not Path(frame_path).exists():
                logging.error(f"Frame file not found: {frame_path}")
                return None
            
            frame = cv2.imread(frame_path)
            
            if frame is None:
                logging.error(f"Could not load frame from {frame_path}")
                return None
            
            return frame
            
        except Exception as e:
            logging.error(f"Error loading frame: {str(e)}")
            return None
    
    @staticmethod
    def frames_to_video(frames: List[np.ndarray], output_path: str, fps: int = 30) -> bool:
        """
        Convert a list of frames to a video file.
        
        Args:
            frames: List of frames
            output_path: Output video path
            fps: Frames per second
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not frames:
                logging.error("No frames provided")
                return False
            
            # Get frame dimensions
            h, w = frames[0].shape[:2]
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            
            if not out.isOpened():
                logging.error(f"Could not create video writer for {output_path}")
                return False
            
            # Write frames
            for frame in frames:
                out.write(frame)
            
            # Release video writer
            out.release()
            
            logging.info(f"Video saved to {output_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error creating video: {str(e)}")
            return False
    
    @staticmethod
    def analyze_frame_sequence(frames: List[np.ndarray]) -> Dict[str, any]:
        """
        Analyze a sequence of frames and return statistics.
        
        Args:
            frames: List of frames
            
        Returns:
            Dictionary containing analysis results
        """
        if not frames:
            return {}
        
        try:
            # Basic statistics
            num_frames = len(frames)
            frame_shapes = [frame.shape for frame in frames]
            
            # Calculate frame differences
            frame_diffs = []
            for i in range(1, num_frames):
                diff = FrameUtils.calculate_frame_difference(frames[i-1], frames[i])
                frame_diffs.append(diff)
            
            # Calculate frame complexities
            complexities = [FrameUtils.calculate_frame_complexity(frame) for frame in frames]
            
            # Detect scene changes
            scene_changes = FrameUtils.detect_scene_changes(frames)
            
            return {
                "num_frames": num_frames,
                "frame_shapes": frame_shapes,
                "mean_frame_difference": np.mean(frame_diffs) if frame_diffs else 0.0,
                "std_frame_difference": np.std(frame_diffs) if frame_diffs else 0.0,
                "mean_complexity": np.mean(complexities),
                "std_complexity": np.std(complexities),
                "scene_changes": scene_changes,
                "num_scene_changes": len(scene_changes)
            }
            
        except Exception as e:
            logging.error(f"Error analyzing frame sequence: {str(e)}")
            return {}


