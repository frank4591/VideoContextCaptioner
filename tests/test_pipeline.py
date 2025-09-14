"""
Unit tests for the video-context image captioning pipeline.

This module contains comprehensive tests for all components
of the video-context image captioning system.
"""

import unittest
import sys
import os
import tempfile
import numpy as np
import cv2
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from pipeline import VideoContextCaptionPipeline
from video_processor import VideoProcessor
from context_extractor import ContextExtractor
from image_captioner import ImageCaptioner
from liquid_integration import LiquidAIIntegration

class TestVideoProcessor(unittest.TestCase):
    """Test cases for VideoProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.video_processor = VideoProcessor(
            strategy="key_frames",
            max_frames=5
        )
        
        # Create a temporary video file for testing
        self.temp_video = self._create_temp_video()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_video):
            os.remove(self.temp_video)
    
    def _create_temp_video(self):
        """Create a temporary video file for testing."""
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        temp_file.close()
        
        # Create a simple video with 30 frames
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_file.name, fourcc, 30.0, (224, 224))
        
        for i in range(30):
            # Create frames with slight variations
            frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            out.write(frame)
        
        out.release()
        return temp_file.name
    
    def test_video_processor_initialization(self):
        """Test VideoProcessor initialization."""
        self.assertEqual(self.video_processor.strategy, "key_frames")
        self.assertEqual(self.video_processor.max_frames, 5)
    
    def test_extract_frames(self):
        """Test frame extraction."""
        frames = self.video_processor.extract_frames(self.temp_video)
        
        self.assertIsInstance(frames, list)
        self.assertGreater(len(frames), 0)
        self.assertLessEqual(len(frames), 5)  # Should not exceed max_frames
        
        # Check frame format
        for frame in frames:
            self.assertIsInstance(frame, np.ndarray)
            self.assertEqual(len(frame.shape), 3)  # Should be 3D (H, W, C)
    
    def test_get_video_info(self):
        """Test video information extraction."""
        info = self.video_processor.get_video_info(self.temp_video)
        
        self.assertIn('fps', info)
        self.assertIn('frame_count', info)
        self.assertIn('width', info)
        self.assertIn('height', info)
        self.assertIn('duration', info)
        
        self.assertGreater(info['fps'], 0)
        self.assertGreater(info['frame_count'], 0)
        self.assertEqual(info['width'], 224)
        self.assertEqual(info['height'], 224)
    
    def test_invalid_video_path(self):
        """Test handling of invalid video path."""
        with self.assertRaises(FileNotFoundError):
            self.video_processor.extract_frames("nonexistent_video.mp4")

class TestContextExtractor(unittest.TestCase):
    """Test cases for ContextExtractor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.context_extractor = ContextExtractor(
            aggregation_method="weighted_average"
        )
        
        # Sample frame captions and features
        self.frame_captions = [
            "A person walking in the park",
            "A dog running on the grass",
            "Children playing in the playground"
        ]
        
        self.frame_features = [
            np.random.randn(768),
            np.random.randn(768),
            np.random.randn(768)
        ]
    
    def test_context_extractor_initialization(self):
        """Test ContextExtractor initialization."""
        self.assertEqual(self.context_extractor.aggregation_method, "weighted_average")
        self.assertEqual(self.context_extractor.similarity_threshold, 0.7)
    
    def test_aggregate_context_weighted_average(self):
        """Test context aggregation with weighted average method."""
        result = self.context_extractor.aggregate_context(
            frame_captions=self.frame_captions,
            frame_features=self.frame_features,
            max_length=100
        )
        
        self.assertIn('text', result)
        self.assertIn('features', result)
        self.assertIn('temporal_consistency', result)
        
        self.assertIsInstance(result['text'], str)
        self.assertIsInstance(result['features'], np.ndarray)
        self.assertIsInstance(result['temporal_consistency'], float)
    
    def test_aggregate_context_concatenation(self):
        """Test context aggregation with concatenation method."""
        extractor = ContextExtractor(aggregation_method="concatenation")
        
        result = extractor.aggregate_context(
            frame_captions=self.frame_captions,
            frame_features=self.frame_features,
            max_length=100
        )
        
        self.assertIn('text', result)
        self.assertIn('features', result)
        self.assertEqual(result['temporal_consistency'], 1.0)
    
    def test_empty_captions(self):
        """Test handling of empty captions."""
        result = self.context_extractor.aggregate_context(
            frame_captions=[],
            frame_features=[],
            max_length=100
        )
        
        self.assertEqual(result['text'], "")
        self.assertEqual(len(result['features']), 0)
        self.assertEqual(result['temporal_consistency'], 0.0)
    
    def test_extract_key_phrases(self):
        """Test key phrase extraction."""
        context_text = "A person walking in the park with a dog running on the grass"
        key_phrases = self.context_extractor.extract_key_phrases(context_text, max_phrases=3)
        
        self.assertIsInstance(key_phrases, list)
        self.assertLessEqual(len(key_phrases), 3)

class TestLiquidAIIntegration(unittest.TestCase):
    """Test cases for LiquidAIIntegration class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.liquid_integration = LiquidAIIntegration(
            model_path="mock_model_path",
            device="cpu"
        )
        
        # Create a sample image
        self.sample_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    def test_liquid_integration_initialization(self):
        """Test LiquidAIIntegration initialization."""
        self.assertEqual(self.liquid_integration.device, "cpu")
        self.assertEqual(self.liquid_integration.max_length, 512)
        self.assertEqual(self.liquid_integration.model_path, "mock_model_path")
    
    def test_generate_caption(self):
        """Test caption generation."""
        result = self.liquid_integration.generate_caption(
            image=self.sample_image,
            return_features=True
        )
        
        self.assertIn('caption', result)
        self.assertIn('features', result)
        self.assertIn('confidence', result)
        self.assertIn('processing_time', result)
        
        self.assertIsInstance(result['caption'], str)
        self.assertIsInstance(result['features'], np.ndarray)
        self.assertIsInstance(result['confidence'], float)
    
    def test_generate_with_context(self):
        """Test caption generation with context."""
        context = {
            "context_text": "A person walking in the park",
            "context_features": np.random.randn(768)
        }
        
        result = self.liquid_integration.generate_with_context(
            image=self.sample_image,
            context=context,
            context_weight=0.7
        )
        
        self.assertIn('caption', result)
        self.assertIn('context_relevance', result)
        self.assertIn('confidence', result)
        self.assertIn('processing_time', result)
    
    def test_get_model_info(self):
        """Test model information retrieval."""
        info = self.liquid_integration.get_model_info()
        
        self.assertIn('model_path', info)
        self.assertIn('device', info)
        self.assertIn('max_length', info)
        self.assertIn('is_mock', info)

class TestImageCaptioner(unittest.TestCase):
    """Test cases for ImageCaptioner class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.liquid_integration = LiquidAIIntegration(
            model_path="mock_model_path",
            device="cpu"
        )
        
        self.image_captioner = ImageCaptioner(
            liquid_integration=self.liquid_integration,
            context_weight=0.7
        )
        
        # Create temporary image file
        self.temp_image = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        self.temp_image.close()
        
        sample_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        cv2.imwrite(self.temp_image.name, sample_image)
        
        self.video_context = {
            "context_text": "A person walking in the park with a dog",
            "context_features": np.random.randn(768),
            "frames_processed": 5,
            "temporal_consistency": 0.8
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_image.name):
            os.remove(self.temp_image.name)
    
    def test_image_captioner_initialization(self):
        """Test ImageCaptioner initialization."""
        self.assertEqual(self.image_captioner.context_weight, 0.7)
        self.assertIsNotNone(self.image_captioner.liquid_integration)
    
    def test_generate_caption_with_context(self):
        """Test caption generation with context."""
        result = self.image_captioner.generate_caption(
            image_path=self.temp_image.name,
            video_context=self.video_context
        )
        
        self.assertIn('caption', result)
        self.assertIn('confidence', result)
        self.assertIn('context_relevance', result)
        self.assertIn('processing_time', result)
        self.assertIn('context_weight_used', result)
    
    def test_generate_caption_without_context(self):
        """Test caption generation without context."""
        result = self.image_captioner.generate_caption_without_context(
            image_path=self.temp_image.name
        )
        
        self.assertIn('caption', result)
        self.assertIn('confidence', result)
        self.assertIn('processing_time', result)
        self.assertEqual(result['context_weight_used'], 0.0)
    
    def test_analyze_caption_quality(self):
        """Test caption quality analysis."""
        caption = "A person walking in the park with a dog"
        
        quality = self.image_captioner.analyze_caption_quality(
            caption, self.video_context
        )
        
        self.assertIn('word_count', quality)
        self.assertIn('diversity_ratio', quality)
        self.assertIn('context_relevance', quality)
        self.assertIn('quality_score', quality)
        
        self.assertGreater(quality['word_count'], 0)
        self.assertGreaterEqual(quality['diversity_ratio'], 0.0)
        self.assertLessEqual(quality['diversity_ratio'], 1.0)

class TestVideoContextCaptionPipeline(unittest.TestCase):
    """Test cases for VideoContextCaptionPipeline class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.pipeline = VideoContextCaptionPipeline(
            model_path="mock_model_path",
            device="cpu",
            frame_extraction_strategy="key_frames",
            max_frames=5
        )
        
        # Create temporary files
        self.temp_image = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        self.temp_image.close()
        
        self.temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        self.temp_video.close()
        
        # Create sample files
        sample_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        cv2.imwrite(self.temp_image.name, sample_image)
        
        # Create sample video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.temp_video.name, fourcc, 30.0, (224, 224))
        for i in range(30):
            frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            out.write(frame)
        out.release()
    
    def tearDown(self):
        """Clean up test fixtures."""
        for temp_file in [self.temp_image.name, self.temp_video.name]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        self.assertEqual(self.pipeline.device, "cpu")
        self.assertEqual(self.pipeline.max_frames, 5)
        self.assertEqual(self.pipeline.context_weight, 0.7)
    
    def test_generate_caption(self):
        """Test caption generation with video context."""
        result = self.pipeline.generate_caption(
            image_path=self.temp_image.name,
            video_path=self.temp_video.name
        )
        
        self.assertIn('caption', result)
        self.assertIn('video_context', result)
        self.assertIn('confidence', result)
        self.assertIn('processing_time', result)
        self.assertIn('frames_processed', result)
        self.assertIn('context_relevance', result)
    
    def test_batch_process(self):
        """Test batch processing."""
        image_video_pairs = [
            (self.temp_image.name, self.temp_video.name),
            (self.temp_image.name, self.temp_video.name)
        ]
        
        results = self.pipeline.batch_process(image_video_pairs)
        
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)
        
        for result in results:
            self.assertIn('image_path', result)
            self.assertIn('video_path', result)
    
    def test_get_pipeline_info(self):
        """Test pipeline information retrieval."""
        info = self.pipeline.get_pipeline_info()
        
        self.assertIn('device', info)
        self.assertIn('max_frames', info)
        self.assertIn('context_weight', info)
        self.assertIn('frame_extraction_strategy', info)
        self.assertIn('context_aggregation', info)
        self.assertIn('model_path', info)

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.pipeline = VideoContextCaptionPipeline(
            model_path="mock_model_path",
            device="cpu"
        )
        
        # Create temporary files
        self.temp_image = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        self.temp_image.close()
        
        self.temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        self.temp_video.close()
        
        # Create sample files
        sample_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        cv2.imwrite(self.temp_image.name, sample_image)
        
        # Create sample video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.temp_video.name, fourcc, 30.0, (224, 224))
        for i in range(30):
            frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            out.write(frame)
        out.release()
    
    def tearDown(self):
        """Clean up test fixtures."""
        for temp_file in [self.temp_image.name, self.temp_video.name]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline."""
        try:
            result = self.pipeline.generate_caption(
                image_path=self.temp_image.name,
                video_path=self.temp_video.name,
                context_weight=0.8
            )
            
            # Verify all expected keys are present
            expected_keys = [
                'caption', 'video_context', 'confidence', 
                'processing_time', 'frames_processed', 'context_relevance'
            ]
            
            for key in expected_keys:
                self.assertIn(key, result)
            
            # Verify data types
            self.assertIsInstance(result['caption'], str)
            self.assertIsInstance(result['video_context'], dict)
            self.assertIsInstance(result['confidence'], float)
            self.assertIsInstance(result['processing_time'], float)
            self.assertIsInstance(result['frames_processed'], int)
            self.assertIsInstance(result['context_relevance'], float)
            
        except Exception as e:
            self.fail(f"End-to-end pipeline test failed: {str(e)}")

def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestVideoProcessor,
        TestContextExtractor,
        TestLiquidAIIntegration,
        TestImageCaptioner,
        TestVideoContextCaptionPipeline,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
