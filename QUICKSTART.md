# Quick Start Guide

## Video-Context Image Captioning with Liquid AI

This guide will help you get started with the video-context image captioning system in just a few minutes.

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- Conda (recommended) or pip

## Installation

### Option 1: Using the installation script (Recommended)

```bash
cd /home/frank/VideoContextImageCaptioning
./install.sh
```

### Option 2: Manual installation

```bash
# Create conda environment
conda create -n video-caption python=3.10
conda activate video-caption

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

## Quick Test

1. **Activate the environment:**
   ```bash
   conda activate video-caption
   ```

2. **Run the example:**
   ```bash
   python examples/run_example.py
   ```

3. **Run tests:**
   ```bash
   python -m pytest tests/
   ```

## Basic Usage

```python
from src.pipeline import VideoContextCaptionPipeline

# Initialize pipeline
pipeline = VideoContextCaptionPipeline(
    model_path="path/to/your/lfm2-vl-model",
    device="cuda"
)

# Generate caption with video context
result = pipeline.generate_caption(
    image_path="path/to/image.jpg",
    video_path="path/to/video.mp4",
    context_weight=0.7
)

print(f"Caption: {result['caption']}")
print(f"Context: {result['video_context']['context_text']}")
```

## Configuration

Edit `config/model_config.yaml` to customize:

- Model path
- Frame extraction strategy
- Context aggregation method
- Processing parameters

## Examples

The `examples/` directory contains:

- `run_example.py` - Complete example with all features
- Sample image and video files (created automatically)

## Testing

Run the test suite:

```bash
python -m pytest tests/ -v
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory:**
   - Reduce `max_frames` in pipeline initialization
   - Use `device="cpu"` for CPU-only processing

2. **Model not found:**
   - Update the model path in your code
   - The system will use a mock model for testing

3. **Video processing errors:**
   - Ensure video file is valid and accessible
   - Check OpenCV installation

### Getting Help

- Check the logs for detailed error messages
- Run tests to verify installation
- Review the example code for usage patterns

## Next Steps

1. **Customize the pipeline** for your specific use case
2. **Train your own LFM2-VL model** for better results
3. **Experiment with different strategies** for frame extraction and context aggregation
4. **Integrate with your existing workflow**

## Architecture Overview

The system works in two steps:

1. **Video Context Extraction:**
   - Extract key frames from video
   - Generate captions for each frame
   - Aggregate context using LNN's state space capabilities

2. **Context-Aware Image Captioning:**
   - Process target image
   - Integrate video context using LNN's adaptive state
   - Generate personalized caption

## Key Features

- **Liquid Neural Network Integration:** Leverages LFM2-VL's continuous-time processing
- **Multiple Frame Extraction Strategies:** Key frames, uniform sampling, adaptive
- **Flexible Context Aggregation:** Weighted average, attention, concatenation
- **Quality Analysis:** Comprehensive caption quality metrics
- **Batch Processing:** Process multiple image-video pairs efficiently

## Performance Tips

- Use GPU for faster processing
- Adjust `max_frames` based on video length
- Choose appropriate frame extraction strategy
- Monitor memory usage for large videos

Happy captioning! 🎥📝


