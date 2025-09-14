# Video-Context Image Captioning with Liquid AI

A two-step project that leverages Liquid AI's LFM2-VL models to generate personalized image captions using video context. This project works around the LFM library's video input limitation by extracting temporal context from video frames and using it to enhance image caption generation.

## Architecture Overview

### Key Innovation
While LFM2-VL doesn't directly support video input, we leverage its **continuous-time processing capabilities** and **Liquid Neural Network (LNN) architecture** to:

1. **Extract temporal context** from video frames using LNN's sequential processing
2. **Maintain adaptive state** across video frame analysis
3. **Apply video context** to enhance image caption generation

### Two-Step Process

#### Step 1: Video Context Extraction
- Extract key frames from input video
- Process frames through LFM2-VL to generate preliminary captions
- Aggregate captions to form comprehensive video context
- Leverage LNN's continuous-time processing for temporal understanding

#### Step 2: Context-Aware Image Captioning
- Input target image into LFM2-VL
- Incorporate extracted video context as additional conditioning
- Generate personalized caption that reflects both image content and video context
- Utilize LNN's adaptive state for context integration

## Technical Approach

### Liquid Neural Network Advantages
- **Continuous-time processing**: Natural fit for video frame sequences
- **State space models**: Maintain context across sequential inputs
- **Adaptive learning**: Model state adapts based on new data
- **Computational efficiency**: Optimized for on-device processing

### Workaround Strategy
1. **Frame Sampling**: Extract representative frames from video
2. **Sequential Processing**: Use LNN's temporal processing capabilities
3. **Context Aggregation**: Combine frame-level insights into video context
4. **Conditional Generation**: Apply video context to image captioning

## Installation

```bash
# Create conda environment
conda create -n video-caption python=3.10
conda activate video-caption

# Install dependencies
pip install -r requirements.txt

# Install Liquid AI (if not already installed)
pip install liquid-ai
```

## Usage

### Basic Usage
```python
from src.pipeline import VideoContextCaptionPipeline

# Initialize pipeline
pipeline = VideoContextCaptionPipeline(
    model_path="path/to/lfm2-vl-model",
    device="cuda"
)

# Generate caption with video context
caption = pipeline.generate_caption(
    image_path="path/to/image.jpg",
    video_path="path/to/video.mp4",
    context_weight=0.7
)
```

### Advanced Usage
```python
# Custom frame extraction
pipeline = VideoContextCaptionPipeline(
    model_path="path/to/lfm2-vl-model",
    frame_extraction_strategy="key_frames",  # or "uniform", "adaptive"
    max_frames=10,
    context_aggregation="weighted_average"  # or "attention", "concatenation"
)

# Generate with custom parameters
caption = pipeline.generate_caption(
    image_path="path/to/image.jpg",
    video_path="path/to/video.mp4",
    context_weight=0.8,
    max_context_length=512
)
```

## Project Structure

```
VideoContextImageCaptioning/
├── README.md
├── requirements.txt
├── config/
│   └── model_config.yaml
├── src/
│   ├── __init__.py
│   ├── video_processor.py      # Video frame extraction
│   ├── context_extractor.py    # Context extraction from frames
│   ├── image_captioner.py      # Image captioning with context
│   ├── liquid_integration.py   # LFM2-VL integration
│   └── pipeline.py             # Main pipeline
├── utils/
│   ├── __init__.py
│   ├── frame_utils.py          # Video utilities
│   └── text_utils.py           # Text processing
├── examples/
│   └── run_example.py          # Example usage
└── tests/
    └── test_pipeline.py        # Unit tests
```

## Features

- **Video Context Extraction**: Intelligent frame sampling and context aggregation
- **Liquid AI Integration**: Leverages LFM2-VL's continuous-time processing
- **Adaptive Context Weighting**: Dynamic context importance based on relevance
- **Memory Efficient**: Optimized for large video files
- **Configurable Pipeline**: Flexible parameters for different use cases

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended)
- Liquid AI LFM2-VL model
- OpenCV for video processing
- Transformers for text processing

## Limitations

- Requires pre-trained LFM2-VL model
- Video processing adds computational overhead
- Context quality depends on video content relevance
- Limited by LFM2-VL's current video input constraints

## Future Enhancements

- Real-time video processing
- Multi-modal context fusion
- Advanced temporal modeling
- Custom LNN architectures for video understanding
