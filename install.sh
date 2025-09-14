#!/bin/bash

# Installation script for Video-Context Image Captioning with Liquid AI

echo "Installing Video-Context Image Captioning with Liquid AI..."
echo "=========================================================="

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "Conda found. Creating environment..."
    
    # Create conda environment
    conda create -n video-caption python=3.10 -y
    
    # Activate environment
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate video-caption
    
    echo "Conda environment 'video-caption' created and activated."
else
    echo "Conda not found. Using pip with system Python..."
fi

# Install requirements
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Install the package in development mode
echo "Installing package in development mode..."
pip install -e .

echo ""
echo "Installation completed successfully!"
echo ""
echo "To use the package:"
echo "1. Activate the conda environment: conda activate video-caption"
echo "2. Run the example: python examples/run_example.py"
echo "3. Run tests: python -m pytest tests/"
echo ""
echo "Note: You'll need to provide a valid LFM2-VL model path in the examples."


