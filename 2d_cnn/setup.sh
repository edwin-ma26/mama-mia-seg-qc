#!/bin/bash

# Setup script for 2D CNN training

echo "Setting up 2D CNN training environment..."

# Create output directory
mkdir -p output

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Check if CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if [ $? -eq 0 ]; then
    echo "PyTorch installation successful"
else
    echo "Warning: PyTorch installation may have failed"
fi

echo "Setup complete!" 