#!/bin/bash

# Update pip
pip install --upgrade pip

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Hugging Face Transformers and Datasets
pip install transformers datasets

# Install additional dependencies
pip install accelerate bitsandbytes sentencepiece
pip install Pillow numpy tqdm

# Install DeepSpeed (optional for optimization and distributed training)
pip install deepspeed
