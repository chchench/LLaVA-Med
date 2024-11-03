
# Update pip
pip install --upgrade pip

# Install PyTorch with CUDA support
##### pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Hugging Face Transformers and Datasets
pip install datasets

# Install additional dependencies
pip install accelerate bitsandbytes sentencepiece
pip install Pillow numpy tqdm

# Install DeepSpeed (optional for optimization and distributed training)
pip install deepspeed

# Replace '11.3' with your CUDA version
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# Install other packages if available
conda install transformers

pip install safetensors
pip install accelerate


