# data_loader.py
from datasets import load_dataset
from PIL import Image
import torch

def load_medical_dataset(image_dir, annotations_file):
    # Load dataset from CSV or JSON
    dataset = load_dataset(
        'csv',
        data_files=annotations_file,
        cache_dir='./cache'
    )

    # Function to load images
    def load_image(example):
        image_path = f"{image_dir}/{example['image_filename']}"
        image = Image.open(image_path).convert('RGB')
        example['image'] = image
        return example

    dataset = dataset.map(load_image)
    return dataset
