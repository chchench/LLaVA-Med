# data_loader.py

import os
import pandas as pd
from datasets import Dataset, DatasetDict

def load_medical_dataset(image_dir, annotations_file):
    # Load annotations
    annotations = pd.read_csv(annotations_file)
    
    # Verify necessary columns
    required_columns = ['image_filename', 'text']  # Adjust based on your CSV
    for col in required_columns:
        if col not in annotations.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Create full image paths
    annotations['image_filename'] = annotations['image_filename'].apply(lambda x: os.path.join(image_dir, x))
    
    # Verify image paths exist
    if not all(annotations['image_filename'].apply(os.path.exists)):
        missing = annotations[~annotations['image_filename'].apply(os.path.exists)]
        raise FileNotFoundError(f"Missing images: {missing['image_filename'].tolist()}")
    
    # Convert to Hugging Face Dataset
    dataset = Dataset.from_pandas(annotations)
    
    # Split into train and test
    dataset = dataset.train_test_split(test_size=0.1)
    
    return DatasetDict({
        'train': dataset['train'],
        'test': dataset['test']
    })