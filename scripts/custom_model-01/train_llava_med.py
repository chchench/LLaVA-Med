# train_llava_med.py
import os
import torch
from torch.utils.data import DataLoader
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from data_loader import load_medical_dataset
from model import LLaVAMedModel
from transformers import DataCollatorWithPadding

def main():
    # Configuration
    vision_model_name = 'openai/clip-vit-base-patch32'
    language_model_name = 'mistralai/Mistral-7B-v0.1'
    image_dir = '/path/to/images'
    annotations_file = '/path/to/annotations.csv'
    batch_size = 4  # Adjust based on GPU memory
    num_epochs = 3
    learning_rate = 5e-5

    # Initialize model
    model = LLaVAMedModel(vision_model_name, language_model_name).to('cuda')

    # Load dataset
    dataset = load_medical_dataset(image_dir, annotations_file)

    # Split dataset
    dataset = dataset['train'].train_test_split(test_size=0.1)
    train_dataset = dataset['train']
    eval_dataset = dataset['test']

    # Preprocessing function
    def preprocess_function(examples):
        # Images are already loaded in the dataset
        return examples

    # Apply preprocessing
    train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(preprocess_function, batched=True, remove_columns=eval_dataset.column_names)

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=model.tokenizer)

    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir='./llava_med_mistral7b',
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        fp16=True,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='loss',
        predict_with_generate=False,
        deepspeed='ds_config.json'  # Optional
    )

    # Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=model.tokenizer,
        data_collator=data_collator
    )

    # Start training
    trainer.train()

    # Save the final model
    trainer.save_model('./llava_med_mistral7b_final')

if __name__ == '__main__':
    main()
    