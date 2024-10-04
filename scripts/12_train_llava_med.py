import os
import torch
from torch.utils.data import DataLoader
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

# Assume data_loader.py and model configuration have been set up as above
from data_loader import load_medical_dataset

def main():
    # Load dataset
    dataset = load_medical_dataset(
        image_dir='/path/to/images',
        annotations_file='/path/to/annotations.csv'
    )

    # Split dataset
    dataset = dataset['train'].train_test_split(test_size=0.1)
    train_dataset = dataset['train']
    eval_dataset = dataset['test']

    # Preprocessing function
    def preprocess_function(examples):
        # Process images
        images = [vision_processor(images=img, return_tensors='pt')['pixel_values'][0] for img in examples['image']]
        examples['pixel_values'] = images

        # Process text
        inputs = tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        examples['input_ids'] = inputs.input_ids
        examples['attention_mask'] = inputs.attention_mask
        return examples

    # Apply preprocessing
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=eval_dataset.column_names
    )

    # Data collator
    from transformers import DataCollatorForSeq2Seq
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir='./llava_med_mistral7b',
        num_train_epochs=3,
        per_device_train_batch_size=1,   # Adjust based on GPU memory
        per_device_eval_batch_size=1,
        fp16=True,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_steps=100,
        learning_rate=5e-5,
        predict_with_generate=True,
        gradient_accumulation_steps=4,   # Adjust as needed
        deepspeed='ds_config.json',      # Optional: DeepSpeed config file
        report_to='none'
    )

    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    # Start training
    trainer.train()

    # Save the model
    trainer.save_model('./llava_med_mistral7b')

if __name__ == '__main__':
    main()

