import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

# Usage
#
# python train_gemini.py                   \
#     <path of downloaded Llama-Med model> \
#     <training output dir>                \
#     <output path of updated model>       \
#     <optional # of epochs>

# Load the LLaMa-Med model and tokenizer

model_name = sys.argv[1]  # Replace with the actual path
print(f'Loading model {model_name}')

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define training arguments

training_output_dir = sys.argv[2]

if len(sys.argv) > 4:
  num_epochs = sys.argv[4]
else:
  num_epochs = 10

training_args = TrainingArguments(
    output_dir=training_output_dir,
    overwrite_output_dir=True,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    save_steps=1000,
    evaluation_strategy="steps",
    logging_steps=1000,
)

# Create a Trainer instance

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # Optional, if you have an evaluation dataset
)

# Start the training process

trainer.train()

# Save the trained model

trained_model_output_path = sys.argv[3]
print(f'Updated model will be written out to {trained_model_output_path}')

trainer.save_model(trained_model_output_path)






