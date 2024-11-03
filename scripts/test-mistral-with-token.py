from transformers import AutoTokenizer, AutoModelForCausalLM
import os

model_name = "mistralai/Mistral-7B-v0.3"
access_token = os.environ['HUGGINGFACE_HUB_TOKEN']
print(f'Access token:  {access_token}')

# Load the tokenizer and model with authentication
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,
    use_auth_token=access_token)

exit()

model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True,
    use_auth_token=access_tokenn)

# Prepare input text
input_text = "Once upon a time"
inputs = tokenizer(input_text, return_tensors="pt")

# Generate text
outputs = model.generate(**inputs, max_length=50)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_text)
