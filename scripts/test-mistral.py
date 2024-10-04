from transformers import AutoConfig, AutoModelForCausalLM, \
    MistralConfig, MistralModel, MistralForCausalLM

# Specify the model name or path
model_name = "mistralai/Mistral-7B-v0.3"

# Using Auto Classes
config = AutoConfig.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Alternatively, using Mistral-specific Classes
config = MistralConfig.from_pretrained(model_name)
model = MistralForCausalLM.from_pretrained(model_name)

# Prepare input text
input_text = "Once upon a time"
inputs = tokenizer(input_text, return_tensor="pt")

# Generate text
outputs = model.generate(**inputs, max_length=50, num_return_sequences=1)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_text)
