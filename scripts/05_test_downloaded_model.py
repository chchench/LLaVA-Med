from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import torch

model_id = sys.argv[1]

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map='auto',
    trust_remote_code=True
)

print('Successfully loaded the model')
