# model.py
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, CLIPProcessor, CLIPModel

class LLaVAMedModel(nn.Module):
    def __init__(self, vision_model_name, language_model_name, device='cuda'):
        super(LLaVAMedModel, self).__init__()
        self.device = device

        # Load CLIP Vision Encoder
        self.clip_processor = CLIPProcessor.from_pretrained(vision_model_name)
        self.clip_model = CLIPModel.from_pretrained(vision_model_name).vision_model
        for param in self.clip_model.parameters():
            param.requires_grad = False  # Freeze vision encoder

        # Load Mistral-7B Language Model
        self.tokenizer = AutoTokenizer.from_pretrained(language_model_name)
        self.language_model = AutoModelForCausalLM.from_pretrained(
            language_model_name,
            torch_dtype=torch.float16,
            device_map='auto'
        )

        # Projection layer to map CLIP embeddings to language model's hidden size
        self.proj = nn.Linear(self.clip_model.config.hidden_size, self.language_model.config.hidden_size).to(self.device)

    def forward(self, images, texts):
        # Process images through CLIP
        image_inputs = self.clip_processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.clip_model(**image_inputs).pooler_output  # Shape: (N, hidden_size)

        # Project image features to language model's hidden size
        image_embeddings = self.proj(image_features)  # Shape: (N, lm_hidden_size)

        # Tokenize texts
        text_inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)

        # Generate text conditioned on image embeddings
        # Here, we can prepend image embeddings to the language model's input or use another mechanism
        # This is a simplified example where image embeddings are used as additional tokens

        # For demonstration, assume image embeddings are used as a prefix
        # You might need a more sophisticated integration method

        # Encode text
        outputs = self.language_model(
            input_ids=text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask,
            past_key_values=image_embeddings.unsqueeze(1)  # Adjust as per model's expectation
        )

        return outputs.logits
    