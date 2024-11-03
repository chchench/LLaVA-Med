import argparse
import datetime
import json
import os
import re
import time
import requests

from PIL import Image
from io import BytesIO
import base64
from llava.conversation import (default_conversation, conv_templates, SeparatorStyle)
from llava.constants import LOGDIR
from llava.utils import (build_logger, server_error_msg, violates_moderation, moderation_msg)
import hashlib


logger = build_logger("cli_interface", "cli_interface.log")
headers = {"User-Agent": "LLaVA-Med Client"}


priority = {
    "vicuna-13b": "aaaaaaa",
    "koala-13b": "aaaaaab",
}


def abbreviate_string(s, max_length, suffix="..."):
    if len(s) > max_length:
        return s[:max_length - len(suffix)] + suffix
    return s


def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name


def get_model_list(controller_url):
    response = requests.post(controller_url + "/list_models")
    response.raise_for_status()
    models = response.json().get("models", [])
    models.sort(key=lambda x: priority.get(x, x))
    return models


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        img = Image.open(image_file)
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def submit_prompt(prompt, model_name, temperature, top_p, max_tokens, controller_url, image_path=None):

    print('Prompt data')
    print(f'  controller_url:  {controller_url}')
    print(f'  model_name:      {model_name}')
    print(f'  temperature:     {temperature}')
    print(f'  top_p:           {top_p}')
    print(f'  max_tokens:      {max_tokens}')
    print(f'  prompt:          {prompt}')
    print(f'  image_path:      {image_path}')
    print('')

    try:

        # pload = {'model': 'llava-med-v1.5-mistral-7b', 
        # 'prompt': '[INST] <image>\nIs there evidence of an aortic aneurysm? Please choose from the following two options: [yes, no]? [/INST]',
        # 'temperature': 0.2, 'top_p': 0.7, 'max_new_tokens': 512, 'stop': '</s>', 
        # 'images': ['Base64 encoding']}

        prompt_data = {
            "model": model_name,
            "prompt": f'[INST] {prompt} [/INST]',
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_tokens,
            "stop": "</s>",
        }

        # Encode and add the image if provided
        if image_path:
            image_data = encode_image(image_path)

            print(f'Encoded data:               {abbreviate_string(image_data, 20)}')

            prompt_data["prompt"] = f'[INST] <image>\n{prompt} [/INST]'
            prompt_data["images"] = [image_data]  # Assuming API accepts a list of images in base64 format

        complete_controller_url = controller_url + "/get_worker_address"
        print(f'Complete controller URL:    {complete_controller_url}')

        response = requests.post(complete_controller_url, json=prompt_data)

        worker_address = response.json()["address"]
        print(f'Responding worker address:  {worker_address}')

        if worker_address:
            complete_worker_url = worker_address + "/worker_generate_stream"
            print(f'Complete worker URL:        {complete_worker_url}')
            print('')

            response = requests.post(complete_worker_url, headers=headers, json=prompt_data)

            output = ""
            for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
                if chunk:
                    data = json.loads(chunk.decode())
                    if data.get("error_code", 0) == 0:
                        output = data.get("text", "")
                    else:
                        return f"Error: {data.get('text')} (error_code: {data['error_code']})"
            return output
    
    except requests.RequestException as e:
        return f"Error: {str(e)}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-url", type=str, default="http://localhost:10000")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()

    # Fetch available models
    models = get_model_list(args.controller_url)
    if not models:
        print("No models available.")
        return
    
    num_models = len(models)
    if num_models == 0:
        print("No valid model is available.")
        return
    elif num_models == 1:
        model_index = 0
    else:
        print("Available models:")
        for i, model in enumerate(models):
            print(f"{i + 1}. {model}")

        model_index = int(input("Select model by number: ")) - 1
        if model_index < 0 or model_index >= len(models):
            print("Invalid model selection.")
            return

    model_name = models[model_index]

    print(f"Using model: {model_name}")
    while True:
        prompt = input("Enter your prompt (or type 'exit' to quit): ")
        if prompt.lower() == 'exit':
            break

        image_path = input("Enter the path to an image file to analyze (or press Enter to skip): ").strip()
        if image_path == "":
            image_path = None

        response = submit_prompt(
            prompt,
            model_name,
            args.temperature,
            args.top_p,
            args.max_tokens,
            args.controller_url,
            image_path
        )

        # Extract the content between the <INST> and </INST> tags
        content_match = re.search(r'\[INST\](.*?)\[/INST\]', response, re.DOTALL)
        inst_text = content_match.group(1).strip() if content_match else ""

        # Remove <..> content in the original instruction
        tags_in_inst_text = re.findall(r'<(.*?)>', inst_text)
        cleaned_inst_text = re.sub(r'<.*?>', '', inst_text).strip()

        # Extract text after [/INST]
        after_inst_content = re.search(r'\[/INST\](.*)', response, re.DOTALL)
        after_inst_text = after_inst_content.group(1).strip() if after_inst_content else ""

        print(f'Response for prompt "{cleaned_inst_text}":')
        print(f'\n{after_inst_text}\n')


if __name__ == "__main__":
    main()
