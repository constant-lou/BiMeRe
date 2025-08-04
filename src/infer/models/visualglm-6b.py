import torch
from transformers import AutoModel, AutoTokenizer
import os
import sys
from torch.cuda.amp import autocast

model_directory = '/path/to/visualglm-6b'
if os.path.isdir(model_directory) and model_directory not in sys.path:
    print(f"Adding model directory to Python search path: {model_directory}")
    sys.path.insert(0, model_directory)

MAX_NEW_TOKEN = 1024

def load_model(model_name, model_args, use_accel=False):
    model_path = model_args.get('model_path_or_name', model_directory)
    model_components = {}
    
    print(f"Loading tokenizer from '{model_path}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model_components['tokenizer'] = tokenizer

    print(f"Loading model from '{model_path}' (half precision mode)...")
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map='auto',
        torch_dtype=torch.float16
    ).eval()
    
    model_components['model'] = model
    model_components['model_name'] = model_name
    
    print("Model and tokenizer loading completed.")
    return model_components

def infer(prompts, **kwargs):
    model = kwargs.get('model')
    tokenizer = kwargs.get('tokenizer')
    
    if not model or not tokenizer:
        raise ValueError("Model and tokenizer must be provided.")

    responses = []

    for i, prompt_item in enumerate(prompts):
        print(f"Processing sample {i+1}/{len(prompts)}...")
        
        image_path = None
        prompt_text = ""
        history = []

        if isinstance(prompt_item, dict) and "conversations" in prompt_item:
            conversations = prompt_item["conversations"]
            for conv in conversations[:-1]:
                if 'prompt' in conv and 'response' in conv:
                    history.append((conv['prompt'], conv['response']))
            test_conv = conversations[-1]
            if 'prompt' in test_conv and 'images' in test_conv and test_conv['images']:
                prompt_text = test_conv['prompt']
                image_path = test_conv['images'][0]
        elif isinstance(prompt_item, dict) and 'prompt' in prompt_item and 'images' in prompt_item:
            if prompt_item['images']:
                prompt_text = prompt_item['prompt']
                image_path = prompt_item['images'][0]
        else:
            responses.append(f"Warning: Skipping invalid format sample: {prompt_item}")
            continue
        if not image_path or not os.path.exists(image_path):
            responses.append(f"Warning: Sample missing valid image path: {image_path}")
            continue

        try:
            print(f" - Using image: '{image_path}'")

            with autocast(dtype=torch.float16):
                response, _ = model.chat(
                    tokenizer,
                    image_path,
                    query=prompt_text,
                    history=history,
                    max_length=MAX_NEW_TOKEN
                )
            responses.append(response)

        except Exception as e:
            error_message = f"Error processing sample: {e}"
            print(error_message)
            responses.append(error_message)

    return responses