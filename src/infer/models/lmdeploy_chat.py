import sys
import torch
from transformers import AutoTokenizer
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
from PIL import Image
import os
from internvl.train.dataset import build_transform

sys.path.insert(0, '/path/to/internvl_chat')
from internvl.model.internvl_chat import InternVLChatModel

def load_model(model_name, model_args, use_accel=True):
    if not model_args or 'model_path_or_name' not in model_args:
        raise ValueError("model_args must contain 'model_path_or_name'.")

    model_path = model_args.get('model_path_or_name')
    
    print(f"Loading model from {model_path} using Hugging Face Transformers standard method...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = InternVLChatModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).to(device)
        model.eval()
        
        print("Model loaded successfully!")
        
        return {
            'tokenizer': tokenizer,
            'model': model,
            'model_name': model_name
        }
    except Exception as e:
        print(f"Failed to load model using standard method: {e}")
        import traceback
        traceback.print_exc()
        return None

def encode_image(image_path):
    try:
        if not os.path.exists(image_path):
            print(f"Image not found at path: {image_path}")
            return None
        with Image.open(image_path) as image:
            return image.convert("RGB")
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

def infer(prompts, **kwargs):
    model = kwargs.get('model')
    tokenizer = kwargs.get('tokenizer')
    
    if not model or not tokenizer:
        return ["Error: Model or tokenizer not found in kwargs."] * len(prompts)
        
    responses = []
    
    image_transform = build_transform(is_train=False, input_size=448)
    generation_config = {
        "max_new_tokens": 1024,
        "do_sample": False,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    }

    for prompt_item in prompts:
        try:
            pixel_values = None
            history = None
            
            if isinstance(prompt_item, dict) and "conversations" in prompt_item:
                conversations = prompt_item["conversations"]
                if not conversations:
                    raise ValueError("Conversations list is empty.")

                history = []
                for turn in conversations[:-1]:
                    history.append((turn['prompt'], turn.get('response', '')))

                final_turn = conversations[-1]
                query = final_turn['prompt']
                image_path = final_turn['images'][0] if final_turn.get('images') else None

            elif isinstance(prompt_item, dict) and 'prompt' in prompt_item:
                query = prompt_item['prompt']
                image_path = prompt_item['images'][0] if prompt_item.get('images') else None
                history = None
            
            else:
                raise ValueError(f"Invalid prompt format: {prompt_item}")

            if image_path:
                image = Image.open(image_path).convert("RGB")
                pixel_values = image_transform(image).unsqueeze(0).to(torch.bfloat16).to(model.device)

            response = model.chat(
                tokenizer,
                pixel_values,
                query,
                generation_config=generation_config,
                history=history
            )
            responses.append(response)

        except Exception as e:
            error_message = f"Error processing item: {e}"
            print(f"Error: {error_message}")
            responses.append(error_message)
            import traceback
            traceback.print_exc()
            
    return responses