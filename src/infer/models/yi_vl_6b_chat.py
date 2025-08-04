import torch
from PIL import Image
import os

try:
    from models.llava.conversation import conv_templates
    from models.llava.mm_utils import (
        KeywordsStoppingCriteria,
        expand2square,
        get_model_name_from_path,
        load_pretrained_model,
        tokenizer_image_token,
    )
    from models.llava.model.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, key_info
except ImportError as e:
    print(f"Error importing LLaVA modules: {e}")
    print("Please ensure 'llava' package is installed or path is correct.")
    raise e

MAX_NEW_TOKEN = 1024

def disable_torch_init():
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

def load_model(model_name, model_args, use_accel=True):
    model_path = model_args.get('model_path_or_name')
    if not model_path:
        raise ValueError("'model_path_or_name' not found in model_args.")

    print(f"[INFO] Loading {model_name} from {model_path}. Use Accel: {use_accel}")

    num_gpus = 0
    try:
        num_gpus = torch.cuda.device_count()
    except Exception as e:
        print(f"Warning: Unable to detect GPU count: {e}")
        
    print(f"Found {num_gpus} GPUs.")

    if num_gpus > 1:
        print("Multi-GPU detected, but currently loading onto a single GPU (cuda:0).")

    disable_torch_init()
    key_info["model_path"] = model_path
    get_model_name_from_path(model_path)

    device = torch.device("cuda:0" if use_accel and num_gpus > 0 else "cpu")
    dtype = torch.bfloat16 if use_accel and num_gpus > 0 else torch.float32

    print(f"[INFO] Using device: {device}, dtype: {dtype}")

    try:
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path)
    except Exception as e:
        print(f"[ERROR] Failed to load pretrained model from {model_path}.")
        print(f"Please ensure it's a valid local path.")
        raise e

    model = model.to(device=device, dtype=dtype).eval()
    print(f"[INFO] Model moved to {device} with {dtype}.")

    model_components = {
        'model': (model, image_processor),
        'tokenizer': tokenizer,
        'use_accel': use_accel,
        'model_name': model_name
    }
    print("[INFO] Model loaded successfully.")
    return model_components

def infer(prompts, **kwargs):
    model_tuple = kwargs.get('model')
    tokenizer = kwargs.get('tokenizer', None)

    if not model_tuple or not tokenizer:
        raise ValueError("Model or Tokenizer not found in kwargs")

    model, image_processor = model_tuple
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    responses = []

    for prompt_item in prompts:
        images_paths = []
        conv = conv_templates['mm_default'].copy()

        if isinstance(prompt_item, dict) and "conversations" in prompt_item:
            conversations = prompt_item["conversations"]
            for conv_turn in conversations[:-1]:
                if not (isinstance(conv_turn, dict) and 'prompt' in conv_turn and 'response' in conv_turn):
                    raise ValueError(f"Invalid history conversation format: {conv_turn}")
                conv.append_message(conv.roles[0], conv_turn['prompt'])
                conv.append_message(conv.roles[1], conv_turn['response'])

            last_conv = conversations[-1]
            if not (isinstance(last_conv, dict) and 'prompt' in last_conv and 'images' in last_conv):
                raise ValueError(f"Invalid test conversation format: {last_conv}")
            images_paths = last_conv['images']
            last_prompt = (DEFAULT_IMAGE_TOKEN + "\n" if images_paths else "") + last_conv['prompt']
            conv.append_message(conv.roles[0], last_prompt)
            conv.append_message(conv.roles[1], None)

        elif isinstance(prompt_item, dict) and 'prompt' in prompt_item and 'images' in prompt_item:
            images_paths = prompt_item['images']
            last_prompt = (DEFAULT_IMAGE_TOKEN + "\n" if images_paths else "") + prompt_item['prompt']
            conv.append_message(conv.roles[0], last_prompt)
            conv.append_message(conv.roles[1], None)
        else:
            raise ValueError(f"Invalid prompt format: {prompt_item}")

        if not images_paths:
            print(f"Warning: No image found for prompt: {prompt_item.get('prompt', 'N/A')}. Inferring without image.")
            if DEFAULT_IMAGE_TOKEN in conv.messages[-2][1]:
                raise ValueError("Image token present but no image paths found.")

        prompt_str = conv.get_prompt()
        input_ids = (
            tokenizer_image_token(prompt_str, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(device)
        )

        image_tensor_to_pass = None
        if images_paths:
            image_file = images_paths[-1]
            try:
                image = Image.open(image_file).convert('RGB')
                if getattr(model.config, "image_aspect_ratio", None) == "pad":
                    image = expand2square(
                        image, tuple(int(x * 255) for x in image_processor.image_mean)
                    )
                image_tensor = image_processor.preprocess(image, return_tensors="pt")[
                    "pixel_values"
                ][0]
                image_tensor_to_pass = image_tensor.unsqueeze(0).to(device=device, dtype=dtype)
            except FileNotFoundError:
                print(f"[ERROR] Image file not found: {image_file}")
                responses.append(f"Error: Image file not found - {image_file}")
                continue
            except Exception as e:
                print(f"[ERROR] Image processing failed: {e}")
                responses.append(f"Error: Image processing failed - {image_file}")
                continue

        stop_str = conv.sep
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor_to_pass,
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                num_beams=5,
                stopping_criteria=[stopping_criteria],
                max_new_tokens=MAX_NEW_TOKEN,
                use_cache=True,
            )

        input_token_len = input_ids.shape[1]
        outputs = tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True
        )[0]
        outputs = outputs.strip()

        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()
        responses.append(outputs)

    return responses

if __name__ == '__main__':
    print("Yi-VL 6B script with GPU detection (single GPU load).")
    pass