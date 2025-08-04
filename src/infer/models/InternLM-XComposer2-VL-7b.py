import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from pathlib import Path
from PIL import Image
import os
from tqdm import tqdm

MAX_NEW_TOKENS = 512

def load_model(model_name, model_args, **kwargs):
    model_path = model_args.get('model_path_or_name')
    if not model_path:
        raise ValueError("model_args must contain 'model_path_or_name'.")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map='auto',
        torch_dtype=torch.bfloat16
    ).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    model_components = {
        'model': model,
        'tokenizer': tokenizer,
        'model_name': model_name,
    }
    return model_components

def infer(prompts, **kwargs):
    model = kwargs.get('model')
    tokenizer = kwargs.get('tokenizer')
    if not model or not tokenizer:
        raise ValueError("'model' and 'tokenizer' must be provided via kwargs.")

    responses = []
    
    for prompt_item in tqdm(prompts, desc="Inference progress", unit="tasks"):
        if "conversations" in prompt_item and isinstance(prompt_item["conversations"], list):
            conversations = prompt_item["conversations"]
            history = []
            
            for conv in conversations[:-1]:
                if 'prompt' in conv and 'images' in conv and 'response' in conv:
                    query_list = [{'image': img_path} for img_path in conv['images']]
                    query_list.append({'text': conv['prompt']})
                    history.append((query_list, conv['response']))
                else:
                    raise ValueError(f"Invalid Few-shot example format: {conv}")
            
            test_conv = conversations[-1]
            if 'prompt' in test_conv and 'images' in test_conv:
                final_query_list = [{'image': img_path} for img_path in test_conv['images']]
                final_query_list.append({'text': test_conv['prompt']})
                
                response, _ = model.chat(
                    tokenizer=tokenizer,
                    query=final_query_list,
                    history=history,
                    max_new_tokens=MAX_NEW_TOKENS
                )
                responses.append(response)
            else:
                raise ValueError(f"Invalid Few-shot test sample format: {test_conv}")

        elif 'prompt' in prompt_item and 'images' in prompt_item:
            query = [{'image': img_path} for img_path in prompt_item['images']]
            query.append({'text': prompt_item['prompt']})
            
            response, _ = model.chat(
                tokenizer=tokenizer,
                query=query,
                history=[],
                max_new_tokens=MAX_NEW_TOKENS
            )
            responses.append(response)
        else:
            raise ValueError(f"Unrecognized prompt format: {prompt_item}")

    return responses

def create_dummy_images():
    temp_dir = Path("./temp_test_images")
    temp_dir.mkdir(exist_ok=True)
    img_path1 = temp_dir / "red_image.png"
    img_path2 = temp_dir / "blue_image.png"
    if not img_path1.exists():
        Image.new('RGB', (100, 100), color = 'red').save(img_path1)
    if not img_path2.exists():
        Image.new('RGB', (100, 100), color = 'blue').save(img_path2)
    return str(img_path1), str(img_path2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="InternLM-XComposer2-VL-7b model inference script")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to InternLM-XComposer2-VL-7B model weights and files."
    )
    args = parser.parse_args()

    print("Loading model...")
    model_name = 'internlm-xcomposer2-vl-7b'
    model_args = {'model_path_or_name': args.model_path}
    model_components = load_model(model_name, model_args)
    print("Model loading completed.")

    img1_path, img2_path = create_dummy_images()

    prompts_to_run = [
        {"prompt": "What is the main color of this image?", "images": [img1_path]},
        {"conversations": [
            {"prompt": "What color is this image?", "images": [img1_path], "response": "This image is red."},
            {"prompt": "Based on the example above, please describe the color of this image.", "images": [img2_path]}
        ]},
        {"prompt": "What is the difference between the first and second images?", "images": [img1_path, img2_path]},
        {"prompt": "Describe the second image.", "images": [img2_path]},
    ]
    
    all_responses = infer(prompts_to_run, **model_components)

    print("\n\n==================== Inference Results Summary ====================")
    for i, (prompt, response) in enumerate(zip(prompts_to_run, all_responses)):
        if "conversations" in prompt:
            final_prompt = prompt['conversations'][-1]['prompt']
            print(f"[Few-shot Task {i+1}]")
            print(f"  > Question: {final_prompt}")
            print(f"  > Answer: {response}")
        else:
            print(f"[Single Sample Task {i+1}]")
            print(f"  > Question: {prompt['prompt']}")
            print(f"  > Answer: {response}")
    print("================================================================")