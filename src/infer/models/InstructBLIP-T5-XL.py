import torch
from PIL import Image
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import os
import re
import requests

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

MAX_NEW_TOKEN = 16

def load_model(model_name, model_args, use_accel=False):
    model_path = model_args.get('model_path_or_name')
    if not model_path:
        raise ValueError("model_args must contain 'model_path_or_name'")

    print(f"Loading model: {model_name}")
    model_components = {}
    
    if use_accel:
        print("Note: 'use_accel=True' acceleration logic needs to be implemented.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = InstructBlipForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    processor = InstructBlipProcessor.from_pretrained(model_path)
    model_components['model'] = model
    model_components['tokenizer'] = processor
    model_components['model_name'] = model_name
    model_components['device'] = device
    
    print(f"Model {model_name} successfully loaded to: {model.device}")
    return model_components

def infer(prompts_to_run, **kwargs):
    model = kwargs.get('model')
    processor = kwargs.get('tokenizer')
    device = model.device
    
    if not all([model, processor]):
        raise ValueError("Model or processor not provided.")
    
    responses = [] 
    
    for i, prompt_item in enumerate(prompts_to_run):
        print(f"Processing sample {i+1}/{len(prompts_to_run)}")
        try:
            if 'conversations' in prompt_item and prompt_item.get('few-shot'):
                print("Few-Shot mode detected")
                conversations = prompt_item['conversations']
                
                final_turn = conversations[-1]
                image_paths = final_turn['images']
                
                full_prompt_text = ""
                
                for turn in conversations[:-1]:
                    question_part = turn['prompt']
                    answer_part = turn['response']
                    
                    full_prompt_text += question_part + " " + answer_part + "\n\n"
                
                full_prompt_text += final_turn['prompt']
                
                prompt_text = full_prompt_text

            else:
                prompt_text = prompt_item['prompt']
                image_paths = prompt_item['images']
            
            cleaned_prompt_text = re.sub(r'<image>.*?</image>', '', prompt_text).strip()
            images = [Image.open(p).convert("RGB") for p in image_paths]

            inputs = processor(
                images=images, text=cleaned_prompt_text, return_tensors="pt",
                truncation=True, max_length=480 
            ).to(device, torch.float16)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKEN,
                num_beams=3 
            )
            
            raw_generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
            print(f"Model raw output: '{raw_generated_text}'")

            responses.append(raw_generated_text) 
            
        except KeyError as e:
            error_msg = f"KeyError occurred while processing sample, please check input data structure: {e}"
            print(error_msg)
            responses.append(error_msg)
        except Exception as e:
            error_msg = f"Unknown error occurred while processing sample: {e}"
            print(error_msg)
            responses.append(f"error: {str(e)}")
            
    return responses