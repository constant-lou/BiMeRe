import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
import re

def load_model(model_name: str, model_args: dict, use_accel: bool = False):
    model_path = model_args.get('model_path_or_name')
    if not model_path:
        raise ValueError("model_args must contain 'model_path_or_name' key.")

    print(f"INFO: Loading model from '{model_path}'...")
    print("INFO: Enabled 4-bit quantization loading, will significantly reduce memory usage and improve speed.")
    if use_accel:
        print("WARNING: 'use_accel' parameter is True, but current loading logic does not implement specific acceleration.")
    try:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )

        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            quantization_config=quantization_config
        ).eval()

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print("INFO: Model loading completed.")
        return {'model': model, 'tokenizer': tokenizer}
    except Exception as e:
        print(f"ERROR: Model loading failed. Error: {e}")
        raise

def infer(prompts: list, **kwargs):
    model = kwargs.get('model')
    tokenizer = kwargs.get('tokenizer')
    if not model or not tokenizer:
        raise ValueError("'model' and 'tokenizer' must be provided when calling infer function.")

    responses = []
    
    for item in prompts:
        try:
            if isinstance(item, dict) and item.get('few-shot') and 'conversations' in item:
                conversations = item['conversations']
                if not conversations:
                    raise ValueError("'conversations' list is empty in few-shot mode.")
                
                history = []
                for turn in conversations[:-1]:
                    original_question = turn['prompt']
                    answer = turn.get('response', '')
                    clean_question = re.sub(r'<image>.*?</image>', '', original_question).strip()
                    history.append((clean_question, answer))

                final_turn = conversations[-1]
                original_final_question = final_turn['prompt']
                clean_final_question = re.sub(r'<image>.*?</image>', '', original_final_question).strip()
                msgs_for_model = [{'role': 'user', 'content': clean_final_question}]
                
                final_image_paths = final_turn.get('images', [])
                final_images = [Image.open(p).convert('RGB') for p in final_image_paths]

                response = model.chat(
                    image=final_images[0] if final_images else None,
                    msgs=msgs_for_model,
                    history=history,
                    tokenizer=tokenizer,
                    sampling=True, temperature=0.7, max_new_tokens=1024
                )
                responses.append(response)

            elif isinstance(item, dict) and 'prompt' in item:
                original_prompt_text = item['prompt']
                image_paths = item.get('images', [])
                clean_prompt_text = re.sub(r'<image>.*?</image>', '', original_prompt_text).strip()
                msgs_for_model = [{'role': 'user', 'content': clean_prompt_text}]
                images = [Image.open(p).convert('RGB') for p in image_paths]
                
                response = model.chat(
                    image=images[0] if images else None,
                    msgs=msgs_for_model,
                    history=None,
                    tokenizer=tokenizer,
                    sampling=True, temperature=0.7, max_new_tokens=1024
                )
                responses.append(response)
            else:
                raise ValueError(f"Unrecognized prompt structure: {item}")

        except Exception as e:
            error_message = f"Error processing item: {e}"
            print(f"WARNING: {error_message}")
            responses.append(error_message)
            
    return responses