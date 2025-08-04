import os
import gc
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from PIL import Image
import re
import traceback 

def load_model(model_name: str, config: dict, use_accel: bool = True):
    model_path = config.get("model_path_or_name")
    if not model_path or not os.path.isdir(model_path):
        raise ValueError(f"Model path does not exist or is invalid: {model_path}")

    cuda_avail = torch.cuda.is_available()
    if cuda_avail:
        torch.cuda.empty_cache()
        gc.collect()

    device_map = "auto"
    if not cuda_avail:
        print("[INFO] No GPU detected, using CPU.")
        device_map = "cpu"
    else:
        print(f"[INFO] Detected {torch.cuda.device_count()} GPU(s).")
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    attn_implementation = "eager"
    torch_dtype = torch.bfloat16
    print(f"[INFO] Using failsafe attention implementation: '{attn_implementation}' as required by the model.")
    
    print(f"[INFO] Loading '{model_path}' with device_map='{device_map}', 4-bit quantization, and attn_implementation='{attn_implementation}'.")

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map=device_map,
        quantization_config=quantization_config,
        attn_implementation=attn_implementation,
        torch_dtype=torch_dtype
    ).eval()
    
    print(f"[INFO] Model {model_name} loaded successfully in 4-bit mode with eager attention.")
    
    return {'model': model, 'processor': processor}

def infer(prompts: list, model, processor) -> list:
    MAX_NEW_TOKENS = 512
    MAX_IMAGE_SIZE = 960

    def resize_image(image: Image.Image, max_size: int = MAX_IMAGE_SIZE) -> Image.Image:
        if max(image.size) > max_size:
            scale = max_size / max(image.size)
            new_width = round(image.width * scale)
            new_height = round(image.height * scale)
            return image.resize((new_width, new_height), Image.LANCZOS)
        return image

    all_responses = []

    for prompt_item in prompts:
        images = []
        try:
            content = []
            image_paths = []
            
            conversations = prompt_item.get("conversations", [prompt_item])
            
            for turn in conversations:
                if turn.get("prompt"):
                    clean_prompt = re.sub(r"<image>.*?</image>", "", turn["prompt"]).strip()
                    content.append({"type": "text", "text": clean_prompt})
                
                if turn.get("images"):
                    for img_path in turn["images"]:
                        image_paths.append(img_path)
                        content.insert(-1, {"type": "image"})

                if turn.get("response"):
                    content.append({"type": "text", "text": turn["response"]})

            images = [resize_image(Image.open(path).convert("RGB")) for path in image_paths]
            
            prompt_text = processor.apply_chat_template(
                [{"role": "user", "content": content}],
                add_generation_prompt=True
            )

            inputs = processor(
                text=[prompt_text],
                images=images,
                return_tensors="pt"
            ).to(model.device)

            generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
            generated_ids = generated_ids[:, inputs['input_ids'].shape[1]:]
            response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            
            all_responses.append(response)

        except Exception as e:
            print("="*20 + " DETAILED TRACEBACK " + "="*20)
            traceback.print_exc()
            print("="*58)
            
            error_message = f"ERROR during inference:\n{traceback.format_exc()}"
            all_responses.append(error_message)
        
        finally:
            del images
            if 'inputs' in locals():
                del inputs
            if 'generated_ids' in locals():
                del generated_ids
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    return all_responses