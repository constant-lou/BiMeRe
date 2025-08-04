import os
import gc
import re
import torch
from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor
from deepseek_vl2.utils.io import load_pil_images

def load_model(model_name: str, config: dict = None, use_accel: bool = True):
    """
    Load DeepSeek-VL2 components.
    Args:
        model_name: should be "deepseek_vl2"
        config: a dict, e.g. {"model_path_or_name": "/path/to/deepseek-vl2"}
        use_accel: whether to use GPU acceleration
    Returns:
        A dict with keys: "model", "processor", "tokenizer"
    """
    config = config or {}
    model_path = config.get(
        "model_path_or_name",
        os.environ.get("MODEL_PATH", "/path/to/deepseek-vl2")
    )
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        
    processor = DeepseekVLV2Processor.from_pretrained(model_path)
    tokenizer = processor.tokenizer
    
    cuda_avail = torch.cuda.is_available() and use_accel
    device_map = "auto" if cuda_avail and torch.cuda.device_count() > 1 else ("cuda:0" if cuda_avail else "cpu")
    torch_dtype = torch.bfloat16 if (cuda_avail and torch.cuda.is_bf16_supported()) else torch.float16
    
    model = DeepseekVLV2ForCausalLM.from_pretrained(
        model_path, trust_remote_code=True,
        device_map=device_map, torch_dtype=torch_dtype
    ).eval()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        
    return {"model": model, "processor": processor, "tokenizer": tokenizer}

USER_ROLE = "<|User|>"
ASSISTANT_ROLE = "<|Assistant|>"
IMAGE_TOKEN = "<image>"
MAX_NEW_TOKENS = 512

def _clean_text(raw: str) -> str:
    """Remove image tags from text."""
    return re.sub(r"<image>.*?</image>", "", raw, flags=re.S).strip()

def _to_conv(item: dict) -> list:
    """
    Convert a prompt dict into the standard conversation format accepted by processor.
    item: {"prompt": str, "images": [path, ...]}
    """
    raw = item.get("prompt", "")
    imgs = item.get("images", []) or []
    text = _clean_text(raw)
    if imgs:
        text = IMAGE_TOKEN + "\n" + text
    return [
        {"role": USER_ROLE, "content": text, "images": imgs},
        {"role": ASSISTANT_ROLE, "content": "", "images": []},
    ]

def _to_conv_with_history(conversations: list) -> list:
    """
    Convert few-shot conversations into DeepSeek-VL2 conversation format.
    Args:
        conversations: List of conversation dicts with 'prompt', 'images', and 'response'
    Returns:
        List of conversation turns in DeepSeek format
    """
    conv_turns = []
    
    for conv_item in conversations:
        raw_prompt = conv_item.get("prompt", "")
        imgs = conv_item.get("images", []) or []
        text = _clean_text(raw_prompt)
        if imgs:
            text = IMAGE_TOKEN + "\n" + text
            
        conv_turns.append({
            "role": USER_ROLE,
            "content": text,
            "images": imgs
        })
        
        if "response" in conv_item and conv_item["response"]:
            conv_turns.append({
                "role": ASSISTANT_ROLE,
                "content": conv_item["response"],
                "images": []
            })
    
    if conv_turns and conv_turns[-1]["role"] == USER_ROLE:
        conv_turns.append({
            "role": ASSISTANT_ROLE,
            "content": "",
            "images": []
        })
    
    return conv_turns

def _single_inference(conv: list, model, processor, tokenizer) -> str:
    """
    Perform single inference with given conversation.
    Args:
        conv: Conversation in DeepSeek format
        model: DeepSeek model
        processor: DeepSeek processor
        tokenizer: DeepSeek tokenizer
    Returns:
        Generated response string
    """
    device = next(model.parameters()).device
    
    pil_imgs = load_pil_images(conv) or []
    
    batch = processor(
        conversations=conv,
        images=pil_imgs,
        force_batchify=True,
        inference_mode=True
    ).to(device)
    
    gen_ids = model.generate(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        images=batch.images,
        images_seq_mask=batch.images_seq_mask,
        images_spatial_crop=batch.images_spatial_crop,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        use_cache=True,
    )
    
    full_response = tokenizer.decode(gen_ids[0].cpu().tolist(), skip_special_tokens=True).strip()
    
    if ASSISTANT_ROLE in full_response:
        parts = full_response.split(ASSISTANT_ROLE)
        if len(parts) > 1:
            response = parts[-1].strip()
        else:
            response = full_response
    else:
        response = full_response
    
    return response

def infer(prompts: list, model, processor, tokenizer) -> list:
    """
    Batch inference for DeepSeek-VL2 under BiMeRe-Bench.
    Supports both few-shot and single-shot inference.
    
    Args:
        prompts: List of dicts, each with either:
                 - {"prompt": str, "images": [str]} for single inference
                 - {"conversations": [{"prompt": str, "images": [str], "response": str}]} for few-shot
        model: the loaded DeepseekVLV2ForCausalLM
        processor: the loaded DeepseekVLV2Processor
        tokenizer: the processor.tokenizer
    Returns:
        List of generated strings, one per prompt.
    """
    responses = []
    
    for prompt_item in prompts:
        try:
            if isinstance(prompt_item, dict) and "conversations" in prompt_item and isinstance(prompt_item["conversations"], list):
                conversations = prompt_item["conversations"]
                conv = _to_conv_with_history(conversations)
                response = _single_inference(conv, model, processor, tokenizer)
                responses.append(response)
                
            elif isinstance(prompt_item, dict) and 'prompt' in prompt_item and 'images' in prompt_item:
                conv = _to_conv(prompt_item)
                response = _single_inference(conv, model, processor, tokenizer)
                responses.append(response)
                
            else:
                raise ValueError(f"Invalid prompt format: {prompt_item}")
                
        except Exception as e:
            print(f"Error processing prompt: {e}")
            responses.append("")
    
    return responses

def infer_batch(prompts: list, model, processor, tokenizer) -> list:
    """Alias for infer function to maintain backward compatibility."""
    return infer(prompts, model, processor, tokenizer)

if __name__ == '__main__':
    pass