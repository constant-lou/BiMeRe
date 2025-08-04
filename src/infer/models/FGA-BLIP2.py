import torch
from PIL import Image
from lavis.models import load_model_and_preprocess

MAX_NEW_TOKEN = 512

def load_model(model_name, model_config, use_accel=False):
    """
    Load InstructBLIP model using LAVIS.
    This function ignores 'model_path_or_name' from model_config
    and uses the official model name defined below.
    """
    print("Loading model: FGA-BLIP2 (via LAVIS)...")

    name = "instruct_blip"
    model_type = "flant5xxl"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"LAVIS loading: name='{name}', model_type='{model_type}'")
    print("Note: This script will automatically download the official pre-trained model and ignore 'model_path_or_name' in the framework configuration.")

    model, vis_processors, txt_processors = load_model_and_preprocess(
        name=name,
        model_type=model_type,
        is_eval=True,
        device=device
    )

    print(f"Model {name} loaded successfully, using device: {device}")
    
    model_components = {
        'model': model,
        'vis_processors': vis_processors,
        'txt_processors': txt_processors,
        'device': device,
        'model_name': model_name,
    }
    
    return model_components

def infer(prompts, **kwargs):
    """
    Perform inference using the loaded InstructBLIP model.
    """
    model = kwargs.get('model')
    vis_processors = kwargs.get('vis_processors')
    device = kwargs.get('device')

    if not all([model, vis_processors, device]):
        raise ValueError("Model components (model, vis_processors, device) not properly provided.")

    responses = []

    for prompt_item in prompts:
        if isinstance(prompt_item, dict) and "conversations" in prompt_item:
            conversations = prompt_item["conversations"]
            context_prompt_parts = []
            for conv in conversations[:-1]:
                context_prompt_parts.append(f"Question: {conv['prompt']} Answer: {conv['response']}")
            
            test_conv = conversations[-1]
            image_path = test_conv['images'][0]
            test_question = test_conv['prompt']
            
            full_prompt = "\n".join(context_prompt_parts)
            if full_prompt:
                full_prompt += f"\nQuestion: {test_question} Answer:"
            else:
                full_prompt = f"Question: {test_question} Answer:"
        
        elif isinstance(prompt_item, dict) and 'prompt' in prompt_item and 'images' in prompt_item:
            image_path = prompt_item['images'][0]
            full_prompt = f"Question: {prompt_item['prompt']} Answer:"
        
        else:
            raise ValueError(f"Invalid prompt format: {prompt_item}")

        raw_image = Image.open(image_path).convert("RGB")
        image_tensor = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

        outputs = model.generate(
            {"image": image_tensor, "prompt": full_prompt},
            max_length=MAX_NEW_TOKEN,
            num_beams=3
        )
        
        responses.append(outputs[0])

    return responses