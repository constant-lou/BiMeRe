from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

MAX_NEW_TOKEN = 512

def load_model(model_name, model_args, use_accel=False):
    model_path = model_args.get('model_path_or_name')
    tp = model_args.get('tp', 2)
    model_components = {}
    if use_accel:
        model_components['use_accel'] = True
        pass
    else:
        model_components['use_accel'] = False
        model_components['model'] = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map='cuda', torch_dtype=torch.float16).eval()
        model_components['tokenizer'] = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model_components['model_name'] = model_name
    return model_components

def infer(prompts, **kwargs):
    model = kwargs.get('model')
    tokenizer = kwargs.get('tokenizer', None)
    use_accel = kwargs.get('use_accel', False)
    responses = []

    for prompt_item in prompts:
        if isinstance(prompt_item, dict) and "conversations" in prompt_item and isinstance(prompt_item["conversations"], list):
            conversations = prompt_item["conversations"]
            history = []
            
            for i, conv in enumerate(conversations[:-1]):
                if isinstance(conv, dict) and 'prompt' in conv and 'images' in conv:
                    prompt_text = conv['prompt']
                    query = tokenizer.from_list_format([{'image': image} for image in conv['images']] + [{'text': prompt_text}])
                    
                    if 'response' in conv and conv['response']:
                        history.append((query, conv['response']))
                else:
                    raise ValueError(f"Invalid conversation format: {conv}")
            
            test_conv = conversations[-1]
            if isinstance(test_conv, dict) and 'prompt' in test_conv and 'images' in test_conv:
                prompt_text = test_conv['prompt']
                query = tokenizer.from_list_format([{'image': image} for image in test_conv['images']] + [{'text': prompt_text}])
                
                response, _ = model.chat(tokenizer, query=query, history=history, max_new_tokens=MAX_NEW_TOKEN)
                responses.append(response)
            else:
                raise ValueError(f"Invalid test conversation format: {test_conv}")
                
        elif isinstance(prompt_item, dict) and 'prompt' in prompt_item and 'images' in prompt_item:
            prompt_text = prompt_item['prompt']
            query = tokenizer.from_list_format([{'image': image} for image in prompt_item['images']] + [{'text': prompt_text}])
            response, history = model.chat(tokenizer, query=query, history=None, max_new_tokens=MAX_NEW_TOKEN)
            responses.append(response)
        else:
            raise ValueError(f"Invalid prompt format: {prompt_item}")

    return responses

if __name__ == '__main__':
    pass