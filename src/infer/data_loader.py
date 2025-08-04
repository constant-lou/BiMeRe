import json
import yaml
import os
import random

random.seed(42)
IMAGE_ROOT="/path/to/BiMeRe"

def read_json_or_jsonl(data_path, split='', mapping_key=None):
    base_path = f'{data_path}/{split}'
    if os.path.exists(f'{base_path}.json'):
        file_path = f'{base_path}.json'
    elif os.path.exists(f'{base_path}.jsonl'):
        file_path = f'{base_path}.jsonl'
    else:
        raise FileNotFoundError(f"No JSON or JSONL file found for {base_path}.")

    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        if file_path.endswith('.json'):
            json_data = json.load(file)
        elif file_path.endswith('.jsonl'):
            json_data = [json.loads(line) for line in file]
        
        for image_data in json_data:
            for item in image_data.get("questions", []):
                if "meta_data" not in item:
                    item["meta_data"] = {}

                for key, value in image_data.get("meta_data", {}).items():
                    item["meta_data"][key] = value
                
                if split == "E_metaphors":
                    item["meta_data"]["language"] = "en"
                elif split == "C_metaphors":
                    item["meta_data"]["language"] = "zh"
                elif split == "E_test":
                    item["meta_data"]["language"] = "en"
                elif split == "C_test":
                    item["meta_data"]["language"] = "zh"

                item["local_path"] = image_data.get("local_path", "")
                data.append(item)
    return data

def read_yaml(config='default'):
    config_path = f'src/config/prompt/{config}.yaml'
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as yaml_file:
        return yaml.safe_load(yaml_file)

def load_data(split='all', mode='none'):
    print(f"Loading data: split='{split}', mode='{mode}'")

    def get_corrected_image_path(instance):
        local_path = instance.get("local_path", "")
        if not local_path:
            print(f"[WARNING] Sample {instance.get('id', 'unknown')} has no 'local_path' field, skipping.")
            return None

        lang = instance.get("meta_data", {}).get("language", "")
        if lang == 'zh':
            correct_folder = 'Cimages'
        elif lang == 'en':
            correct_folder = 'Eimages'
        else:
            print(f"[WARNING] Cannot determine image directory from 'language' field (ID: {instance.get('id', 'unknown')}), using default directory.")
            correct_folder = 'images'
        
        image_filename = os.path.basename(local_path)
        corrected_local_path = os.path.join(correct_folder, image_filename)
        image_path = os.path.join(IMAGE_ROOT, corrected_local_path)

        if not os.path.exists(image_path):
            print(f"[ERROR] Image not found! Attempted path: {image_path}. Please check if file exists.")
            return None
        
        return image_path

    if split in ["all", "E_metaphors", "C_metaphors"] and mode in ['one-shot', 'two-shot', 'three-shot']:
        template = read_yaml("none")
        samples = read_json_or_jsonl('./data', split)
        print(f"[INFO] Loaded {len(samples)} samples for split: {split}")

        num_examples = min(3, len(samples))
        examples = random.sample(samples, num_examples) if len(samples) >= num_examples else samples
        print(f"[INFO] Selected {len(examples)} examples for few-shot mode.")
    
        for sample in samples:
            prompts = {"id": sample.get("id", "unknown"), "few-shot": True, "conversations": []}
            
            shot_map = {'one-shot': 1, 'two-shot': 2, 'three-shot': 3}
            example_shots = examples[:shot_map[mode]]

            for turn_id, instance in enumerate(conversation_instances):
                image_path = get_corrected_image_path(instance)
                if not image_path:
                    continue
                
                question = instance.get("question", "")
                question_content = selected_template["prompt_format"].format(question, image_path, *options)

                prompt = {
                    'prompt': question_content,
                    'images': [image_path],
                    "id": f"{instance.get('id', 'unknown')}-turn-{turn_id}"
                }
                
                if turn_id < len(conversation_instances) - 1:
                    prompt["response"] = instance.get("answer", "")
                
                prompts["conversations"].append(prompt)

            if prompts["conversations"]:
                yield prompts, sample

    elif split in ['all', 'E_metaphors', 'C_metaphors'] and mode in ['none', 'cot', 'domain', 'emotion']:
        config_name = mode if mode in ['none', 'cot'] else 'key-words'
        template = read_yaml(config_name)
        
        samples = read_json_or_jsonl('./data', "E_metaphors") + read_json_or_jsonl('./data', "C_metaphors") if split == "all" else read_json_or_jsonl('./data', split)
        print(f"[INFO] Loaded {len(samples)} samples for split: {split}")

        for sample in samples:
            image_path = get_corrected_image_path(sample)
            if not image_path:
                continue

            options = sample.get("options", []) + ["N/A"] * (6 - len(sample.get("options", [])))
            lang = sample.get("meta_data", {}).get("language", "en")
            selected_template = template["prompt_templates"].get(lang, template["prompt_templates"]["en"])
            question = sample.get("question", "")
            
            if mode in ['none', 'cot']:
                question_content = selected_template["prompt_format"].format(question, image_path, *options)
            else:
                key_words = sample.get("meta_data", {}).get(mode, "")

            prompt = {
                "prompt": f"{selected_template['instruction']}\n{question_content}",
                "images": [image_path]
            }
            yield prompt, sample
            
    else:
        print(f"[ERROR] Invalid parameter combination: split='{split}', mode='{mode}'")

if __name__ == '__main__':
    print("="*50)
    print("Testing data loader (v2 fixed version)...")
    print("="*50)
    
    print("\n--- Test 1: split='C_metaphors', mode='none' ---")
    data_generator_c = load_data('C_metaphors', 'none')
    try:
        first_prompt_c, first_sample_c = next(data_generator_c)
        print("\nSuccessfully generated first prompt for C_metaphors:")
        print(json.dumps(first_prompt_c, indent=2, ensure_ascii=False))
    except StopIteration:
        print("\nFailed to generate any data for C_metaphors. Please check related files and paths.")
    except Exception as e:
        print(f"\nUnknown error occurred while processing C_metaphors: {e}")

    print("\n" + "="*50)
    print("\n--- Test 2: split='E_metaphors', mode='none' ---")
    data_generator_e = load_data('E_metaphors', 'none')
    try:
        first_prompt_e, first_sample_e = next(data_generator_e)
        print("\nSuccessfully generated first prompt for E_metaphors:")
        print(json.dumps(first_prompt_e, indent=2, ensure_ascii=False))
    except StopIteration:
        print("\nFailed to generate any data for E_metaphors. Please check related files and paths.")
    except Exception as e:
        print(f"\nUnknown error occurred while processing E_metaphors: {e}")