from data_loader import load_data
from models import load_model, infer
import json
import sys
import argparse
from tqdm import tqdm
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from config_wrapper import ConfigWrapper
from tenacity import RetryError
sys.stdout.reconfigure(encoding="utf-8")

import sys
import os

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(src_dir)

sys.path.insert(0, '/path/to/internvl_chat')

print("sys.path:", sys.path)
try:
    import utils
    print("utils module found!")
except ImportError:
    print("utils module not found!")

def check_completed(output_file):
    completed = {}
    no_response_id = []
    try:
        with open(output_file, 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line)
                response_key = config_wrapper.get('response_key')
                error_key = config_wrapper.get('error_key')
                id_key = config_wrapper.get('id_key')
                if response_key in data and (isinstance(data[response_key], str) or (isinstance(data[response_key], dict) and error_key not in data[response_key]) or (error_key in data[response_key] and 'Request failed: 400' in data[response_key][error_key])):
                    completed[config_wrapper.get_id(data)] = data[response_key]
                else:
                    no_response_id.append(config_wrapper.get_id(data))
    except FileNotFoundError:
        pass
    except json.JSONDecodeError:
        pass
    return completed, no_response_id

def infer_batch(model_components, model_name, batch):
    results = []
    prompts = [sample[config_wrapper.get('prompt_key')] for sample in batch]
    
    responses = infer(model_name)(prompts, **model_components)
    for sample, response in zip(batch, responses):
        results.append(sample)
    return results

def main(model_name='gpt4o', splits='E_metaphors', modes=['dp', 'pyagent'], output_dir='results', infer_limit=None, num_workers=1, batch_size=4, use_accel=False):
    print('-'*100)
    print("[INFO] model_name:", model_name)
    print("[INFO] splits:", splits)
    print("[INFO] modes:", modes)
    print("[INFO] output_dir:", output_dir)
    print("[INFO] Infer Limit:", "No limit" if infer_limit is None else infer_limit)
    print("[INFO] Number of Workers:", num_workers)
    print("[INFO] Batch Size:", batch_size)
    print("[INFO] Use Accel:", use_accel)
    print('-'*100)
    model_components = None
    
    os.makedirs(output_dir, exist_ok=True)
    for split in splits:
        for mode in modes:
            output_file_path = f'{output_dir}/{model_name}_{split}_{mode}.jsonl'
            temp_output_file_path = f'{output_file_path}.tmp'
            
            completed, _ = check_completed(output_file_path)
            temp_completed, _ = check_completed(temp_output_file_path)
            merged = {**temp_completed, **completed}
            infer_count = 0

            with open(temp_output_file_path, 'w', encoding='utf-8') as temp_file:
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = []
                    batch = []
                    for prompt, sample in tqdm(load_data(split=split, mode=mode), desc=f'Processing {mode}'):
                        sample[config_wrapper.get('prompt_key')] = prompt
                        if config_wrapper.get_id(sample) in merged:
                            sample[config_wrapper.get('response_key')] = merged[config_wrapper.get_id(sample)]
                            json.dump(sample, temp_file, ensure_ascii=False)
                            temp_file.write('\n')
                            temp_file.flush()
                            continue
                        if infer_limit is not None and infer_count >= infer_limit:
                            break
                        if model_components is None:
                            model_components = load_model(model_name, use_accel)
                        batch.append(sample)
                        infer_count += 1
                        if len(batch) == batch_size:
                            futures.append(executor.submit(infer_batch, model_components, model_name, batch.copy()))
                            batch = []
                        if infer_limit is not None and infer_count >= infer_limit:
                            break

                    if batch:
                        futures.append(executor.submit(infer_batch, model_components, model_name, batch))

                    for future in tqdm(as_completed(futures), total=len(futures), desc=f'Writing {mode} results'):
                        results = future.result()
                        print("Saving results...")
                        for result in results:
                            json.dump(result, temp_file, ensure_ascii=False)
                            temp_file.write('\n')
                            temp_file.flush()
            
            shutil.move(temp_output_file_path, output_file_path)
            _, no_response_id = check_completed(output_file_path)
            if len(no_response_id) > 0:
                print(f"Failed to get response for {len(no_response_id)} questions in {mode} mode. IDs: {no_response_id}", file=sys.stderr)
        print(f'Inference for {split} completed.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference and save results.')
    parser.add_argument('--model_name', type=str, default='Qwen-VL-Chat', help='Model name to use')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Config file to use')
    parser.add_argument('--split', nargs='+', default=['E_metaphors'], help='Data split to use')
    parser.add_argument('--mode', nargs='+', default=['none'], help='Modes to use for data loading, separated by space')
    parser.add_argument('--output_dir', type=str, default='results_bimere', help='Directory to write results')
    parser.add_argument('--infer_limit', type=int, help='Limit the number of inferences per run, default is no limit', default=None)
    parser.add_argument('--num_workers', type=int, default=1, help='Number of concurrent workers for inference')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--use_accel', action='store_true', help='Use inference acceleration framework for inference, LLM-->vLLM, VLM-->lmdeploy')
    args = parser.parse_args()
    config_wrapper = ConfigWrapper(args.config)

    main(model_name=args.model_name, splits=args.split, modes=args.mode, output_dir=args.output_dir, infer_limit=args.infer_limit, num_workers=args.num_workers, batch_size=args.batch_size, use_accel=args.use_accel)