import json
import re
from collections import Counter
import argparse
import os
from prettytable import PrettyTable
from difflib import SequenceMatcher

def extract_option_labels(text, options=None):
    """
    A robust function for extracting option labels (A-F) from model responses.
    V4 update:
    - Added strategy 7: Use text similarity for fuzzy matching as a final fallback strategy.
    """
    if not isinstance(text, str):
        return 'error'

    text = text.strip()

    pattern = r'\\boxed{([A-F])}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1]

    pattern = r'(?::|is:)\s*([A-F])\b'
    matches = re.findall(pattern, text, re.IGNORECASE)
    if matches:
        return matches[-1]

    final_answer_candidates = []
    keyword_patterns = [
        r"[Tt]he(?: final| correct)? (?:answer|option) is\s*\(?([A-F])\)?",
        r"(?:(?:最终)?答案|选择|选项)\s*(?:是|为|：|:)\s*\(?([A-F])\)?",
        r"^[Aa]nswer\s*[:：]\s*\(?([A-F])\)?",
    ]
    for pattern in keyword_patterns:
        matches = re.findall(pattern, text, re.DOTALL | re.MULTILINE)
        if matches:
            final_answer_candidates.extend(matches)
    
    if final_answer_candidates:
        return final_answer_candidates[-1]

    pattern = r"[\(（\[【]\s*([A-F])\s*[\)）\]】]"
    matches = re.findall(pattern, text)
    if matches:
        return Counter(matches).most_common(1)[0][0]

    pattern = r"\b([A-F])\b"
    matches = re.findall(pattern, text)
    if matches:
        return Counter(matches).most_common(1)[0][0]

    if options:
        sorted_options = sorted(options, key=len, reverse=True)
        for option_text in sorted_options:
            if option_text.strip() in text:
                label = chr(65 + options.index(option_text))
                return label

    if options:
        best_match_label = None
        highest_similarity = 0.0

        for i, option_content in enumerate(options):
            label = chr(65 + i)
            option_stripped = option_content.strip()
            
            if not option_stripped: continue

            similarity = SequenceMatcher(None, text, option_stripped).ratio()
            
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match_label = label
        
        if highest_similarity > 0.8:
            return best_match_label

    return None

def calculate_accuracy(file_path, save_dir):
    data = []
    acc, count, err, miss = 0, 0, 0, 0
    
    with open(file_path, "r", encoding='utf-8') as file:
        print(f"Processing file: {file_path}")
        for line in file:
            data.append(json.loads(line))
            
    for sample in data:
        if sample.get("response") is not None and sample["response"] != "":
            options_list = [sample.get(f"option_{chr(65+i).lower()}") for i in range(6) if sample.get(f"option_{chr(65+i).lower()}")]
            predict = extract_option_labels(sample["response"], options_list)
            sample["extracted_answer"] = predict
            
            if predict and sample["answer"] == predict:
                acc += 1
                sample["status"] = "correct"
            elif predict is None:
                miss += 1
                sample["status"] = "miss"
            elif predict == 'error':
                err += 1
                sample["status"] = "error"
            else:
                sample["status"] = "incorrect"
        else:
            miss += 1
            sample["extracted_answer"] = None
            sample["status"] = "miss"
        
        count += 1
    
    accuracy, errors, miss_rate = (acc / count, err / count, miss / count) if count > 0 else (0, 0, 0)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, os.path.basename(file_path))
    
    with open(save_path, "w", encoding='utf-8') as file:
        for sample in data:
            json.dump(sample, file, ensure_ascii=False)
            file.write("\n")
    
    return accuracy, errors, miss_rate

def evaluate_all_files(output_dir, save_dir):
    results = PrettyTable()
    results.field_names = ["Model", "Split", "Mode", "Accuracy", "Errors", "Miss"]
    
    try:
        files = sorted(os.listdir(output_dir))
    except FileNotFoundError:
        print(f"[Error] Directory not found: {os.path.abspath(output_dir)}")
        return

    if not files:
        print(f"[Warning] No files found in directory: {os.path.abspath(output_dir)}")
        return

    for file_name in files:
        if file_name.endswith('.jsonl'):
            base_name = file_name.replace('.jsonl', '')
            separator = '_E_'
            
            if separator in base_name:
                try:
                    model_part, rest = base_name.split(separator, 1)
                    task_part, mode = rest.rsplit('_', 1)
                    model_name = model_part
                    split = f"E_{task_part}"
                    
                    file_path = os.path.join(output_dir, file_name)
                    accuracy, errors, miss = calculate_accuracy(file_path, save_dir)
                    results.add_row([model_name, split, mode, f"{accuracy:.2%}", f"{errors:.2%}", f"{miss:.2%}"])
                except ValueError:
                    print(f"Skipping file with unexpected format (cannot split task and mode): {file_name}")
            else:
                print(f"Skipping file (separator '{separator}' not found): {file_name}")
    
    print("\n--- Evaluation Results ---")
    print(results)
    result_file_path = "evaluation_results.txt"
    with open(result_file_path, "w", encoding='utf-8') as f:
        f.write(results.get_string())
    print(f"Results have been saved to {result_file_path}")

def main(args):
    if args.evaluate_all:
        evaluate_all_files(args.output_dir, args.save_dir)
    else:
        print(f"Evaluating single model: {args.model_name}")
        results = PrettyTable()
        results.field_names = ["Mode", "Accuracy", "Errors", "Miss"]
        
        for mode in args.mode:
            file_name = f"{args.model_name}_{args.split}_{mode}.jsonl"
            file_path = os.path.join(args.output_dir, file_name)
            if os.path.exists(file_path):
                accuracy, errors, miss = calculate_accuracy(file_path, args.save_dir)
                results.add_row([mode, f"{accuracy:.2%}", f"{errors:.2%}", f"{miss:.2%}"])
            else:
                results.add_row([mode, "File not found", "N/A", "N/A"])
        
        print(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate accuracy for model evaluation results.")
    parser.add_argument('--model_name', type=str, default='yi-vl-6b-chat', help='Model name to use for single evaluation.')
    parser.add_argument('--split', type=str, default='test', help='Data split to use (e.g., test, E_test).')
    parser.add_argument('--mode', nargs='+', default=['none', 'cot', '1-shot'], help='Modes for single evaluation.')
    parser.add_argument('--output_dir', type=str, default='results_bimere', help='Directory to read result files from.')
    parser.add_argument('--save_dir', type=str, default='results_bimere_with_status', help='Directory to save result files with status.')
    parser.add_argument('--evaluate_all', action='store_true', help='Evaluate all conforming .jsonl files in the output directory.')
    
    args = parser.parse_args()
    main(args)