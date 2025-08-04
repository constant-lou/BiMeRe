import json
import re
from collections import Counter
import argparse
import os
from prettytable import PrettyTable
import logging
from typing import Optional, Tuple, List
from difflib import SequenceMatcher
import sys

def find_bad_json_line(file_path):
    """
    Read a .jsonl file line by line and report the first line that fails to parse.
    """
    print(f"--- Checking file: {file_path} ---")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                try:
                    json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"\n!!! Error: JSON format issue found at line {i} !!!")
                    print(f"Error message: {e}")
                    print(f"Problematic line content: \n{line.strip()}")
                    return
        print("\n--- File check completed, no obvious JSON format errors found. ---")
    except FileNotFoundError:
        print(f"Error: File not found {file_path}")
    except Exception as e:
        print(f"Unexpected error occurred: {e}")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_option_labels(text: str, options: List = None) -> Optional[str]:
    """
    Extract multiple choice options (such as A, B, C) from model output text.
    Optimized logic:
    1. Look for explicit answer indicators (such as "Answer:A"), take the last one.
    2. Look for independent option letters (such as "(B)"), take the last one.
    3. Calculate text similarity between answer and each option, take the one with highest similarity.
    """
    if not isinstance(text, str):
        return 'error'
    
    text = text.strip()

    explicit_patterns = [
        r"答案是\s*([A-H])", r"答案[：:]\s*\(([A-H])\)", r"答案[：:]\s*([A-H])",
        r"\*\*答案[：:]\s*([A-H])", r"The correct answer is\s*\(([A-H])\)",
        r"The correct answer is\s*([A-H])"
    ]
    all_explicit_matches = []
    for pattern in explicit_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
        if matches:
            all_explicit_matches.extend(matches)
    if all_explicit_matches:
        return all_explicit_matches[-1].upper()

    general_matches = re.findall(r'\(?([A-H])\)?', text)
    if general_matches:
        return general_matches[-1].upper()

    if options:
        best_match_label = None
        highest_similarity = 0.0

        for i, option_content in enumerate(options):
            label = chr(65 + i)
            option_stripped = option_content.strip()
            
            similarity = SequenceMatcher(None, text, option_stripped).ratio()
            
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match_label = label
        
        if highest_similarity > 0.8:
            return best_match_label

    return None


def calculate_accuracy(file_path: str, save_dir: str) -> Tuple[float, float, float]:
    """
    Calculate accuracy, error rate, and extraction failure rate for a single result file.
    """
    data = []
    acc, err, miss, count = 0, 0, 0, 0
    
    try:
        with open(file_path, "r", encoding='utf-8') as file:
            for line in file:
                data.append(json.loads(line))
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return 0.0, 0.0, 0.0
    except json.JSONDecodeError:
        logging.error(f"JSON parsing error: {file_path}")
        return 0.0, 0.0, 0.0

    for sample in data:
        count += 1
        if sample.get("response"):
            predict = extract_option_labels(sample["response"], sample.get("options"))
            sample["extracted_answer"] = predict
            
            if predict and sample.get("answer") == predict:
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
            sample["status"] = "miss (empty response)"

    if count == 0:
        return 0.0, 0.0, 0.0

    accuracy = acc / count
    error_rate = err / count
    miss_rate = miss / count

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, os.path.basename(file_path))
    with open(save_path, "w", encoding='utf-8') as file:
        for sample in data:
            json.dump(sample, file, ensure_ascii=False)
            file.write("\n")
            
    return accuracy, error_rate, miss_rate

def evaluate_all_files(output_dir: str, save_dir: str):
    """
    Evaluate the performance of all .jsonl files in the specified directory and print results in table format.
    """
    results = PrettyTable()
    results.field_names = ["Model", "Split", "Mode", "Accuracy", "Errors", "Miss"]
    results.align["Model"] = "l"

    files = sorted([f for f in os.listdir(output_dir) if f.endswith('.jsonl')])
    if not files:
        logging.warning(f"No .jsonl files found in directory '{output_dir}'.")
        return

    for file_name in files:
        base_name = file_name.replace('.jsonl', '')
        
        try:
            model_split_part, mode = base_name.rsplit('_', 1)
            model_name, split = model_split_part.rsplit('_C_', 1)
            split = 'C_' + split

            file_path = os.path.join(output_dir, file_name)
            logging.info(f"Processing file: {file_name}")
            accuracy, errors, miss = calculate_accuracy(file_path, save_dir)
            
            results.add_row([
                model_name, 
                split, 
                mode, 
                f"{accuracy:.2%}", 
                f"{errors:.2%}", 
                f"{miss:.2%}"
            ])
        except ValueError:
            logging.warning(f"Skipping file with mismatched format: {file_name}")
            continue

    print(results)
    result_file_path = os.path.join(os.path.dirname(save_dir), "results_cn_eval_all.txt")
    with open(result_file_path, "w", encoding='utf-8') as f:
        f.write(results.get_string())
    logging.info(f"Evaluation results saved to: {result_file_path}")


def main(args):
    """Main function that executes corresponding operations based on command line arguments."""
    if args.evaluate_all:
        evaluate_all_files(args.output_dir, args.save_dir)
    else:
        print(f"Model: {args.model_name}")
        results = PrettyTable()
        results.field_names = ["Mode", "Accuracy", "Errors", "Miss"]
        
        for mode in args.mode:
            file_name = f"{args.model_name}_{args.split}_{mode}.jsonl"
            file_path = os.path.join(args.output_dir, file_name)
            if os.path.exists(file_path):
                accuracy, errors, miss = calculate_accuracy(file_path, args.save_dir)
                results.add_row([mode, f"{accuracy:.2%}", f"{errors:.2%}", f"{miss:.2%}"])
            else:
                results.add_row([mode, "File does not exist", "N/A", "N/A"])
        
        print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate accuracy for large model multiple choice question evaluation.")
    parser.add_argument('--model_name', type=str, default='gpt4o', help='Single model name to evaluate')
    parser.add_argument('--split', type=str, default='C_metaphors', help='Dataset split to evaluate')
    parser.add_argument('--mode', nargs='+', default=['none', 'cot', 'domain', 'emotion', 'one-shot', 'two-shot', 'three-shot'], help='List of modes to evaluate, separated by spaces')
    parser.add_argument('--output_dir', type=str, default='results_cn', help='Directory containing model output .jsonl files')
    parser.add_argument('--save_dir', type=str, default='results_cn_with_status', help='Directory to save .jsonl files with evaluation status')
    parser.add_argument('--evaluate_all', action='store_true', help='Evaluate all .jsonl files in output_dir')
    
    args = parser.parse_args()
    main(args)
    if len(sys.argv) > 1:
        file_to_check = sys.argv[1]
        find_bad_json_line(file_to_check)
    else:
        print("Please provide the path to the .jsonl file to check as an argument.")
        print("Usage: python check_json.py /path/to/your/file.jsonl")