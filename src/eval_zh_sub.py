import json
import os
import argparse
from prettytable import PrettyTable
from collections import defaultdict

def get_model_name_from_filename(filename_with_ext):
    """
    Robustly extract model name and mode from filename.
    """
    base_name = filename_with_ext.replace('.jsonl', '')
    try:
        model_split_part, mode = base_name.rsplit('_', 1)
        if '_C_' in model_split_part:
            model_name, _ = model_split_part.rsplit('_C_', 1)
        elif '_E_' in model_split_part:
            model_name, _ = model_split_part.rsplit('_E_', 1)
        else: 
            model_name = model_split_part
        return model_name, mode
    except ValueError:
        return "Unknown_Model", "Unknown_Mode"

def evaluate_by_dimension(input_dir, dimension_key, categories, table_headers, filter_mode=None, category_mapping=None):
    """
    A generic evaluation function that can group statistics by any dimension (such as domain, emotion).
    Can filter files by specified filter_mode.
    Generated tables are sorted by Overall accuracy from smallest to largest.
    
    New parameters:
    category_mapping: Dictionary for mapping original categories to target categories
    """
    model_scores = defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0}))
    
    files = sorted([f for f in os.listdir(input_dir) if f.endswith('.jsonl')])

    for file_name in files:
        model_name, mode = get_model_name_from_filename(file_name)
        
        if filter_mode and mode != filter_mode:
            continue
            
        file_path = os.path.join(input_dir, file_name)
        with open(file_path, "r", encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    
                    if 'meta_data' not in data or dimension_key not in data['meta_data'] or 'status' not in data:
                        continue
                    
                    is_correct = 1 if data["status"] == "correct" else 0
                    dim_values = data['meta_data'][dimension_key]
                    
                    if isinstance(dim_values, dict) and 'choices' in dim_values:
                        dim_values = dim_values['choices']
                    elif isinstance(dim_values, str):
                        dim_values = [dim_values]

                    if isinstance(dim_values, list):
                        overall_counted_this_line = False
                        for value in dim_values:
                            mapped_value = category_mapping.get(value, value) if category_mapping else value
                            
                            if mapped_value in categories:
                                model_scores[model_name][mapped_value]['correct'] += is_correct
                                model_scores[model_name][mapped_value]['total'] += 1
                                if not overall_counted_this_line:
                                    model_scores[model_name]['Overall']['correct'] += is_correct
                                    model_scores[model_name]['Overall']['total'] += 1
                                    overall_counted_this_line = True
                                    
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue

    results_table = PrettyTable()
    results_table.field_names = table_headers
    if table_headers:
        results_table.align[table_headers[0]] = "l"

    model_accuracies = []
    for model_name, scores in model_scores.items():
        overall_total = scores['Overall']['total']
        if overall_total > 0:
            accuracy = scores['Overall']['correct'] / overall_total
        else:
            accuracy = 0
        model_accuracies.append((model_name, accuracy))

    sorted_models = sorted(model_accuracies, key=lambda item: item[1])

    for model_name, _ in sorted_models:
        row = [model_name]
        
        overall_correct = model_scores[model_name]['Overall']['correct']
        overall_total = model_scores[model_name]['Overall']['total']
        row.append(f"{(overall_correct / overall_total if overall_total > 0 else 0):.2%}")

        for category in categories:
            cat_correct = model_scores[model_name][category]['correct']
            cat_total = model_scores[model_name][category]['total']
            row.append(f"{(cat_correct / cat_total if cat_total > 0 else 0):.2%}")
        
        results_table.add_row(row)
        
    return results_table.get_string()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate BiMeRe-Bench results by different dimensions.")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing .jsonl files with evaluation status.')
    parser.add_argument('--save_to', type=str, required=True, help='File path to save the final result table.')
    args = parser.parse_args()

    MODES_TO_EVALUATE = ['none', 'cot', 'domain', 'emotion', 'one-shot', 'two-shot', 'three-shot']

    domain_cats = ["生活", "艺术", "社会", "政治", "环境", "中华传统文化"]
    domain_headers = ["Model", "Overall"] + domain_cats
    
    emotion_cats = ["积极", "消极", "中性"]
    emotion_headers = ["Model", "Overall"] + emotion_cats

    type_cats = ["插画(Illustration)", "绘画(Painting)", "海报(Poster)", "单格漫画(Single-panel Comic)", "多格漫画(Multi-panel Comic)", "Meme"]
    type_headers = ["Model", "Overall"] + type_cats
    
    type_mapping = {
        "梗图(Meme)": "Meme"
    }
    
    difficulty_cats = ["简单", "中等", "困难"]
    difficulty_headers = ["Model", "Overall"] + difficulty_cats

    print(f"Reading data from directory '{args.input_dir}'...")
    print(f"Evaluation modes: {MODES_TO_EVALUATE}")
    
    full_report_parts = []
    for mode in MODES_TO_EVALUATE:
        print(f"\n--- Processing Mode: {mode} ---")
        
        mode_header = f"==================== Analysis for Mode: {mode} ===================="
        
        domain_t = "---------- Domain Analysis ----------\n" + evaluate_by_dimension(args.input_dir, "domain", domain_cats, domain_headers, filter_mode=mode)
        emotion_t = "---------- Emotion Analysis ----------\n" + evaluate_by_dimension(args.input_dir, "emotion", emotion_cats, emotion_headers, filter_mode=mode)
        type_t = "---------- Image Type Analysis ----------\n" + evaluate_by_dimension(args.input_dir, "image_type", type_cats, type_headers, filter_mode=mode, category_mapping=type_mapping)
        diff_t = "---------- Difficulty Analysis ----------\n" + evaluate_by_dimension(args.input_dir, "difficulty", difficulty_cats, difficulty_headers, filter_mode=mode)
        
        mode_report = f"{mode_header}\n\n{domain_t}\n\n{emotion_t}\n\n{type_t}\n\n{diff_t}"
        full_report_parts.append(mode_report)

    full_report = "\n\n".join(full_report_parts)
    
    with open(args.save_to, "w", encoding='utf-8') as f:
        f.write(full_report)

    print(f"\nEvaluation results successfully saved to: {args.save_to}")