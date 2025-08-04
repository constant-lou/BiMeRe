import json
import os
import argparse
from prettytable import PrettyTable
from collections import defaultdict

def evaluate_by_category(output_dir, category_json_key, category_labels, valid_modes):
    """
    Generic evaluation function that evaluates model performance based on specified categories and modes.
    Returns a dict where key is mode and value is the corresponding PrettyTable string.
    """
    tables_by_mode = {}
    results_by_mode = defaultdict(list)

    try:
        files = sorted(os.listdir(output_dir))
    except FileNotFoundError:
        print(f"Error: Directory not found at '{output_dir}'")
        return {}

    for file_name in files:
        if not file_name.endswith('.jsonl'):
            continue
        
        base_name = file_name.rsplit('.', 1)[0]
        try:
            parts = base_name.rsplit('_', 1)
            if len(parts) != 2:
                continue
            model_and_task, mode = parts[0], parts[1]

            if mode not in valid_modes:
                continue
            
            model_name = model_and_task.rsplit('_', 2)[0]
        except (IndexError, ValueError):
            continue

        score_dict = {label: [] for label in category_labels}
        overall_scores = []
        
        file_path = os.path.join(output_dir, file_name)
        with open(file_path, "r", encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "status" not in data or "meta_data" not in data:
                        continue
                    if category_json_key not in data["meta_data"]:
                        continue
                    is_correct = 1 if data["status"] == "correct" else 0
                    overall_scores.append(is_correct)
                    category_value = data['meta_data'][category_json_key]
                    if category_value in score_dict:
                        score_dict[category_value].append(is_correct)
                except Exception:
                    continue

        if not overall_scores:
            continue

        overall_acc = sum(overall_scores) / len(overall_scores)
        acc_by_category = {
            label: (sum(score_dict[label]) / len(score_dict[label]) if score_dict[label] else 0.0)
            for label in category_labels
        }

        row_data = [model_name, mode, f"{overall_acc:.2%}"] + [f"{acc_by_category[label]:.2%}" for label in category_labels]
        results_by_mode[mode].append(row_data)

    for mode, rows in results_by_mode.items():
        sorted_rows = sorted(rows, key=lambda x: float(x[2].strip('%')))
        table = PrettyTable()
        table.field_names = ["Model", "Mode", "Overall"] + category_labels
        for row in sorted_rows:
            table.add_row(row)
        tables_by_mode[mode] = table.get_string()

    return tables_by_mode


def main():
    parser = argparse.ArgumentParser(description="Analyze model performance from .jsonl result files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory containing the .jsonl result files.")
    parser.add_argument("--save_to", type=str, required=True, help="Path to save the final analysis report.")
    args = parser.parse_args()

    IMAGE_TYPES = ["Single-panel Comic", "Multi-panel Comic", "Illustration", "Meme", "Poster", "Painting", "Logo"]
    DIFFICULTY_LEVELS = ["Easy", "Middle", "Hard"]
    DOMAINS = ["Society", "Psychology", "Life", "Art", "Others", "Environment"]
    EMOTIONS = ["Neutral", "Negative", "Positive"]

    VALID_MODES = ['none', 'cot', 'domain','emotion','one-shot', 'two-shot', 'three-shot']

    print(f"Starting analysis on directory: '{args.output_dir}'...")
    print(f"Processing files with modes: {VALID_MODES}")

    type_tables = evaluate_by_category(args.output_dir, "image_type", IMAGE_TYPES, VALID_MODES)
    difficulty_tables = evaluate_by_category(args.output_dir, "difficulty", DIFFICULTY_LEVELS, VALID_MODES)
    domain_tables = evaluate_by_category(args.output_dir, "domain", DOMAINS, VALID_MODES)
    emotion_tables = evaluate_by_category(args.output_dir, "emotion", EMOTIONS, VALID_MODES)

    def format_section(title, tables_dict):
        lines = [f"--- {title} ---"]
        for mode in VALID_MODES:
            if mode in tables_dict:
                lines.append(f"\n>>> Mode: {mode}")
                lines.append(tables_dict[mode])
        return "\n".join(lines)

    full_report = f"""Model Performance Analysis Report
=================================

{format_section("Image Type Evaluation", type_tables)}

{format_section("Difficulty Evaluation", difficulty_tables)}

{format_section("Domain Evaluation", domain_tables)}

{format_section("Emotion Evaluation", emotion_tables)}
"""
    with open(args.save_to, "w", encoding='utf-8') as f:
        f.write(full_report)

    print(f"Analysis complete. Report saved to '{args.save_to}'.")
    

if __name__ == "__main__":
    main()
