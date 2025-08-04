#!/usr/bin/env python3
"""
generate_rationale.py

Generate concise four-part rationales for questions in a JSON file using a custom requests-based API call,
now with multimodal support, incremental saving, and resume capability.
The output rationale will not include the 'explanation_from_meta_data' field in the question object.
Usage: python generate_rationale.py --input data.json --output completed.json [--model gpt-4o] [--workers 3]
python generate_rationale.py  --input /path/to/input.json --output /path/to/output.json
"""

import os
import sys
import json
import argparse
import logging
import time
import requests 
import base64   
import mimetypes 
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional
from pathlib import Path

DEFAULT_MODEL_NAME_FOR_PAYLOAD = "gpt-4o" 
DEFAULT_WORKERS = 3
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS_FOR_RATIONALE = 1024 
LOG_LEVEL = logging.INFO

OPENAI_API_KEY = "sk-your-api-key-here"
OPENAI_API_BASE_URL = "https://api.example.com/v1/chat/completions"

PROMPT_TEMPLATE = """
Please generate a rationale based on the following input (including text description and provided images), divided into four parts, each part completed in one or two short sentences, with a total length controlled to 5-8 sentences. When generating the rationale field, do not break it into sections, and finally, therefore the correct option is {{answer}}.

**Provided Image Information**: {{image_references}}

1. Image Description  
   Combined with the provided image, briefly describe the composition, main character and other characters' actions, and scene atmosphere.

2. Metaphorical Meaning  
   Combined with the image content, explain the symbolic meaning of characters or elements in the scene, and how they suggest deeper social or psychological themes.

3. Option Analysis  
   For each option A, B, C... combined with image and text information, explain why they do not match the scene or metaphor, and finally highlight the correct option:  
   A. …  
   B. …  
   …  
   {{correct_option}}. …

4. Conclusion  
   Use one sentence to explain why the answer is {{answer}} ({{correct_option}}).

Note: Do not copy the explanation text directly, but you can draw on its key information. Be sure to analyze in combination with the image.

{data}"""

logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def convert_local_path_to_data_uri(image_path_str: str, base_input_dir: Optional[Path] = None) -> Optional[str]:
    """Converts a local image file to a base64 data URI."""
    if base_input_dir:
        image_path = (base_input_dir / image_path_str).resolve()
    else:
        image_path = Path(image_path_str).resolve()
    
    if not image_path.is_file():
        logger.error(f"Image file not found or is not a file: {image_path}")
        return None
    
    mime_type, _ = mimetypes.guess_type(image_path.name)
    if not mime_type or not mime_type.startswith("image"):
        logger.warning(f"Could not determine valid image MIME type for: {image_path}. Defaulting to 'image/jpeg'.")
        mime_type = "image/jpeg"
        
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:{mime_type};base64,{encoded_string}"
    except Exception as e:
        logger.error(f"Error reading or encoding image {image_path}: {e}")
        return None

class RationaleGenerator:
    def __init__(self, api_key: str, base_url: str, model_for_payload: str, 
                 temperature: float = DEFAULT_TEMPERATURE, 
                 max_tokens: int = DEFAULT_MAX_TOKENS_FOR_RATIONALE):
        
        if not api_key: raise ValueError("API_KEY cannot be empty.")
        if not base_url: raise ValueError("API_BASE_URL cannot be empty.")
        if not model_for_payload: raise ValueError("MODEL_FOR_PAYLOAD cannot be empty.")

        self.api_key = api_key
        self.base_url = base_url
        self.model_for_payload = model_for_payload
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        logger.info(f"RationaleGenerator initialized for model: {self.model_for_payload} at URL: {self.base_url}")

    def generate_rationale(self, item_data_for_prompt: Dict[str, Any], local_image_path: str, base_input_dir: Optional[Path] = None, timeout: int = 300, verbose: bool = False) -> str:
        item_id_for_log = item_data_for_prompt.get('id', 'unknown_item')
        
        actual_image_path_str = str((base_input_dir / local_image_path).resolve()) if base_input_dir and local_image_path else local_image_path
        image_reference_text = f"Please refer to the image related to this question (from path: {local_image_path}, this image has been provided as visual input) for analysis."
        
        format_dict = {
            "image_references": image_reference_text,
            "correct_option": str(item_data_for_prompt.get("correct_option", "N/A")),
            "answer": str(item_data_for_prompt.get("answer", "N/A")),
            "data": json.dumps(item_data_for_prompt, ensure_ascii=False, indent=2)
        }
        
        try:
            text_prompt_content = PROMPT_TEMPLATE.format(**format_dict)
        except KeyError as e:
            logger.error(f"Missing key for PROMPT_TEMPLATE formatting for item_id {item_id_for_log}: {e}. Available keys: {list(format_dict.keys())}. Data keys: {list(item_data_for_prompt.keys())}")
            return ""

        image_data_uri = None
        if local_image_path:
             image_data_uri = convert_local_path_to_data_uri(local_image_path, base_input_dir)
        
        system_message_content = "You are an expert in generating concise rationales according to the provided template, text, and image data."
        
        if not image_data_uri:
            if local_image_path:
                logger.warning(f"Could not get image data URI for {actual_image_path_str}. Proceeding with text-only for item_id: {item_id_for_log}")
            system_message_content = "You are an expert in generating concise rationales... (Image was not available or path not provided for this item)."
            messages_payload = [
                {"role": "system", "content": system_message_content},
                {"role": "user", "content": text_prompt_content}
            ]
        else:
            messages_payload = [
                {"role": "system", "content": system_message_content},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_prompt_content},
                        {"type": "image_url", "image_url": {"url": image_data_uri}}
                    ]
                }
            ]

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.model_for_payload,
            "messages": messages_payload,
            "max_tokens": self.max_tokens,
        }

        if verbose:
            logger.debug(f"[DEBUG] Request to {self.base_url} for item_id: {item_id_for_log}")
            logger.debug(f"[DEBUG] Model: {self.model_for_payload}")
            user_message_entry = next((msg for msg in messages_payload if msg["role"] == "user"), None)
            if user_message_entry:
                user_content = user_message_entry.get("content")
                if isinstance(user_content, list): 
                    text_part = next((item.get("text") for item in user_content if item.get("type") == "text"), "")
                    logger.debug(f"[DEBUG] User text prompt (first 200 chars): {text_part[:200]}...")
                    logger.debug(f"[DEBUG] Image data URI included: {bool(image_data_uri)}")
                elif isinstance(user_content, str): 
                    logger.debug(f"[DEBUG] User text prompt (first 200 chars): {user_content[:200]}...")
        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            response_json = response.json()

            if "choices" in response_json and response_json["choices"]:
                first_choice = response_json["choices"][0]
                if "message" in first_choice and "content" in first_choice["message"]:
                    rationale = first_choice["message"]["content"].strip()
                    logger.debug(f"Successfully generated rationale for item_id: {item_id_for_log}")
                    return rationale
            logger.error(f"API response for item_id {item_id_for_log} missing expected structure. Response: {json.dumps(response_json, ensure_ascii=False, indent=2)}")
            return ""
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP Error for item_id {item_id_for_log}: {e}")
            response_text = e.response.text if e.response is not None else "No response text available"
            logger.error(f"Response text for item_id {item_id_for_log}: {response_text}")
            return ""
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for item_id {item_id_for_log}: {e}")
            return ""
        except Exception as e:
            logger.error(f"Unexpected error for rationale generation for item_id {item_id_for_log}: {type(e).__name__} - {e}")
            return ""

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate rationales with multimodal support, incremental saving, and resume capability.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", required=True, help="Input JSON file path")
    parser.add_argument("--output", "-o", required=True, help="Output JSON file path")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL_NAME_FOR_PAYLOAD,
                        help=f"Model name for API payload (default: {DEFAULT_MODEL_NAME_FOR_PAYLOAD})")
    parser.add_argument("--workers", "-w", type=int, default=DEFAULT_WORKERS, help="Number of concurrent workers")
    parser.add_argument("--temperature", "-t", type=float, default=DEFAULT_TEMPERATURE, help="Temperature for text generation")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS_FOR_RATIONALE, help="Max tokens for generated rationale")
    parser.add_argument("--dry-run", action="store_true", help="Dry run")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    return parser.parse_args()

def validate_files(input_path: str, output_path: str) -> None:
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

def load_json_data(file_path_str: str) -> List[Dict[str, Any]]:
    file_path = Path(file_path_str)
    if not file_path.exists():
        return [] 
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            logger.error(f"Data in {file_path_str} is not a list.")
            return [] 
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in {file_path_str}. Error: {e}.")
        return [] 
    except Exception as e:
        logger.error(f"Error loading JSON from {file_path_str}: {e}.")
        return []

def save_json_data(data: List[Dict[str, Any]], output_path_str: str) -> None:
    output_path = Path(output_path_str)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.debug(f"Data successfully saved to {output_path_str}")
    except Exception as e:
        logger.error(f"Failed to write output file {output_path_str}: {e}")

def process_single_question(args_tuple: tuple) -> Optional[tuple]:
    item_idx, question_idx, question_dict, generator, item_explanation, verbose, local_image_path, base_input_dir_for_images = args_tuple
    
    data_for_prompt = question_dict.copy()
    if item_explanation:
        data_for_prompt["explanation_from_meta_data"] = item_explanation 
    
    rationale = generator.generate_rationale(data_for_prompt, local_image_path, base_input_dir_for_images, verbose=verbose)
    
    if rationale: 
        return item_idx, question_idx, rationale
    return None 

def process_questions_parallel(input_data: List[Dict[str, Any]], 
                               output_data_on_disk: List[Dict[str, Any]], 
                               generator: RationaleGenerator, 
                               output_file_path: str,
                               base_input_dir_for_images: Path,
                               max_workers: int = DEFAULT_WORKERS, 
                               verbose_flag: bool = False) -> List[Dict[str, Any]]:
    
    processed_rationales_from_disk = {}
    if output_data_on_disk:
        for item in output_data_on_disk:
            for q_idx, question in enumerate(item.get("questions", [])):
                q_id = question.get("id")
                if q_id and question.get("rationale", "").strip():
                    processed_rationales_from_disk[q_id] = question["rationale"] 

    for item_idx, item_data_master in enumerate(input_data):
        for q_idx, question_master in enumerate(item_data_master.get("questions", [])):
            q_id = question_master.get("id")
            if q_id in processed_rationales_from_disk:
                input_data[item_idx]["questions"][q_idx]["rationale"] = processed_rationales_from_disk[q_id]

    tasks_to_submit = []
    for item_idx, item_data in enumerate(input_data): 
        item_explanation = item_data.get("meta_data", {}).get("explanation", "")
        local_image_path_rel = item_data.get("local_path", "")

        questions_in_item = item_data.get("questions", [])
        for question_idx, question_dict in enumerate(questions_in_item):
            current_rationale_in_mem = question_dict.get("rationale", "")
            if not (isinstance(current_rationale_in_mem, str) and current_rationale_in_mem.strip()):
                task_args_for_func = (
                    item_idx, question_idx, question_dict.copy(), generator, 
                    item_explanation, verbose_flag, local_image_path_rel, base_input_dir_for_images
                )
                full_task_context = task_args_for_func + (output_file_path, input_data) 
                tasks_to_submit.append(full_task_context)

    if not tasks_to_submit:
        logger.info("All questions already have non-empty rationales. Nothing new to process.")
        return input_data
    
    logger.info(f"Processing {len(tasks_to_submit)} new questions for rationale generation using {max_workers} workers...")
    
    completed_count = 0
    total_tasks = len(tasks_to_submit)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task_info = {
            executor.submit(process_single_question, full_context_args[:8]): full_context_args 
            for full_context_args in tasks_to_submit
        }
        
        for future in as_completed(future_to_task_info):
            original_task_args = future_to_task_info[future]
            
            logger.debug(f"Attempting to unpack original_task_args. Length: {len(original_task_args)}. Content (first few elements): {original_task_args[:3]}...")

            try:
                task_item_idx, task_question_idx, task_original_question_dict, _, _, _, _, _, task_output_path, _ = original_task_args
            except ValueError as e:
                logger.error(f"ValueError during unpacking original_task_args: {e}. Args were: {original_task_args}")
                continue 

            try:
                result = future.result() 
                if result: 
                    returned_item_idx, returned_question_idx, rationale = result
                    
                    input_data[returned_item_idx]['questions'][returned_question_idx]["rationale"] = rationale
                    
                    completed_count += 1
                    q_id_log = input_data[returned_item_idx]['questions'][returned_question_idx].get('id', f'item_{returned_item_idx}_q_{returned_question_idx}')
                    
                    if rationale:
                        logger.info(f"[{completed_count}/{total_tasks}] Completed rationale for question id={q_id_log}")
                    else:
                        logger.warning(f"[{completed_count}/{total_tasks}] Empty rationale for question id={q_id_log} (successful call, but no content).")

                    save_json_data(input_data, output_file_path) 
                    logger.info(f"Progress saved to {output_file_path} after processing question id={q_id_log}")
                else: 
                    failed_q_id_log = task_original_question_dict.get('id', f'item_{task_item_idx}_q_{task_question_idx}')
                    logger.warning(f"Rationale generation returned None (likely failed inside API call) for question id={failed_q_id_log}")

            except Exception as e:
                failed_q_id_log = task_original_question_dict.get('id', f'item_{task_item_idx}_q_{task_question_idx}')
                logger.error(f"Exception processing result for question id={failed_q_id_log}: {type(e).__name__} - {str(e)}")
    return input_data

def print_dry_run_info(input_data: List[Dict[str, Any]], output_data_on_disk: List[Dict[str, Any]] ) -> None:
    processed_ids = set()
    if output_data_on_disk:
        for item in output_data_on_disk:
            for q in item.get("questions", []):
                if q.get("id") and q.get("rationale", "").strip():
                    processed_ids.add(q["id"])

    total_items = len(input_data)
    grand_total_questions = 0
    questions_needing_rationale_count = 0
    
    questions_to_process_details = []

    for item_idx, item_data in enumerate(input_data):
        questions_in_item = item_data.get("questions", [])
        grand_total_questions += len(questions_in_item)
        item_path_for_log = item_data.get('local_path', f'item_idx_{item_idx}')
        for q_idx, question_dict in enumerate(questions_in_item):
            q_id = question_dict.get("id")
            current_rationale_in_mem = question_dict.get("rationale", "")
            is_in_mem_processed = isinstance(current_rationale_in_mem, str) and current_rationale_in_mem.strip()

            if not (q_id in processed_ids or is_in_mem_processed) :
                questions_needing_rationale_count += 1
                detail = f"  - Item Path: {item_path_for_log}, Question ID: {q_id}"
                questions_to_process_details.append(detail)
                
    questions_with_rationale_count = grand_total_questions - questions_needing_rationale_count
                
    print(f"\n=== DRY RUN MODE ===")
    print(f"Total items in input JSON: {total_items}")
    print(f"Grand total questions across all items: {grand_total_questions}")
    print(f"Questions already processed (in output or non-empty in input): {questions_with_rationale_count}")
    print(f"Questions needing new rationale: {questions_needing_rationale_count}")

    if questions_to_process_details:
        print(f"\nQuestions that would be processed (up to 5 shown):")
        for detail_line in questions_to_process_details[:5]: print(detail_line)
        if len(questions_to_process_details) > 5: print(f"  ... and {len(questions_to_process_details) - 5} more")
    else:
        print("\nNo new questions to process.")
    print("===================\n")

def main() -> None:
    args = parse_arguments()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled.")
    else:
        logging.getLogger().setLevel(LOG_LEVEL)

    try:
        validate_files(args.input, args.output)
        
        logger.info(f"Loading input data from {args.input}")
        input_data_master = load_json_data(args.input)
        if not input_data_master:
            logger.error(f"No data loaded from input file {args.input}. Exiting.")
            sys.exit(1)

        logger.info(f"Attempting to load previously processed data from {args.output} for resume capability.")
        output_data_for_resume = load_json_data(args.output)
        
        all_data = input_data_master 

        base_input_dir = Path(args.input).parent

        if args.dry_run:
            print_dry_run_info(all_data, output_data_for_resume)
            return
            
        generator = RationaleGenerator(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_API_BASE_URL,
            model_for_payload=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
        
        start_time = time.time()
        all_data = process_questions_parallel(
            all_data, 
            output_data_for_resume,
            generator, 
            args.output,
            base_input_dir,
            args.workers, 
            verbose_flag=args.verbose
        )
        elapsed_time = time.time() - start_time
        
        logger.info(f"Final save of all data to {args.output}")
        save_json_data(all_data, args.output)
        
        logger.info(f"Processing completed in {elapsed_time:.2f} seconds")
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user. Partially processed data should be in the output file.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred in main: {type(e).__name__} - {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()