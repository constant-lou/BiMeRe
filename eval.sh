# !/bin/bash

# Set model list
MODELS=("InternVL2-8B" "InternVL2_5-8B" "Qwen-VL-Chat")

# Set evaluation modes
MODES=("none" "cot" "domain" "emotion" "one-shot" "two-shot" "three-shot")

# Create result directories
# mkdir -p my_results
# mkdir -p my_results_with_status

# Process inference and evaluation for each model
for model in "${MODELS[@]}"; do
    echo "Processing model: $model"
    
    # Run inference for each mode
    for mode in "${MODES[@]}"; do
        echo "Running inference for mode: $mode"

        # English dataset testing
        if [ "$model" = "Qwen-VL-Chat" ] || [ "$model" = "idefics2-8b" ]; then
            # For Qwen-VL-Chat model, do not use --use_accel
            python ./src/infer/infer.py --config ./src/config/config_bimere.yaml --split E_metaphors --mode "$mode" --model_name "$model" --output_dir ./results_en/ --batch_size 1
        else
            # For other models, use --use_accel normally
            python ./src/infer/infer.py --config ./src/config/config_bimere.yaml --split E_metaphors --mode "$mode" --model_name "$model" --output_dir ./results_en/ --batch_size 1 --use_accel
        fi
        
        # Chinese dataset testing
        if [ "$model" = "Qwen-VL-Chat" ] || [ "$model" = "idefics2-8b" ]; then
            # For Qwen-VL-Chat model, do not use --use_accel
            python ./src/infer/infer.py --config ./src/config/config_bimere.yaml --split C_metaphors --mode "$mode" --model_name "$model" --output_dir ./results_zh/ --batch_size 1
        else
            # For other models, use --use_accel normally
            python ./src/infer/infer.py --config ./src/config/config_bimere.yaml --split C_metaphors --mode "$mode" --model_name "$model" --output_dir ./results_zh/ --batch_size 1 --use_accel
        fi
    done

    # Run evaluation
    python ./src/eval_bimere.py --evaluate_all --output_dir "./results_cn" --save_dir "./results_zh_with_status"
    python ./src/eval_bimere_sub.py --input_dir "./results_cn_with_status" --save_to "./results_zh_table.txt"

done

# Run overall evaluation at the end
# echo "Running overall evaluation..."
# python eval.py --evaluate_all --output_dir ./results --save_dir ./results_with_status

# echo "All evaluations completed!"
