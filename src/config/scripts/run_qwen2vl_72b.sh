#!/bin/bash

set -x

cd /path/to/BiMeRe-Bench/src
export PYTHONPATH=$(pwd)

python infer/infer.py --config config/config_bimere.yaml --split E_metaphors --mode none cot domain emotion rhetoric --model_name Qwen2-VL-72B --output_dir results_bimere --batch_size 100 --use_accel

