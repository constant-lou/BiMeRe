#!/bin/bash

set -x

cd /path/to/BiMeRe-Bench/src
export PYTHONPATH=$(pwd)

python infer/infer.py --config config/config_bimere.yaml --split E_metaphors --mode one cot domain emotion rhetoric one-shot two-shot three-shot --model_name gpt4o --output_dir results_bimere --num_workers 8