#!/bin/bash

set -x

cd /path/to/BiMeRe-Bench/src
export PYTHONPATH=$(pwd)

python infer/infer.py --config config/config_bimere.yaml --split E_metaphors --mode none cot domain emotion rhetoric one-shot two-shot three-shot --model_name claude-3-5-sonnet-20240620 --output_dir results_bimere --num_workers 16