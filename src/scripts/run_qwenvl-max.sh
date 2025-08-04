#!/bin/bash

set -x

cd /path/to/BiMeRe/src
export PYTHONPATH=$(pwd)

python infer/infer.py --config config/config_bimere.yaml --split E_metaphors --mode none cot domain emotion rhetoric --model_name qwen-vl-max --output_dir results_bimere --num_workers 8
