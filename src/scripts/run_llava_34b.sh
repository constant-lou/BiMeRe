#!/bin/bash

set -x

cd /path/to/BiMeRe-Bench/src
export PYTHONPATH=$(pwd)


python infer/infer.py --config config/config_bimere.yaml --split E_metaphors --mode none --model_name llava-v1.6-34b-hf --output_dir results_bimere --batch_size 1
sleep 5
python infer/infer.py --config config/config_bimere.yaml --split E_metaphors --mode cot --model_name llava-v1.6-34b-hf --output_dir results_bimere --batch_size 1
sleep 5
python infer/infer.py --config config/config_bimere.yaml --split E_metaphors --mode domain --model_name llava-v1.6-34b-hf --output_dir results_bimere --batch_size 1
sleep 5
python infer/infer.py --config config/config_bimere.yaml --split E_metaphors --mode emotion --model_name llava-v1.6-34b-hf --output_dir results_bimere --batch_size 1
sleep 5
python infer/infer.py --config config/config_bimere.yaml --split E_metaphors --mode rhetoric --model_name llava-v1.6-34b-hf --output_dir results_bimere --batch_size 1
sleep 5