# BiMeRe: Bimodal Metaphor Reasoning Benchmark

## Project Introduction

BiMeRe (Bimodal Metaphor Reasoning) is a benchmark project specifically designed to evaluate the performance of multimodal large language models on metaphor reasoning tasks. The project combines image and text information to test models' understanding and reasoning capabilities of metaphorical meanings.

## Project Features

- **Bimodal Input**: Combines image and text information for reasoning
- **Multiple Reasoning Modes**: Supports various reasoning modes including none, cot, domain, emotion, one-shot, two-shot, three-shot
- **Multi-dimensional Evaluation**: Evaluates model performance from multiple dimensions including image type, difficulty level, domain, and emotion
- **Multi-model Support**: Supports various mainstream multimodal models including GPT-4o, Claude, Qwen-VL, InternVL, etc.
- **Bilingual Datasets**: Provides both Chinese (C_metaphors_cot.json) and English (E_metaphors_cot.json) datasets

## Project Structure

```
BiMeRe/
├── assets/                    # Project resources
│   ├── Example.pdf           # Example documentation
│   ├── Error.pdf             # Error analysis documentation
│   ├── Error type distribution.pdf  # Error type distribution
│   └── Performance compairson.png   # Performance comparison chart
├── data/                     # Datasets
│   ├── C_metaphors_cot.json  # Chinese metaphor dataset
│   └── E_metaphors_cot.json  # English metaphor dataset
├── images/                   # Image data
│   ├── Cimages/             # Images for Chinese dataset
│   └── Eimages/             # Images for English dataset
├── results_zh/              # Chinese results output directory
├── results_zh_with_status/  # Chinese results with status directory
├── src/                     # Source code
│   ├── config/              # Configuration files
│   ├── infer/               # Inference module
│   ├── scripts/             # Running scripts
│   └── utils/               # Utility functions
├── eval.sh                  # Evaluation script
└── generate_rationale.py    # Rationale generation script
```

## Dataset Description

### Chinese Dataset (C_metaphors_cot.json)
- **Size**: 5.6MB
- **Content**: Chinese metaphor reasoning questions with images and text descriptions
- **Features**: Covers domains including life, art, society, politics, environment, and Chinese traditional culture

### English Dataset (E_metaphors_cot.json)
- **Size**: 9.1MB
- **Content**: English metaphor reasoning questions with images and text descriptions
- **Features**: Covers domains including society, psychology, life, art, environment, etc.

## Supported Models

The project supports various mainstream multimodal models:

- **GPT Series**: GPT-4o
- **Claude Series**: Claude-3-7-sonnet
- **Domestic Models**: Qwen-VL-Chat, Yi-VL-6B/34B, InternVL2/3 series
- **Open Source Models**: LLaVA, IDEFICS2, etc.

## Reasoning Modes

The project supports the following reasoning modes:

1. **none**: No special prompts
2. **cot**: Chain-of-thought reasoning
3. **domain**: Domain-specific reasoning
4. **emotion**: Emotion analysis reasoning
5. **one-shot**: One-shot learning
6. **two-shot**: Two-shot learning
7. **three-shot**: Three-shot learning

## Evaluation Dimensions

### Image Type Evaluation
- Illustration
- Painting
- Poster
- Single-panel Comic
- Multi-panel Comic
- Meme

### Difficulty Level Evaluation
- Easy
- Middle
- Hard

### Domain Evaluation
- Life
- Art
- Society
- Politics
- Environment
- Chinese Traditional Culture

### Emotion Evaluation
- Positive
- Negative
- Neutral

## Quick Start

### Requirements

- Python 3.8+
- Related dependencies (according to specific model requirements)

### Install Dependencies

```bash
# Install basic dependencies
pip install -r requirements.txt

# Install specific dependencies based on the model used
# For example, using InternVL series models
pip install lmdeploy
```

### Run Inference

1. **Using Evaluation Script**:
```bash
# Run complete evaluation pipeline
bash eval.sh
```

2. **Manual Inference**:
```bash
# Run inference for specific model
python ./src/infer/infer.py \
    --config ./src/config/config_bimere.yaml \
    --split E_metaphors \
    --mode cot \
    --model_name Qwen-VL-Chat \
    --output_dir ./results_en/ \
    --batch_size 1
```

3. **Generate Rationale**:
```bash
# Use generate_rationale.py to generate detailed reasoning process
python generate_rationale.py \
    --input data/E_metaphors_cot.json \
    --output results_with_rationale.json \
    --model gpt-4o \
    --workers 3
```

### Run Evaluation

```bash
# Evaluate Chinese results
python ./src/eval_zh.py --evaluate_all --output_dir ./results_zh --save_dir ./results_zh_with_status

# Evaluate English results
python ./src/eval_en.py --evaluate_all --output_dir ./results_en --save_dir ./results_en_with_status
```

## Configuration

### Model Configuration

Configure model parameters in `src/config/config_bimere.yaml`:

```yaml
models:
  GPT-4o:
    api_base: "your_api_base"
    api_key: "your_api_key"
  
  Qwen-VL-Chat:
    api_base: "your_qwen_api_base"
    api_key: "your_qwen_api_key"
  
  InternVL2-8B:
    model_path: "path_to_model"
    device: "cuda"
```

### Inference Configuration

- `batch_size`: Batch size
- `num_workers`: Number of parallel workers
- `use_accel`: Whether to use accelerated inference
- `infer_limit`: Inference limit

## Script Description

### eval.sh Script Functions

`eval.sh` is the main evaluation script with the following functions:

- **Multi-model Support**: Supports models like InternVL2-8B, InternVL2_5-8B, Qwen-VL-Chat, etc.
- **Multi-mode Evaluation**: Automatically runs all reasoning modes
- **Bilingual Datasets**: Processes both Chinese and English datasets
- **Smart Acceleration**: Automatically chooses whether to use accelerated inference based on model type
- **Auto Evaluation**: Automatically runs evaluation scripts to generate result reports

### Script Usage Example

```bash
# Run complete evaluation
bash eval.sh

# Modify model list
# Edit the MODELS array in eval.sh file
MODELS=("your_model_1" "your_model_2")

# Modify reasoning modes
# Edit the MODES array in eval.sh file
MODES=("none" "cot")
```

## Result Analysis

### Output Format

Inference results are saved in JSONL format, including:
- Original question information
- Model answers
- Reasoning process (if enabled)
- Error information (if any)

### Evaluation Report

Evaluation scripts generate detailed performance reports, including:
- Accuracy statistics for each dimension
- Performance comparison of different reasoning modes
- Error type distribution analysis

## Notes

1. **API Keys**: API keys need to be configured when using commercial API models
2. **Model Paths**: Model paths need to be correctly configured when using local models
3. **Hardware Requirements**: Large model inference requires sufficient GPU memory
4. **Data Paths**: Ensure image file paths are correctly configured

## Contributing

Welcome to submit Issues and Pull Requests to improve the project. Before submitting code, please ensure:

1. Code follows project standards
2. Add necessary documentation
3. Pass relevant tests

## License

This project is licensed under the MIT License, see LICENSE file for details.

## Contact

For questions or suggestions, please contact through:

- **GitHub**: [BiMeRe Repository](https://github.com/constant-lou/BiMeRe)
- **Email**: 2077658501@qq.com
- **Submit GitHub Issue**: [Issues](https://github.com/constant-lou/BiMeRe/issues)

## Acknowledgments

Thanks to all researchers and developers who contributed to this project. 