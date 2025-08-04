# BiMeRe: Bimodal Metaphor Reasoning Benchmark

## 项目简介 / Project Introduction

BiMeRe (Bimodal Metaphor Reasoning) 是一个专门用于评估多模态大语言模型在隐喻推理任务上表现能力的基准测试项目。该项目结合了图像和文本信息，测试模型对隐喻含义的理解和推理能力。

BiMeRe (Bimodal Metaphor Reasoning) is a benchmark project specifically designed to evaluate the performance of multimodal large language models on metaphor reasoning tasks. The project combines image and text information to test models' understanding and reasoning capabilities of metaphorical meanings.

## 项目特点 / Project Features

- **双模态输入 / Bimodal Input**: 结合图像和文本信息进行推理 / Combines image and text information for reasoning
- **多种推理模式 / Multiple Reasoning Modes**: 支持none、cot、domain、emotion、one-shot、two-shot、three-shot等多种推理模式 / Supports various reasoning modes including none, cot, domain, emotion, one-shot, two-shot, three-shot
- **多维度评估 / Multi-dimensional Evaluation**: 从图像类型、难度等级、领域、情感等多个维度评估模型性能 / Evaluates model performance from multiple dimensions including image type, difficulty level, domain, and emotion
- **多模型支持 / Multi-model Support**: 支持多种主流多模态模型，包括GPT-4o、Claude、Qwen-VL、InternVL等 / Supports various mainstream multimodal models including GPT-4o, Claude, Qwen-VL, InternVL, etc.
- **中英文数据集 / Bilingual Datasets**: 提供中文(C_metaphors_cot.json)和英文(E_metaphors_cot.json)两个数据集 / Provides both Chinese (C_metaphors_cot.json) and English (E_metaphors_cot.json) datasets

## 项目结构 / Project Structure

```
BiMeRe/
├── assets/                    # 项目资源文件 / Project resources
│   ├── Example.pdf           # 示例说明文档 / Example documentation
│   ├── Error.pdf             # 错误分析文档 / Error analysis documentation
│   ├── Error type distribution.pdf  # 错误类型分布 / Error type distribution
│   └── Performance compairson.png   # 性能对比图 / Performance comparison chart
├── data/                     # 数据集 / Datasets
│   ├── C_metaphors_cot.json  # 中文隐喻数据集 / Chinese metaphor dataset
│   └── E_metaphors_cot.json  # 英文隐喻数据集 / English metaphor dataset
├── images/                   # 图像数据 / Image data
│   ├── Cimages/             # 中文数据集对应图像 / Images for Chinese dataset
│   └── Eimages/             # 英文数据集对应图像 / Images for English dataset
├── results_zh/              # 中文结果输出目录 / Chinese results output directory
├── results_zh_with_status/  # 带状态的中文结果目录 / Chinese results with status directory
├── src/                     # 源代码 / Source code
│   ├── config/              # 配置文件 / Configuration files
│   ├── infer/               # 推理模块 / Inference module
│   ├── scripts/             # 运行脚本 / Running scripts
│   └── utils/               # 工具函数 / Utility functions
├── eval.sh                  # 评估脚本 / Evaluation script
└── generate_rationale.py    # 推理生成脚本 / Rationale generation script
```

## 数据集说明 / Dataset Description

### 中文数据集 (C_metaphors_cot.json) / Chinese Dataset
- **大小 / Size**: 5.6MB
- **内容 / Content**: 中文隐喻推理题目，包含图像和文本描述 / Chinese metaphor reasoning questions with images and text descriptions
- **特点 / Features**: 涵盖生活、艺术、社会、政治、环境、中华传统文化等领域 / Covers domains including life, art, society, politics, environment, and Chinese traditional culture

### 英文数据集 (E_metaphors_cot.json) / English Dataset
- **大小 / Size**: 9.1MB
- **内容 / Content**: 英文隐喻推理题目，包含图像和文本描述 / English metaphor reasoning questions with images and text descriptions
- **特点 / Features**: 涵盖社会、心理学、生活、艺术、环境等多个领域 / Covers domains including society, psychology, life, art, environment, etc.

## 支持的模型 / Supported Models

项目支持多种主流多模态模型 / The project supports various mainstream multimodal models:

- **GPT系列 / GPT Series**: GPT-4o
- **Claude系列 / Claude Series**: Claude-3-7-sonnet
- **国产模型 / Domestic Models**: Qwen-VL-Chat, Yi-VL-6B/34B, InternVL2/3系列 / Qwen-VL-Chat, Yi-VL-6B/34B, InternVL2/3 series
- **开源模型 / Open Source Models**: LLaVA, IDEFICS2等 / LLaVA, IDEFICS2, etc.

## 推理模式 / Reasoning Modes

项目支持以下推理模式 / The project supports the following reasoning modes:

1. **none**: 无特殊提示 / No special prompts
2. **cot**: 思维链推理 / Chain-of-thought reasoning
3. **domain**: 领域特定推理 / Domain-specific reasoning
4. **emotion**: 情感分析推理 / Emotion analysis reasoning
5. **one-shot**: 单样本学习 / One-shot learning
6. **two-shot**: 双样本学习 / Two-shot learning
7. **three-shot**: 三样本学习 / Three-shot learning

## 评估维度 / Evaluation Dimensions

### 图像类型评估 / Image Type Evaluation
- 插画(Illustration)
- 绘画(Painting)
- 海报(Poster)
- 单格漫画(Single-panel Comic)
- 多格漫画(Multi-panel Comic)
- 梗图(Meme)

### 难度等级评估 / Difficulty Level Evaluation
- 简单(Easy)
- 中等(Middle)
- 困难(Hard)

### 领域评估 / Domain Evaluation
- 生活(Life)
- 艺术(Art)
- 社会(Society)
- 政治(Politics)
- 环境(Environment)
- 中华传统文化(Chinese Traditional Culture)

### 情感评估 / Emotion Evaluation
- 积极(Positive)
- 消极(Negative)
- 中性(Neutral)

## 快速开始 / Quick Start

### 环境要求 / Requirements

- Python 3.8+
- 相关依赖包（根据具体模型要求）/ Related dependencies (according to specific model requirements)

### 安装依赖 / Install Dependencies

```bash
# 安装基础依赖 / Install basic dependencies
pip install -r requirements.txt

# 根据使用的模型安装特定依赖 / Install specific dependencies based on the model used
# 例如使用InternVL系列模型 / For example, using InternVL series models
pip install lmdeploy
```

### 运行推理 / Run Inference

1. **使用评估脚本 / Using Evaluation Script**:
```bash
# 运行完整的评估流程 / Run complete evaluation pipeline
bash eval.sh
```

2. **手动运行推理 / Manual Inference**:
```bash
# 运行特定模型的推理 / Run inference for specific model
python ./src/infer/infer.py \
    --config ./src/config/config_bimere.yaml \
    --split E_metaphors \
    --mode cot \
    --model_name Qwen-VL-Chat \
    --output_dir ./results_en/ \
    --batch_size 1
```

3. **生成推理过程 / Generate Rationale**:
```bash
# 使用generate_rationale.py生成详细的推理过程 / Use generate_rationale.py to generate detailed reasoning process
python generate_rationale.py \
    --input data/E_metaphors_cot.json \
    --output results_with_rationale.json \
    --model gpt-4o \
    --workers 3
```

### 运行评估 / Run Evaluation

```bash
# 评估中文结果 / Evaluate Chinese results
python ./src/eval_zh.py --evaluate_all --output_dir ./results_zh --save_dir ./results_zh_with_status

# 评估英文结果 / Evaluate English results
python ./src/eval_en.py --evaluate_all --output_dir ./results_en --save_dir ./results_en_with_status
```

## 配置说明 / Configuration

### 模型配置 / Model Configuration

在 `src/config/config_bimere.yaml` 中配置模型参数 / Configure model parameters in `src/config/config_bimere.yaml`:

```yaml
models:
  GPT-4o:
    api_base: "your_api_base"
    api_key: "your_api_key"
  
  InternVL2-8B:
    model_path: "path_to_model"
    device: "cuda"
```

### 推理配置 / Inference Configuration

- `batch_size`: 批处理大小 / Batch size
- `num_workers`: 并行工作进程数 / Number of parallel workers
- `use_accel`: 是否使用加速推理 / Whether to use accelerated inference
- `infer_limit`: 推理数量限制 / Inference limit

## 脚本说明 / Script Description

### eval.sh 脚本功能 / eval.sh Script Functions

`eval.sh` 是主要的评估脚本，具有以下功能 / `eval.sh` is the main evaluation script with the following functions:

- **多模型支持 / Multi-model Support**: 支持InternVL2-8B、InternVL2_5-8B、Qwen-VL-Chat等模型 / Supports models like InternVL2-8B, InternVL2_5-8B, Qwen-VL-Chat, etc.
- **多模式评估 / Multi-mode Evaluation**: 自动运行所有推理模式 / Automatically runs all reasoning modes
- **中英文数据集 / Bilingual Datasets**: 同时处理中文和英文数据集 / Processes both Chinese and English datasets
- **智能加速 / Smart Acceleration**: 根据模型类型自动选择是否使用加速推理 / Automatically chooses whether to use accelerated inference based on model type
- **自动评估 / Auto Evaluation**: 自动运行评估脚本生成结果报告 / Automatically runs evaluation scripts to generate result reports

### 脚本使用示例 / Script Usage Example

```bash
# 运行完整评估 / Run complete evaluation
bash eval.sh

# 修改模型列表 / Modify model list
# 编辑eval.sh文件中的MODELS数组 / Edit the MODELS array in eval.sh file
MODELS=("your_model_1" "your_model_2")

# 修改推理模式 / Modify reasoning modes
# 编辑eval.sh文件中的MODES数组 / Edit the MODES array in eval.sh file
MODES=("none" "cot")
```

## 结果分析 / Result Analysis

### 输出格式 / Output Format

推理结果以JSONL格式保存，包含 / Inference results are saved in JSONL format, including:
- 原始问题信息 / Original question information
- 模型回答 / Model answers
- 推理过程（如果启用）/ Reasoning process (if enabled)
- 错误信息（如果有）/ Error information (if any)

### 评估报告 / Evaluation Report

评估脚本会生成详细的性能报告，包括 / Evaluation scripts generate detailed performance reports, including:
- 各维度的准确率统计 / Accuracy statistics for each dimension
- 不同推理模式的性能对比 / Performance comparison of different reasoning modes
- 错误类型分布分析 / Error type distribution analysis

## 注意事项 / Notes

1. **API密钥 / API Keys**: 使用商业API模型时需要配置相应的API密钥 / API keys need to be configured when using commercial API models
2. **模型路径 / Model Paths**: 使用本地模型时需要正确配置模型路径 / Model paths need to be correctly configured when using local models
3. **硬件要求 / Hardware Requirements**: 大模型推理需要足够的GPU内存 / Large model inference requires sufficient GPU memory
4. **数据路径 / Data Paths**: 确保图像文件路径正确配置 / Ensure image file paths are correctly configured

## 贡献指南 / Contributing

欢迎提交Issue和Pull Request来改进项目 / Welcome to submit Issues and Pull Requests to improve the project. 在提交代码前，请确保 / Before submitting code, please ensure:

1. 代码符合项目规范 / Code follows project standards
2. 添加必要的文档说明 / Add necessary documentation
3. 通过相关测试 / Pass relevant tests

## 许可证 / License

本项目采用MIT许可证，详见LICENSE文件 / This project is licensed under the MIT License, see LICENSE file for details.

## 联系方式 / Contact

如有问题或建议，请通过以下方式联系 / For questions or suggestions, please contact through:

- **GitHub**: [BiMeRe Repository](https://github.com/constant-lou/BiMeRe)
- **邮箱 / Email**: 2077658501@qq.com
- **提交GitHub Issue / Submit GitHub Issue**: [Issues](https://github.com/constant-lou/BiMeRe)

## 致谢 / Acknowledgments

感谢所有为这个项目做出贡献的研究者和开发者 / Thanks to all researchers and developers who contributed to this project. 