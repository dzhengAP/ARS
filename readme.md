# ARS: Adaptive Reflection Scheduling

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Adaptive Reflection Scheduling (ARS) is a framework for efficient language model reasoning that dynamically adjusts computational depth based on question difficulty.

## Features

-  **Adaptive Scheduling**: Automatically selects reasoning strategy based on difficulty
-  **Multiple Policies**: Vanilla, TALE, CGRS, and ARS scheduling
-  **Comprehensive Evaluation**: Support for MATH500, GSM8K, and ARC datasets
-  **Efficient**: Optimizes token usage while maintaining accuracy
-  **Benchmarking**: Built-in harness for systematic evaluation

## Installation

```bash
# Clone the repository
git clone https://github.com/dzhengAP/ars.git
cd ars

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Test Mode

Validate your setup before running experiments:

```bash
python -m ars.main --test
```

### Single Run

Run a single experiment:

```bash
python -m ars.main \
    --model qwen-1_5b \
    --dataset gsm8k \
    --policy ars \
    --max_n 100
```

### Benchmark Mode

Run comprehensive benchmarks:

```bash
python -m ars.main --benchmark \
    --models qwen-1_5b \
    --datasets gsm8k arc \
    --policies vanilla tale cgrs ars \
    --max_n 200
```

## Project Structure

```
ars/
├── __init__.py           # Package initialization
├── config.py             # Configuration and constants
├── main.py               # CLI entry point
├── utils.py              # Utility functions
├── answer_processing.py  # Answer extraction and normalization
├── policies.py           # Reasoning policies
├── datasets.py           # Dataset loaders
├── model.py              # Model loading and generation
├── difficulty.py         # Difficulty estimation
├── energy.py             # Energy measurement
├── ars.py                # ARS core implementation
├── evaluation.py         # Evaluation and metrics
├── benchmark.py          # Benchmarking harness
└── testing.py            # Test mode
```

## Policies

### Vanilla
Standard step-by-step reasoning without constraints.

### TALE (Token-Aware Limited Explanation)
Constrains reasoning to a fixed token budget.

### CGRS (Confidence-Guided Reflection Scheduling)
Reflects only when uncertain, minimizing unnecessary computation.

### ARS (Adaptive Reflection Scheduling)
Dynamically selects reasoning depth based on difficulty:
- **FAST mode**: Chain-of-Draft for simple questions
- **MOD mode**: Elastic reasoning with token budget
- **DEEP mode**: Deep reflection with verification

## Configuration

Key parameters can be configured via command-line arguments:

- `--max_new_tokens`: Maximum tokens to generate (default: 1024)
- `--d1`, `--d2`: Difficulty thresholds for mode selection
- `--temperature`: Sampling temperature (default: 0.6)
- `--top_p`: Top-p sampling parameter (default: 0.95)

See `python -m ars.main --help` for full options.

## Supported Models

- Qwen2.5-Math-1.5B-Instruct
- Qwen2.5-Math-7B-Instruct
- DeepSeek-R1-Distill-Qwen-7B

## Supported Datasets

- **MATH500**: Challenging math problems
- **GSM8K**: Grade school math word problems
- **ARC**: AI2 Reasoning Challenge

## Results

Results are saved in the specified output directory:

```
results/
├── benchmark/
│   ├── benchmark_summary.csv
│   ├── benchmark_summary.json
│   └── <model>__<dataset>__<policy>/
│       ├── samples.jsonl
│       ├── summary.json
│       └── debug_sample_*.txt
└── single/
    ├── samples.jsonl
    └── summary.json
```

## Metrics

- **Accuracy**: Percentage of correct answers
- **TPC (Tokens Per Correct)**: Average tokens used per correct answer
- **Latency**: Mean and P95 inference time
- **Energy**: Joules per correct answer

## Citation

If you use this code in your research, please cite:

```bibtex
@software{ars2025,
  title={ARS: Adaptive Reflection Scheduling for Efficient LLM Reasoning},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/ars}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Built with 🤗 Transformers
- Datasets from HuggingFace Hub
- Inspired by recent work in efficient LLM inference
