"""
Configuration settings for ARS (Adaptive Reflection Scheduling)
"""

# Model configurations
MODELS = {
    "qwen-1_5b": "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "qwen-7b": "Qwen/Qwen2.5-Math-7B-Instruct",
    "deepseek-distill-qwen-7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
}

# Dataset names
DATASETS = ["math500", "gsm8k", "arc"]

# Policy names
POLICIES = ["vanilla", "tale", "cgrs", "ars"]

# Minimum samples per dataset
MIN_SAMPLES = 50

# Answer extraction patterns
AMC_CHOICES = {"a": "A", "b": "B", "c": "C", "d": "D", "e": "E"}
PLACEHOLDER_TOKENS = {"<value>", "<answer>", "<final>", "<ans>", "<result>", "placeholder"}

# Math keywords for difficulty heuristic
MATH_KEYWORDS = [
    "triangle", "circle", "prime", "integer", "probability", "sequence", "sum",
    "geometry", "algebra", "factor", "remainder", "mod", "number", "ratio",
    "combination", "permutation", "derivative", "integral", "matrix", "vector", "function"
]

# Dataset loading configurations
DATASET_CONFIGS = {
    "gsm8k": [
        ("openai/gsm8k", "main", "train"),
        ("openai/gsm8k", "main", "test"),
        ("gsm8k", "main", "train"),
        ("gsm8k", "main", "test"),
    ],
    "arc": [
        ("allenai/ai2_arc", "ARC-Challenge", "test"),
        ("ai2_arc", "ARC-Challenge", "test"),
        ("allenai/ai2_arc", "ARC-Easy", "test"),
        ("ai2_arc", "ARC-Easy", "test"),
    ],
    "math500": [
        ("HuggingFaceH4/MATH-500", "test"),
        ("lightmanben/MATH-500", "test"),
        ("HuggingFaceH4/MATH-500", "main"),
        ("lightmanben/MATH-500", "main"),
    ],
}

# Default hyperparameters
DEFAULT_HYPERPARAMS = {
    "max_new_tokens": 1024,
    "drafts": 2,
    "think_budget": 64,
    "sc_k": 3,
    "d1": 0.4,
    "d2": 0.7,
    "c1": 0.7,
    "c2": 0.7,
    "c3": 0.8,
    "budget_tokens": 128,
    "cgrs_delta": 0.9,
    "temperature": 0.6,
    "top_p": 0.95,
    "avg_power_w": 6.0,
}
