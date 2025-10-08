#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main entry point for ARS (Adaptive Reflection Scheduling)
"""

import argparse
import sys
import warnings
warnings.filterwarnings("ignore")

from .config import MODELS, DATASETS, POLICIES, DEFAULT_HYPERPARAMS
from .utils import set_seed, log_info
from .testing import run_test_mode
from .benchmark import run_benchmark
from .evaluation import run_single_experiment


def create_parser():
    """Create command-line argument parser"""
    parser = argparse.ArgumentParser(
        description="Adaptive Reflection Scheduling (ARS) Evaluation"
    )
    
    # Mode selection
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test mode to validate setup"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run full benchmark across models/datasets/policies"
    )
    
    # Single run mode
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        help="Model for single run mode"
    )
    parser.add_argument(
        "--dataset",
        choices=DATASETS,
        default="math500",
        help="Dataset for single run mode"
    )
    parser.add_argument(
        "--policy",
        choices=POLICIES,
        default="ars",
        help="Policy for single run mode"
    )
    
    # Benchmark defaults
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()),
        default=["qwen-1_5b"],
        help="Models for benchmark mode"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=["gsm8k", "arc"],
        help="Datasets for benchmark mode"
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=POLICIES,
        default=["vanilla", "tale", "cgrs", "ars"],
        help="Policies for benchmark mode"
    )
    
    # ARS parameters
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=DEFAULT_HYPERPARAMS["max_new_tokens"]
    )
    parser.add_argument(
        "--drafts",
        type=int,
        default=DEFAULT_HYPERPARAMS["drafts"]
    )
    parser.add_argument(
        "--think_budget",
        type=int,
        default=DEFAULT_HYPERPARAMS["think_budget"]
    )
    parser.add_argument(
        "--sc_k",
        type=int,
        default=DEFAULT_HYPERPARAMS["sc_k"]
    )
    parser.add_argument(
        "--d1",
        type=float,
        default=DEFAULT_HYPERPARAMS["d1"]
    )
    parser.add_argument(
        "--d2",
        type=float,
        default=DEFAULT_HYPERPARAMS["d2"]
    )
    parser.add_argument(
        "--c1",
        type=float,
        default=DEFAULT_HYPERPARAMS["c1"]
    )
    parser.add_argument(
        "--c2",
        type=float,
        default=DEFAULT_HYPERPARAMS["c2"]
    )
    parser.add_argument(
        "--c3",
        type=float,
        default=DEFAULT_HYPERPARAMS["c3"]
    )
    
    # Baseline parameters
    parser.add_argument(
        "--budget_tokens",
        type=int,
        default=DEFAULT_HYPERPARAMS["budget_tokens"]
    )
    parser.add_argument(
        "--cgrs_delta",
        type=float,
        default=DEFAULT_HYPERPARAMS["cgrs_delta"]
    )
    
    # Generation parameters
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_HYPERPARAMS["temperature"]
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=DEFAULT_HYPERPARAMS["top_p"]
    )
    
    # Infrastructure
    parser.add_argument(
        "--avg_power_w",
        type=float,
        default=DEFAULT_HYPERPARAMS["avg_power_w"]
    )
    parser.add_argument(
        "--max_n",
        type=int,
        default=200,
        help="Max samples per dataset. Minimum 50 enforced if dataset has enough rows."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )
    
    # Output
    parser.add_argument(
        "--result_dir",
        type=str,
        default="results/single"
    )
    parser.add_argument(
        "--bench_dir",
        type=str,
        default="results/benchmark"
    )
    
    # Debug
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output for first few samples"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Enable verbose logging"
    )
    
    return parser


def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Run appropriate mode
    if args.test:
        ok = run_test_mode(args)
        sys.exit(0 if ok else 1)
        
    elif args.benchmark:
        run_benchmark(args)
        
    else:
        # Single run mode
        if not args.model:
            log_info("Single run mode: please specify --model")
            log_info(f"Available models: {list(MODELS.keys())}")
            sys.exit(1)
        
        model_id = MODELS[args.model]
        run_dir = args.result_dir
        
        run_single_experiment(args, model_id, args.dataset, args.policy, run_dir)
        log_info(f"Results saved to {run_dir}")


if __name__ == "__main__":
    main()
