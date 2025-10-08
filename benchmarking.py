"""
Benchmarking harness for running experiments
"""

import json
import os
from typing import List, Dict, Any
import traceback

from .config import MODELS
from .utils import log_info, ensure_dir, sanitize, write_csv
from .evaluation import run_single_experiment


def run_benchmark(args) -> str:
    """
    Run full benchmark across models, datasets, and policies.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Path to CSV results file
    """
    models = [MODELS[m] for m in args.models]
    datasets = args.datasets
    policies = args.policies
    
    ensure_dir(args.bench_dir)
    
    table_rows = []
    total_experiments = len(args.models) * len(datasets) * len(policies)
    experiment_count = 0
    
    for model_key in args.models:
        model_id = MODELS[model_key]
        
        for dataset in datasets:
            for policy in policies:
                experiment_count += 1
                
                run_name = f"{sanitize(model_key)}__{dataset}__{policy}"
                run_dir = os.path.join(args.bench_dir, run_name)
                
                log_info(f"\n=== Experiment {experiment_count}/{total_experiments}: {run_name} ===")
                
                try:
                    summary = run_single_experiment(args, model_id, dataset, policy, run_dir)
                    
                    row = {
                        "model": model_key,
                        "dataset": dataset,
                        "policy": policy
                    }
                    row.update(summary)
                    table_rows.append(row)
                    
                except Exception as e:
                    log_info(f"Experiment failed: {e}")
                    traceback.print_exc()
                    
                    row = {
                        "model": model_key,
                        "dataset": dataset,
                        "policy": policy,
                        "accuracy": 0.0,
                        "TPC": 999999.0,
                        "lat_ms": 0.0,
                        "lat_p95": 0.0,
                        "J_per_correct": 999999.0,
                        "total_samples": 0,
                        "correct_samples": 0
                    }
                    table_rows.append(row)
    
    # Save results
    csv_path = os.path.join(args.bench_dir, "benchmark_summary.csv")
    write_csv(csv_path, table_rows)
    
    json_path = os.path.join(args.bench_dir, "benchmark_summary.json")
    with open(json_path, "w") as f:
        json.dump(table_rows, f, indent=2)
    
    log_info(f"\nBenchmark complete! Results saved to:")
    log_info(f"  CSV: {csv_path}")
    log_info(f"  JSON: {json_path}")
    
    return csv_path
