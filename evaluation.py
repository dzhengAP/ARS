"""
Evaluation and metrics computation
"""

import json
import os
import time
import numpy as np
from typing import Dict, List, Any
from tqdm import tqdm

from .config import MIN_SAMPLES
from .utils import log_info, ensure_dir, count_tokens
from .model import load_model, decode_once
from .datasets import get_dataset_iter
from .policies import VanillaPolicy, TALEPolicy, CGRSPolicy
from .ars import ars_generate
from .answer_processing import extract_final_answer, normalize_pred, normalize_gold
from .difficulty import heuristic_difficulty
from .energy import EnergyMeter


def summarize(records: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute summary statistics from evaluation records.
    
    Args:
        records: List of evaluation results
        
    Returns:
        Dictionary with summary metrics
    """
    if not records:
        return {
            "accuracy": 0.0,
            "TPC": 0.0,
            "lat_ms": 0.0,
            "lat_p95": 0.0,
            "J_per_correct": 0.0,
            "total_samples": 0,
            "correct_samples": 0
        }
    
    correct_count = sum(r["correct"] for r in records)
    total_count = len(records)
    
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    
    total_tokens = sum(r["total_tokens"] for r in records)
    tpc = total_tokens / correct_count if correct_count > 0 else float('inf')
    
    latencies = [r["lat_ms"] for r in records]
    lat_mean = np.mean(latencies) if latencies else 0.0
    lat_p95 = np.percentile(latencies, 95) if latencies else 0.0
    
    total_joules = sum(r["joules"] for r in records)
    jpc = total_joules / correct_count if correct_count > 0 else float('inf')
    
    return {
        "accuracy": float(accuracy),
        "TPC": float(tpc) if tpc != float('inf') else 999999.0,
        "lat_ms": float(lat_mean),
        "lat_p95": float(lat_p95),
        "J_per_correct": float(jpc) if jpc != float('inf') else 999999.0,
        "total_samples": total_count,
        "correct_samples": correct_count
    }


def run_single_experiment(args, model_id: str, dataset: str, policy: str, run_dir: str) -> Dict[str, float]:
    """
    Run a single experiment configuration.
    
    Args:
        args: Command-line arguments
        model_id: Model identifier
        dataset: Dataset name
        policy: Policy name
        run_dir: Directory to save results
        
    Returns:
        Summary statistics dictionary
    """
    log_info(f"Running: {model_id} | {dataset} | {policy}")
    
    # Setup device and dtype
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    
    # Load model
    tokenizer, model = load_model(model_id, device, dtype, verbose=args.verbose)
    
    # Load dataset
    dataset_iter = get_dataset_iter(dataset, max_n=args.max_n, verbose=args.verbose)
    
    # Setup output directory
    ensure_dir(run_dir)
    
    # Initialize energy meter
    meter = EnergyMeter(args.avg_power_w)
    
    records = []
    out_path = os.path.join(run_dir, "samples.jsonl")
    
    with open(out_path, "w", encoding="utf-8") as outf:
        progress_desc = f"{dataset}-{policy}"
        
        for i, ex in enumerate(tqdm(dataset_iter, total=None, desc=progress_desc)):
            q, gold = ex["question"], ex["gold"]
            debug_enabled = args.debug and len(records) < 5
            
            # Generate answer
            t0 = time.time()
            with meter:
                try:
                    if policy == "ars":
                        text, ans, D = ars_generate(
                            model, tokenizer, q,
                            dataset=dataset,
                            drafts=args.drafts,
                            think_budget=args.think_budget,
                            sc_k=args.sc_k,
                            d1=args.d1, d2=args.d2,
                            c1=args.c1, c2=args.c2, c3=args.c3,
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            debug=debug_enabled
                        )
                    else:
                        # Select policy
                        if policy == "vanilla":
                            pol = VanillaPolicy()
                        elif policy == "tale":
                            pol = TALEPolicy(budget_tokens=args.budget_tokens)
                        elif policy == "cgrs":
                            pol = CGRSPolicy(confidence_threshold=args.cgrs_delta)
                        else:
                            pol = VanillaPolicy()
                        
                        # Generate
                        prompt = pol.build_prompt(q, {"dataset": dataset})
                        text = decode_once(
                            model, tokenizer, prompt, args.max_new_tokens,
                            temp=args.temperature, top_p=args.top_p
                        )
                        ans = extract_final_answer(text, debug=debug_enabled)
                        D = heuristic_difficulty(q)
                        
                except Exception as e:
                    log_info(f"Generation error for sample {i}: {e}")
                    text, ans, D = "[GENERATION_ERROR]", "", 0.0
            
            # Compute metrics
            lat_ms = (time.time() - t0) * 1000.0
            joules = meter.read_last()
            
            # Normalize and check correctness
            pred_norm = normalize_pred(ans, dataset, debug=debug_enabled)
            gold_norm = normalize_gold(gold, dataset, debug=debug_enabled)
            is_correct = int(pred_norm == gold_norm and pred_norm != "")
            
            # Debug output
            if debug_enabled and i < 5:
                log_info(f"\n=== DEBUG SAMPLE {i} ===")
                log_info(f"Question: {q[:100]}...")
                log_info(f"Generated: {text[-200:]}")
                log_info(f"Raw answer: '{ans}'")
                log_info(f"Gold: '{gold}'")
                log_info(f"Pred normalized: '{pred_norm}'")
                log_info(f"Gold normalized: '{gold_norm}'")
                log_info(f"Correct: {is_correct}")
                log_info("====================")
                
                with open(os.path.join(run_dir, f"debug_sample_{i}.txt"), "w") as debug_f:
                    debug_f.write(f"Question: {q}\n")
                    debug_f.write(f"Full generated text:\n{text}\n")
                    debug_f.write(f"Raw extracted answer: '{ans}'\n")
                    debug_f.write(f"Gold: '{gold}'\n")
                    debug_f.write(f"Pred normalized: '{pred_norm}'\n")
                    debug_f.write(f"Gold normalized: '{gold_norm}'\n")
                    debug_f.write(f"Correct: {is_correct}\n")
            
            # Record results
            rec = {
                "model": model_id,
                "dataset": dataset,
                "policy": policy,
                "question": q,
                "gold": gold,
                "prediction": ans,
                "pred_norm": pred_norm,
                "gold_norm": gold_norm,
                "correct": is_correct,
                "lat_ms": lat_ms,
                "joules": joules,
                "total_tokens": count_tokens(text),
                "difficulty": D
            }
            outf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            records.append(rec)
            
            # Check if we've reached the minimum
            if len(records) >= (args.max_n if args.max_n and args.max_n >= MIN_SAMPLES else MIN_SAMPLES):
                break
    
    # Compute summary
    summary = summarize(records)
    
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    log_info(f"Results: Acc={summary['accuracy']:.3f}, TPC={summary['TPC']:.1f}, Latency={summary['lat_ms']:.1f}ms")
    
    # Cleanup
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return summary


# Import torch here for the function above
import torch
