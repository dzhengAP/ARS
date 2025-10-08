"""
Test mode for validating setup
"""

import torch
from .config import MODELS, DATASETS
from .utils import log_info
from .model import load_model, decode_once
from .datasets import get_dataset_iter
from .policies import VanillaPolicy
from .answer_processing import extract_final_answer, normalize_pred, normalize_gold


def run_test_mode(args) -> bool:
    """
    Run tests to validate model loading, datasets, and answer extraction.
    
    Args:
        args: Command-line arguments
        
    Returns:
        True if all tests passed, False otherwise
    """
    log_info("=== RUNNING TEST MODE ===")
    
    # Test model loading
    test_model = list(MODELS.keys())[0]
    log_info(f"Testing model loading: {test_model}")
    
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    
    try:
        model_id = MODELS[test_model]
        tokenizer, model = load_model(model_id, device, dtype)
        log_info("✓ Model loading successful")
    except Exception as e:
        log_info(f"✗ Model loading failed: {e}")
        return False
    
    # Test each dataset
    all_tests_passed = True
    
    for dataset in DATASETS:
        log_info(f"\nTesting dataset: {dataset}")
        
        try:
            # Load a couple samples to validate
            dataset_iter = get_dataset_iter(dataset, max_n=2, verbose=False)
            samples = list(dataset_iter)
            
            if not samples:
                log_info(f"✗ No samples loaded for {dataset}")
                all_tests_passed = False
                continue
            
            log_info(f"✓ Loaded {len(samples)} samples")
            
            # Test generation on first sample
            q, gold = samples[0]["question"], samples[0]["gold"]
            policy = VanillaPolicy()
            prompt = policy.build_prompt(q, {"dataset": dataset})
            
            try:
                generated_text = decode_once(model, tokenizer, prompt, 300, temp=0.3)
                extracted_answer = extract_final_answer(generated_text, debug=True)
                pred_norm = normalize_pred(extracted_answer, dataset, debug=True)
                gold_norm = normalize_gold(gold, dataset, debug=True)
                is_correct = pred_norm == gold_norm and pred_norm != ""
                
                log_info(f"  Generated (last 200): ...{generated_text[-200:]}")
                log_info(f"  Extracted: '{extracted_answer}'")
                log_info(f"  Pred normalized: '{pred_norm}'")
                log_info(f"  Gold normalized: '{gold_norm}'")
                log_info(f"  Correct: {is_correct}")
                
                if "Final Answer:" in generated_text or pred_norm:
                    log_info(f"✓ {dataset} test passed - extracted answer successfully")
                else:
                    log_info(f"⚠  {dataset} test warning - no clear answer found")
                    
            except Exception as e:
                log_info(f"✗ {dataset} generation failed: {e}")
                all_tests_passed = False
                
        except Exception as e:
            log_info(f"✗ {dataset} dataset loading failed: {e}")
            all_tests_passed = False
    
    # Cleanup
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if all_tests_passed:
        log_info("\n🎉 ALL TESTS PASSED! Ready for full experiments.")
        return True
    else:
        log_info("\n❌ Some tests failed. Please check the issues above.")
        return False
