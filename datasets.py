"""
Dataset loading utilities
"""

from typing import Iterator, Dict, Optional
from datasets import load_dataset
from .config import MIN_SAMPLES, DATASET_CONFIGS
from .utils import log_info, log_debug, enforce_min_samples


def iter_gsm8k(max_n: Optional[int] = None, verbose: bool = True) -> Iterator[Dict[str, str]]:
    """
    Load GSM8K dataset.
    
    Args:
        max_n: Maximum number of samples to load
        verbose: Whether to log loading information
        
    Yields:
        Dict with 'question' and 'gold' keys
    """
    tried = DATASET_CONFIGS["gsm8k"]
    ds = None
    
    for name, config, split in tried:
        try:
            ds = load_dataset(name, config, split=split)
            if verbose:
                log_info(f"[GSM8K] Loaded {name} | config={config} | split={split} | rows={len(ds)}")
            break
        except Exception as e:
            if verbose:
                log_debug(f"[GSM8K] Failed {name}:{config}:{split} → {type(e).__name__}: {e}")
    
    if ds is None:
        raise ValueError("[GSM8K] Could not load any variant (ensure config='main').")

    total = len(ds)
    n = enforce_min_samples(total, max_n, MIN_SAMPLES)
    
    if n < MIN_SAMPLES and total >= MIN_SAMPLES:
        log_info(f"[GSM8K] Enforcing minimum {MIN_SAMPLES} samples (override).")
        n = MIN_SAMPLES

    for i in range(n):
        item = ds[i]
        q = (item.get("question") or "").strip()
        a_full = (item.get("answer") or "").strip()
        gold = a_full.split("####")[-1].strip() if "####" in a_full else ""
        yield {"question": q, "gold": gold}


def iter_arc(max_n: Optional[int] = None, verbose: bool = True) -> Iterator[Dict[str, str]]:
    """
    Load ARC dataset.
    
    Args:
        max_n: Maximum number of samples to load
        verbose: Whether to log loading information
        
    Yields:
        Dict with 'question' and 'gold' keys
    """
    tried = DATASET_CONFIGS["arc"]
    ds = None
    
    for name, config, split in tried:
        try:
            ds = load_dataset(name, config, split=split)
            if verbose:
                log_info(f"[ARC] Loaded {name} | config={config} | split={split} | rows={len(ds)}")
            break
        except Exception as e:
            if verbose:
                log_debug(f"[ARC] Failed {name}:{config}:{split} → {type(e).__name__}: {e}")
    
    if ds is None:
        raise ValueError("[ARC] Could not load any ARC variant.")

    total = len(ds)
    n = enforce_min_samples(total, max_n, MIN_SAMPLES)
    
    if n < MIN_SAMPLES and total >= MIN_SAMPLES:
        log_info(f"[ARC] Enforcing minimum {MIN_SAMPLES} samples (override).")
        n = MIN_SAMPLES

    for i in range(n):
        item = ds[i]
        q_base = (item.get("question") or "").strip()
        choices = item.get("choices", {})
        labels = choices.get("label", []) or []
        texts = choices.get("text", []) or []
        formatted = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts)) if labels and texts else ""
        q = f"{q_base}\n{formatted}" if formatted else q_base
        a = (item.get("answerKey") or "").strip()
        yield {"question": q, "gold": a}


def iter_math500(max_n: Optional[int] = None, verbose: bool = True) -> Iterator[Dict[str, str]]:
    """
    Load MATH-500 dataset.
    
    Args:
        max_n: Maximum number of samples to load
        verbose: Whether to log loading information
        
    Yields:
        Dict with 'question' and 'gold' keys
    """
    tried = DATASET_CONFIGS["math500"]
    ds = None
    
    for name, split in tried:
        try:
            ds = load_dataset(name, split=split)
            if verbose:
                log_info(f"[MATH500] Loaded {name}:{split} | rows={len(ds)}")
            break
        except Exception as e:
            if verbose:
                log_debug(f"[MATH500] Failed {name}:{split} → {type(e).__name__}: {e}")
    
    if ds is None:
        raise ValueError("[MATH500] Could not load any MATH-500 variant.")

    total = len(ds)
    n = enforce_min_samples(total, max_n, MIN_SAMPLES)
    
    if n < MIN_SAMPLES and total >= MIN_SAMPLES:
        log_info(f"[MATH500] Enforcing minimum {MIN_SAMPLES} samples (override).")
        n = MIN_SAMPLES

    for i in range(n):
        item = ds[i]
        q = (item.get("problem") or item.get("question") or item.get("input") or "").strip()
        a = (item.get("answer") or item.get("label") or "").strip()
        gold = a.split("####")[-1].strip() if "####" in a else a
        yield {"question": q, "gold": gold}


def get_dataset_iter(name: str, max_n: Optional[int] = None, verbose: bool = True) -> Iterator[Dict[str, str]]:
    """
    Get dataset iterator by name.
    
    Args:
        name: Dataset name (math500, gsm8k, or arc)
        max_n: Maximum number of samples to load
        verbose: Whether to log loading information
        
    Returns:
        Iterator yielding dataset samples
        
    Raises:
        ValueError: If dataset name is unknown
    """
    if name == "math500":
        return iter_math500(max_n, verbose)
    elif name == "gsm8k":
        return iter_gsm8k(max_n, verbose)
    elif name == "arc":
        return iter_arc(max_n, verbose)
    else:
        raise ValueError(f"Unknown dataset: {name}"
