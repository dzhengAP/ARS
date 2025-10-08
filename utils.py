"""
Utility functions for ARS
"""

import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_tokens(text: str) -> int:
    """Simple token counting by splitting on whitespace"""
    return max(1, len(text.strip().split()))


def ensure_dir(path: str):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)


def log_info(msg: str, verbose: bool = True):
    """Log info message if verbose"""
    if verbose:
        print(f"[INFO] {msg}")


def log_debug(msg: str, debug: bool = False):
    """Log debug message if debug mode enabled"""
    if debug:
        print(f"[DEBUG] {msg}")


def sanitize(s: str):
    """Sanitize string for use in filenames"""
    import re
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def enforce_min_samples(total_samples: int, requested_max_n: int, min_samples: int) -> int:
    """
    Enforce minimum sample count if dataset has enough rows.
    
    Args:
        total_samples: Total available samples in dataset
        requested_max_n: User-requested maximum samples
        min_samples: Minimum samples to enforce
        
    Returns:
        Number of samples to use
    """
    if total_samples < min_samples:
        return total_samples
    
    if requested_max_n is None or requested_max_n < min_samples:
        return min(total_samples, min_samples)
    
    return min(total_samples, requested_max_n)


def write_csv(path: str, rows: list):
    """Write list of dicts to CSV file"""
    if not rows:
        return
    
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            vals = [str(r.get(k, "")) for k in keys]
            f.write(",".join(vals) + "\n")
