"""
Answer extraction and normalization utilities
"""

import re
from typing import Optional
from .config import AMC_CHOICES, PLACEHOLDER_TOKENS
from .utils import log_debug


# Compiled regex patterns
BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
COORD_FRAC_PI_RE = re.compile(r"\(\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*\\frac\{\\pi\}\{([0-9]+)\}\s*\)")
COORD_SIMPLE_RE = re.compile(r"\(\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*\)")


def _strip_tex(s: str) -> str:
    """Remove LaTeX formatting from string"""
    s = s.replace("$", " ").replace("\\;", " ").replace("\\,", " ").replace("\\ ", " ")
    return s.strip()


def _clean_quotes(s: str) -> str:
    """Remove leading/trailing quotes"""
    return s.strip().strip("'\"")


def extract_final_answer(text: str, debug: bool = False) -> str:
    """
    Extract the final answer from generated text using multiple heuristics.
    
    Args:
        text: Generated text to extract answer from
        debug: Whether to log debug information
        
    Returns:
        Extracted answer string
    """
    if not isinstance(text, str):
        text = str(text)
    t = text.strip()

    # Try boxed notation first
    boxed_matches = BOXED_RE.findall(t)
    if boxed_matches:
        candidate = _clean_quotes(boxed_matches[-1])
        if candidate.lower() not in PLACEHOLDER_TOKENS:
            log_debug(f"extract: using last boxed -> '{candidate}'", debug)
            return candidate

    # Try GSM8K-style marker
    gsm_marker_idx = t.rfind("####")
    if gsm_marker_idx != -1:
        after_marker = t[gsm_marker_idx + 4:].strip()
        candidate = after_marker.splitlines()[0].split()[0] if after_marker else ""
        candidate = _clean_quotes(candidate)
        if candidate.lower() not in PLACEHOLDER_TOKENS:
            log_debug(f"extract: from '####' -> '{candidate}'", debug)
            return candidate

    # Try "Final Answer:" pattern
    fa_lines = re.findall(r"(?i)Final Answer:\s*(.*)", t)
    if fa_lines:
        candidate = _clean_quotes(fa_lines[-1].splitlines()[0])
        if candidate.lower() not in PLACEHOLDER_TOKENS:
            log_debug(f"extract: from 'Final Answer:' -> '{candidate}'", debug)
            return candidate

    # Try "answer is" pattern
    answer_phrases = re.findall(r"(?i)(?:The )?answer(?: is)?:?\s*(.*)", t)
    if answer_phrases:
        candidate = _clean_quotes(answer_phrases[-1].splitlines()[0])
        if candidate.lower() not in PLACEHOLDER_TOKENS:
            log_debug(f"extract: from 'answer is' -> '{candidate}'", debug)
            return candidate

    # Try single letter (for multiple choice)
    m = re.search(r"(^|\s)([A-Ea-e])(\s|[.\)]|$)", t)
    if m:
        letter = m.group(2).upper()
        log_debug(f"extract: AMC letter -> '{letter}'", debug)
        return letter

    # Try coordinate patterns
    m = COORD_FRAC_PI_RE.search(t)
    if m:
        r_val, denom = m.group(1), m.group(2)
        cand = f"({r_val}, \\frac{{\\pi}}{{{denom}}})"
        log_debug(f"extract: coord frac pi -> '{cand}'", debug)
        return cand
    
    m = COORD_SIMPLE_RE.search(t)
    if m:
        cand = f"({m.group(1)}, {m.group(2)})"
        log_debug(f"extract: coord simple -> '{cand}'", debug)
        return cand

    # Try last number
    numbers = re.findall(r"-?\d+(?:\.\d+)?", t)
    if numbers:
        cand = numbers[-1]
        if cand and cand not in ['-', '.', '+']:
            log_debug(f"extract: last number -> '{cand}'", debug)
            return cand

    # Fallback to last token
    toks = [w for w in re.split(r"\s+", t) if w and w.lower() not in PLACEHOLDER_TOKENS]
    if toks:
        cand = _clean_quotes(toks[-1])
        if cand and len(cand) > 1 and not cand.startswith('\\'):
            log_debug(f"extract: fallback token -> '{cand}'", debug)
            return cand

    log_debug(f"extract: no clear answer found, returning empty string", debug)
    return ""


def normalize_pred(pred: str, dataset: str, debug: bool = False) -> str:
    """
    Normalize predicted answer for comparison.
    
    Args:
        pred: Raw predicted answer
        dataset: Dataset name (affects normalization strategy)
        debug: Whether to log debug information
        
    Returns:
        Normalized answer string
    """
    if not pred:
        return ""
    
    s_raw = pred.strip()
    s = _strip_tex(s_raw)
    sl = s.lower()
    
    log_debug(f"normalize_pred (raw): '{s_raw}'", debug)
    log_debug(f"normalize_pred (stripped): '{s}'", debug)

    # Strip common prefixes
    prefixes_to_strip = ["final answer:", "answer:", "the answer is:"]
    for prefix in prefixes_to_strip:
        if sl.startswith(prefix):
            s = s[len(prefix):].strip()
            sl = s.lower()
            break

    s = _clean_quotes(s)
    if s.lower() in PLACEHOLDER_TOKENS:
        log_debug(f"normalize_pred: placeholder token -> ''", debug)
        return ""

    # Dataset-specific normalization
    if dataset == "arc":
        # Try to extract letter answer
        patterns = [
            r"\b([abcde])\b",
            r"^\s*\(?([abcde])\)?\s*[.)]?",
            r"choice\s*([abcde])",
            r"option\s*([abcde])",
            r"answer\s*(?:is\s*)?([abcde])",
        ]
        for pattern in patterns:
            m = re.search(pattern, sl, re.IGNORECASE)
            if m:
                return AMC_CHOICES.get(m.group(1).lower(), m.group(1).upper())
        
        su = s.strip().upper()
        if su in AMC_CHOICES.values():
            return su

    elif dataset == "gsm8k":
        # Extract numeric answer
        numbers = re.findall(r"-?\d+", sl)
        if numbers:
            return numbers[-1].lstrip("+0") or "0"

    elif dataset == "math500":
        # Try coordinate patterns
        m = COORD_FRAC_PI_RE.search(s_raw)
        if m:
            return f"({m.group(1)}, \\frac{{\\pi}}{{{m.group(2)}}})"
        
        m = COORD_SIMPLE_RE.search(s_raw)
        if m:
            return f"({m.group(1)}, {m.group(2)})"
        
        # Extract numeric answer
        numbers = re.findall(r"-?\d+(?:\.\d+)?", s)
        if numbers:
            return numbers[-1].lstrip("+0") or "0"

    # Generic numeric extraction
    s2 = s.replace("=", " ").replace("≈", " ")
    numbers = re.findall(r"-?\d+(?:\.\d+)?", s2)
    if numbers:
        return numbers[-1].lstrip("+0") or "0"

    # Fallback to last token
    tokens = [t for t in re.split(r"\s+", s) if t and t.lower() not in PLACEHOLDER_TOKENS]
    if tokens:
        return tokens[-1]
    
    return s


def normalize_gold(gold: str, dataset: str, debug: bool = False) -> str:
    """
    Normalize gold answer for comparison.
    
    Args:
        gold: Raw gold answer
        dataset: Dataset name (affects normalization strategy)
        debug: Whether to log debug information
        
    Returns:
        Normalized answer string
    """
    if not gold:
        return ""
    
    g_raw = str(gold).strip()
    g = _strip_tex(g_raw)
    
    log_debug(f"normalize_gold (raw): '{g_raw}'", debug)
    log_debug(f"normalize_gold (stripped): '{g}'", debug)

    if dataset == "arc":
        gU = g.strip().upper()
        m = re.search(r"\(?([A-E])\)?", gU)
        if m:
            return m.group(1)

    elif dataset == "gsm8k":
        nums = re.findall(r"-?\d+", g)
        if nums:
            return nums[-1].lstrip("+0") or "0"

    elif dataset == "math500":
        # Try boxed notation
        m = BOXED_RE.search(g_raw)
        if m:
            inner = m.group(1)
            m2 = COORD_FRAC_PI_RE.search(inner)
            if m2:
                return f"({m2.group(1)}, \\frac{{\\pi}}{{{m2.group(2)}}})"
        
        # Try coordinate patterns
        m = COORD_FRAC_PI_RE.search(g_raw)
        if m:
            return f"({m.group(1)}, \\frac{{\\pi}}{{{m.group(2)}}})"
        
        m = COORD_SIMPLE_RE.search(g_raw)
        if m:
            return f"({m.group(1)}, {m.group(2)})"
        
        # Extract numeric answer
        numbers = re.findall(r"-?\d+(?:\.\d+)?", g)
        if numbers:
            return numbers[-1].lstrip("+0") or "0"

    # Generic numeric extraction
    numbers = re.findall(r"-?\d+(?:\.\d+)?", g)
    if numbers:
        return numbers[-1].lstrip("+0") or "0"
    
    return g.strip()
