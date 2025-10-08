"""
Difficulty estimation utilities
"""

import re
from .config import MATH_KEYWORDS


def heuristic_difficulty(question: str) -> float:
    """
    Estimate question difficulty using heuristics.
    
    Combines:
    - Length score (longer questions tend to be harder)
    - Keyword density (presence of mathematical terms)
    - Symbol density (mathematical operators and notation)
    
    Args:
        question: Question text
        
    Returns:
        Difficulty score between 0.0 and 1.0
    """
    L = len(question.split())
    
    # Length score: normalized by 80 words
    length_score = min(1.0, L / 80.0)
    
    # Keyword density: how many math keywords appear
    keyword_count = sum(question.lower().count(k) for k in MATH_KEYWORDS)
    keyword_density = keyword_count / max(1, len(MATH_KEYWORDS))
    keyword_density = max(0.0, min(1.0, keyword_density / 3.0))
    
    # Symbol density: mathematical operators and parentheses
    symbol_count = len(re.findall(r"[-=+*/^()]", question))
    symbol_density = symbol_count / max(10, L)
    symbol_density = max(0.0, min(1.0, symbol_density))
    
    # Weighted combination
    difficulty = 0.4 * length_score + 0.4 * keyword_density + 0.2 * symbol_density
    
    return difficulty
