"""
Adaptive Reflection Scheduling (ARS) core implementation
"""

from typing import Tuple
from .difficulty import heuristic_difficulty
from .policies import CoDFastPolicy, ElasticModeratePolicy, DeepReflectPolicy
from .model import decode_once
from .answer_processing import extract_final_answer


def schedule_mode_from_difficulty(difficulty: float, d1: float = 0.4, d2: float = 0.7) -> str:
    """
    Determine reasoning mode based on difficulty thresholds.
    
    Args:
        difficulty: Difficulty score (0.0 to 1.0)
        d1: Lower threshold (below = FAST mode)
        d2: Upper threshold (above = DEEP mode)
        
    Returns:
        Mode string: "FAST", "MOD", or "DEEP"
    """
    if difficulty < d1:
        return "FAST"
    elif difficulty < d2:
        return "MOD"
    else:
        return "DEEP"


def ars_generate(model, tokenizer, question: str, *,
                 dataset: str,
                 drafts: int = 2,
                 think_budget: int = 64,
                 sc_k: int = 3,
                 d1: float = 0.4,
                 d2: float = 0.7,
                 c1: float = 0.7,
                 c2: float = 0.7,
                 c3: float = 0.8,
                 max_new_tokens: int = 256,
                 temperature: float = 0.6,
                 top_p: float = 0.95,
                 debug: bool = False) -> Tuple[str, str, float]:
    """
    Generate answer using Adaptive Reflection Scheduling.
    
    The ARS policy dynamically selects a reasoning strategy based on
    estimated question difficulty:
    - FAST mode: Chain-of-Draft for simple questions
    - MOD mode: Elastic reasoning with token budget
    - DEEP mode: Deep reflection with verification
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        question: Input question
        dataset: Dataset name (for formatting hints)
        drafts: Number of drafts for FAST mode
        think_budget: Token budget for MOD mode
        sc_k: Self-consistency samples for DEEP mode
        d1: Lower difficulty threshold
        d2: Upper difficulty threshold
        c1: Confidence threshold for FAST mode
        c2: Confidence threshold for MOD mode
        c3: Confidence threshold for DEEP mode
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        debug: Enable debug output
        
    Returns:
        Tuple of (generated_text, final_answer, difficulty_score)
    """
    # Estimate difficulty
    difficulty = heuristic_difficulty(question)
    
    # Select mode based on difficulty
    mode = schedule_mode_from_difficulty(difficulty, d1, d2)
    
    # Choose policy based on mode
    if mode == "FAST":
        policy = CoDFastPolicy(drafts=drafts, per_draft=10)
    elif mode == "MOD":
        policy = ElasticModeratePolicy(budget_tokens=think_budget)
    else:  # DEEP
        policy = DeepReflectPolicy(sc_k=sc_k)
    
    # Build prompt and generate
    prompt = policy.build_prompt(question, {"dataset": dataset})
    generated_text = decode_once(
        model, tokenizer, prompt,
        max_new_tokens=max_new_tokens,
        temp=temperature,
        top_p=top_p
    )
    
    # Extract answer
    final_answer = extract_final_answer(generated_text, debug=debug)
    
    return generated_text, final_answer, difficulty
