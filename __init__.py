"""
ARS (Adaptive Reflection Scheduling) - A framework for efficient LLM reasoning

This package implements adaptive scheduling strategies for language model inference,
dynamically adjusting reasoning depth based on question difficulty.
"""

__version__ = "1.0.0"
__author__ = "ARS Research Team"

from .config import MODELS, DATASETS, POLICIES
from .ars import ars_generate, schedule_mode_from_difficulty
from .policies import (
    VanillaPolicy,
    TALEPolicy,
    CGRSPolicy,
    CoDFastPolicy,
    ElasticModeratePolicy,
    DeepReflectPolicy
)
from .difficulty import heuristic_difficulty
from .evaluation import run_single_experiment, summarize
from .benchmark import run_benchmark

__all__ = [
    # Constants
    "MODELS",
    "DATASETS",
    "POLICIES",
    
    # Core ARS
    "ars_generate",
    "schedule_mode_from_difficulty",
    "heuristic_difficulty",
    
    # Policies
    "VanillaPolicy",
    "TALEPolicy",
    "CGRSPolicy",
    "CoDFastPolicy",
    "ElasticModeratePolicy",
    "DeepReflectPolicy",
    
    # Evaluation
    "run_single_experiment",
    "run_benchmark",
    "summarize",
]
