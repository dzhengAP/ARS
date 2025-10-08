"""
Reasoning policies for ARS
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from .answer_processing import extract_final_answer


def _format_hint(meta: Optional[Dict[str, Any]]) -> str:
    """Generate dataset-specific answer format hint"""
    ds = (meta or {}).get("dataset", "")
    
    if ds == "gsm8k":
        return ("Your final line MUST be EXACTLY: 'Final Answer: <number>'.\n"
                "Example: Final Answer: 42")
    elif ds == "arc":
        return ("Your final line MUST be EXACTLY: 'Final Answer: <letter>'.\n"
                "Example: Final Answer: C")
    else:
        return ("End your response with a line: 'Final Answer: <value>'.\n"
                "Example: Final Answer: (3, \\frac{\\pi}{2})")


class ReasoningPolicy(ABC):
    """Base class for reasoning policies"""
    
    @abstractmethod
    def build_prompt(self, question: str, meta: Optional[Dict[str, Any]] = None) -> str:
        """Build prompt for the question"""
        pass
    
    @abstractmethod
    def postprocess(self, raw_output: str) -> Dict[str, Any]:
        """Postprocess model output to extract answer and metrics"""
        pass


class VanillaPolicy(ReasoningPolicy):
    """Standard prompting without constraints"""
    
    def build_prompt(self, question: str, meta: Optional[Dict[str, Any]] = None) -> str:
        hint = _format_hint(meta)
        return (f"Solve this step by step. Be very clear about your final answer.\n"
                f"{hint}\n"
                f"Question: {question}\n"
                f"Solution:\n")
    
    def postprocess(self, raw_output: str) -> Dict[str, Any]:
        before = raw_output.split("Final Answer:")[0] if "Final Answer:" in raw_output else raw_output
        final = extract_final_answer(raw_output)
        return {
            "final_answer": final,
            "thinking_tokens": len(before.split()),
            "total_tokens": len(raw_output.split())
        }


class TALEPolicy(ReasoningPolicy):
    """Token-Aware Limited Explanation policy"""
    
    def __init__(self, budget_tokens: int = 128):
        self.budget = budget_tokens
    
    def build_prompt(self, question: str, meta: Optional[Dict[str, Any]] = None) -> str:
        hint = _format_hint(meta)
        return (f"Solve concisely within ~{self.budget} tokens. Provide minimal but clear steps.\n"
                f"{hint}\nQuestion: {question}\nSolution:\n")
    
    def postprocess(self, raw_output: str) -> Dict[str, Any]:
        before = raw_output.split("Final Answer:")[0] if "Final Answer:" in raw_output else raw_output
        final = extract_final_answer(raw_output)
        return {
            "final_answer": final,
            "thinking_tokens": len(before.split()),
            "total_tokens": len(raw_output.split())
        }


class CGRSPolicy(ReasoningPolicy):
    """Confidence-Guided Reflection Scheduling policy"""
    
    def __init__(self, confidence_threshold: float = 0.9):
        self.threshold = confidence_threshold
    
    def build_prompt(self, question: str, meta: Optional[Dict[str, Any]] = None) -> str:
        hint = _format_hint(meta)
        return (f"Answer directly and concisely. Only reflect if you are uncertain.\n"
                f"Avoid speculative fillers if confident.\n{hint}\n"
                f"Question: {question}\nSolution:\n")
    
    def postprocess(self, raw_output: str) -> Dict[str, Any]:
        ans = extract_final_answer(raw_output)
        return {
            "final_answer": ans,
            "thinking_tokens": 0,
            "total_tokens": len(raw_output.split())
        }


class CoDFastPolicy(ReasoningPolicy):
    """Chain-of-Draft Fast policy"""
    
    def __init__(self, drafts: int = 2, per_draft: int = 10):
        self.drafts = drafts
        self.per_draft = per_draft
    
    def build_prompt(self, question: str, meta: Optional[Dict[str, Any]] = None) -> str:
        hint = _format_hint(meta)
        return (f"Write {self.drafts} ultra-brief draft ideas (≤{self.per_draft} tokens each), then solve.\n"
                f"Format:\nDRAFTS:\n- <idea1>\n- <idea2>\nSOLUTION:\n<solution>\n{hint}\n"
                f"Question: {question}\n")
    
    def postprocess(self, raw_output: str) -> Dict[str, Any]:
        think_part = raw_output.split("SOLUTION:")[0] if "SOLUTION:" in raw_output else ""
        final = extract_final_answer(raw_output)
        return {
            "final_answer": final,
            "thinking_tokens": len(think_part.split()),
            "total_tokens": len(raw_output.split())
        }


class ElasticModeratePolicy(ReasoningPolicy):
    """Elastic moderate-depth reasoning policy"""
    
    def __init__(self, budget_tokens: int = 64):
        self.budget = budget_tokens
    
    def build_prompt(self, question: str, meta: Optional[Dict[str, Any]] = None) -> str:
        hint = _format_hint(meta)
        return (f"THINK (≤ {self.budget} tokens): write key steps briefly.\n"
                f"SOLUTION: provide final answer.\n{hint}\nQuestion: {question}\n")
    
    def postprocess(self, raw_output: str) -> Dict[str, Any]:
        parts = raw_output.split("SOLUTION:")
        think = parts[0] if len(parts) > 1 else ""
        final = extract_final_answer(raw_output)
        return {
            "final_answer": final,
            "thinking_tokens": len(think.split()),
            "total_tokens": len(raw_output.split())
        }


class DeepReflectPolicy(ReasoningPolicy):
    """Deep reflection policy with self-consistency"""
    
    def __init__(self, sc_k: int = 3):
        self.sc_k = sc_k
    
    def build_prompt(self, question: str, meta: Optional[Dict[str, Any]] = None) -> str:
        hint = _format_hint(meta)
        return (f"Solve carefully with detailed reasoning. Reflect and verify.\n"
                f"{hint}\nQuestion: {question}\nDetailed Solution:\n")
    
    def postprocess(self, raw_output: str) -> Dict[str, Any]:
        before = raw_output.split("Final Answer:")[0] if "Final Answer:" in raw_output else raw_output
        final = extract_final_answer(raw_output)
        return {
            "final_answer": final,
            "thinking_tokens": len(before.split()),
            "total_tokens": len(raw_output.split())
        }
