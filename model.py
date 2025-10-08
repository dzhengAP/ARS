"""
Model loading and text generation utilities
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Tuple
from .utils import log_info


def load_model(model_id: str, device: str, dtype: torch.dtype, verbose: bool = True) -> Tuple:
    """
    Load model and tokenizer.
    
    Args:
        model_id: HuggingFace model identifier
        device: Device to load model on ('cuda', 'mps', or 'cpu')
        dtype: Data type for model weights
        verbose: Whether to log loading progress
        
    Returns:
        Tuple of (tokenizer, model)
        
    Raises:
        Exception: If model loading fails
    """
    try:
        log_info(f"Loading tokenizer: {model_id}", verbose)
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=False,
            padding_side="left",
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        log_info(f"Loading model: {model_id}", verbose)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        ).eval()

        if device != "cuda":
            model.to(device)

        log_info(f"Model loaded successfully on {device}", verbose)
        return tokenizer, model
        
    except Exception as e:
        log_info(f"Error loading model {model_id}: {e}", verbose)
        raise


def decode_once(model, tokenizer, prompt: str, max_new_tokens: int,
                temp: float = 0.6, top_p: float = 0.95) -> str:
    """
    Generate text from prompt using the model.
    
    Args:
        model: Language model
        tokenizer: Tokenizer for the model
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        temp: Sampling temperature
        top_p: Top-p sampling parameter
        
    Returns:
        Complete text (prompt + generated)
    """
    device = next(model.parameters()).device
    
    try:
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=32768)
        input_ids = enc.input_ids.to(device)
        attention_mask = enc.attention_mask.to(device)

        # Use mixed precision on CUDA
        if hasattr(device, "type") and device.type == "cuda":
            with torch.no_grad(), torch.cuda.amp.autocast(
                dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            ):
                out = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temp,
                    top_p=top_p,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
        else:
            with torch.no_grad():
                out = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temp,
                    top_p=top_p,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )

        generated_tokens = out[0][input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return prompt + generated_text
        
    except Exception as e:
        log_info(f"Error during generation: {e}")
        return prompt + " [GENERATION_ERROR]"
