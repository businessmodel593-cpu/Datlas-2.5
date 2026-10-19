"""Datlas 2.5 Inference Generation

Supports greedy, sampling, and future KV-cache optimization.
"""

import torch
from datlas.models.transformer import DatlasModel
from typing import Optional, List

def generate(model: DatlasModel, prompt_ids: torch.Tensor, max_new_tokens: int = 100,
             temperature: float = 0.7, top_p: float = 0.9, top_k: int = 50) -> torch.Tensor:
    """Autoregressive generation."""
    model.eval()
    idx = prompt_ids.clone()
    
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits, _ = model(idx)
            logits = logits[:, -1, :] / temperature
            
            # Top-k / top-p filtering stub
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
            
            if next_token.item() == 0:  # EOS stub
                break
    
    return idx

def generate_text(model: DatlasModel, tokenizer, prompt: str, **kwargs) -> str:
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    out_ids = generate(model, ids, **kwargs)
    return tokenizer.decode(out_ids[0].tolist())
