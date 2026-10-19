"""Datlas 2.5 Data Pipeline Stubs

Dataset loading, tokenization, and packing left for CEO implementation.
This module provides interfaces only.
"""

from typing import Protocol, Iterator
import torch

class DatasetLoader(Protocol):
    def get_dataloader(self, batch_size: int, seq_len: int) -> Iterator[dict]:
        """Return iterator of batches with 'input_ids' and optional 'labels'."""
        ...

def load_dataset(path: str, **kwargs):
    raise NotImplementedError("Dataset loading left for CEO. Implement in this module.")

# Placeholder tokenizer stub
class DatlasTokenizer:
    def __init__(self, vocab_size: int = 50257):
        self.vocab_size = vocab_size
    
    def encode(self, text: str) -> list:
        # Stub — replace with real tokenizer (BPE/SentencePiece)
        return [ord(c) % self.vocab_size for c in text[:1024]]
    
    def decode(self, ids: list) -> str:
        return ''.join(chr(i % 128) for i in ids)
