"""Datlas 2.5 Training Pipeline

This provides a practical local training path for larger corpora while keeping the
custom transformer architecture in place.
"""

import math
import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from datlas.models.transformer import DatlasModel


class TextDataset(Dataset):
    def __init__(self, texts: Sequence[str], vocab_size: int = 50257, max_seq_len: int = 512):
        self.texts = list(texts)
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int):
        text = self.texts[index]
        token_ids = [ord(ch) % self.vocab_size for ch in text[: self.max_seq_len]]
        if len(token_ids) < self.max_seq_len:
            token_ids.extend([0] * (self.max_seq_len - len(token_ids)))
        return {"input_ids": torch.tensor(token_ids, dtype=torch.long)}


class Trainer:
    def __init__(self, model: DatlasModel, lr: float = 3e-4, weight_decay: float = 0.1,
                 warmup_steps: int = 2000, max_steps: int = 100000):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.step = 0

    def get_lr(self) -> float:
        if self.step < self.warmup_steps:
            if self.warmup_steps == 0:
                return self.optimizer.defaults['lr']
            return self.optimizer.defaults['lr'] * (self.step / self.warmup_steps)
        if self.max_steps <= self.warmup_steps:
            return self.optimizer.defaults['lr']
        progress = (self.step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        progress = max(0.0, min(1.0, progress))
        return self.optimizer.defaults['lr'] * 0.5 * (1 + math.cos(math.pi * progress))

    def train_step(self, batch: dict) -> float:
        self.model.train()
        idx = batch['input_ids']
        targets = batch.get('labels', idx)
        logits, loss = self.model(idx, targets)
        if loss is None:
            raise ValueError("Model returned no loss. Check target shape.")
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.step += 1
        return float(loss.item())

    def train(self, dataloader: Optional[DataLoader] = None, steps: Optional[int] = None):
        if dataloader is None:
            print("WARNING: No dataloader provided. Dataset loading is not yet wired to a corpus.")
            return
        max_steps = steps or self.max_steps
        for batch in dataloader:
            loss = self.train_step(batch)
            if self.step % 100 == 0:
                print(f"Step {self.step}: loss={loss:.4f}")
            if self.step >= max_steps:
                break


def load_corpus_from_text(path: str) -> List[str]:
    corpus_path = Path(path)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {path}")
    text = corpus_path.read_text(encoding="utf-8", errors="ignore")
    chunks = [segment.strip() for segment in text.split("\n\n") if segment.strip()]
    return chunks[:10000]


def main():
    model = DatlasModel(vocab_size=50257, d_model=768, n_layers=12, n_heads=12, max_seq_len=512)
    trainer = Trainer(model)
    corpus_path = os.getenv("DATLAS_CORPUS", "")
    if corpus_path:
        corpus = load_corpus_from_text(corpus_path)
        dataset = TextDataset(corpus, vocab_size=50257, max_seq_len=512)
        loader = DataLoader(dataset, batch_size=2, shuffle=True)
        trainer.train(loader, steps=100)
    else:
        print("Datlas 2.5 Trainer initialized. Set DATLAS_CORPUS to a text corpus to train on larger data.")


if __name__ == "__main__":
    main()
