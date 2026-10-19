#!/usr/bin/env python3
"""Pretrain a randomly initialized Datlas Qwen2-shaped causal LM."""

import argparse
import math
import os
from pathlib import Path
from typing import Iterator

import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

DEFAULT_DATASET_DIR = Path("datasets/Edge-Agent-Reasoning-WebSearch-260K")
DEFAULT_TOKENIZER_DIR = Path("datlas-2.5-final")
DEFAULT_OUTPUT_DIR = Path("datlas-2.5-independent-model")
DATASET_CARD_TOKEN_ESTIMATE = 692_100_000
SYSTEM_PROMPT = (
    "You are Datlas 2.5, a careful assistant. Answer the user request using the "
    "available context, and state uncertainty rather than inventing facts."
)


def build_config(tokenizer, sequence_length: int):
    return AutoConfig.for_model(
        "qwen2",
        vocab_size=len(tokenizer),
        hidden_size=1024,
        num_hidden_layers=22,
        num_attention_heads=16,
        num_key_value_heads=16,
        intermediate_size=2816,
        max_position_embeddings=sequence_length,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )


class PackedReasoningStream(IterableDataset):
    """Stream Parquet rows and pack prompt/target tokens into fixed-size blocks."""

    def __init__(self, parquet_files: list[str], tokenizer, sequence_length: int, seed: int):
        super().__init__()
        self.parquet_files = parquet_files
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.seed = seed

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        token_buffer: list[int] = []
        label_buffer: list[int] = []
        eos_id = self.tokenizer.eos_token_id
        if eos_id is None:
            raise ValueError("The tokenizer must define an EOS token")

        while True:
            dataset = load_dataset(
                "parquet",
                data_files={"train": self.parquet_files},
                split="train",
                streaming=True,
            ).shuffle(seed=self.seed, buffer_size=10_000)
            blocks_yielded = 0

            for row in dataset:
                user_prompt = str(row.get("user_prompt") or "").strip()
                reasoning = str(row.get("agent_reasoning") or "").strip()
                if not user_prompt or not reasoning:
                    continue

                metadata = "\n".join(
                    f"{key}: {row[key]}"
                    for key in ("role", "industry", "os")
                    if row.get(key)
                )
                user_content = f"{metadata}\n\n{user_prompt}" if metadata else user_prompt
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ]
                if self.tokenizer.chat_template:
                    prompt_encoding = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                    )
                    prompt_ids = prompt_encoding["input_ids"]
                else:
                    prompt = f"System: {SYSTEM_PROMPT}\nUser: {user_content}\nAssistant:"
                    prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)

                target_ids = self.tokenizer.encode(reasoning, add_special_tokens=False)
                example_ids = list(prompt_ids) + target_ids + [eos_id]
                example_labels = [-100] * len(prompt_ids) + target_ids + [eos_id]
                token_buffer.extend(example_ids)
                label_buffer.extend(example_labels)

                while len(token_buffer) >= self.sequence_length:
                    yield {
                        "input_ids": torch.tensor(token_buffer[: self.sequence_length], dtype=torch.long),
                        "labels": torch.tensor(label_buffer[: self.sequence_length], dtype=torch.long),
                    }
                    del token_buffer[: self.sequence_length]
                    del label_buffer[: self.sequence_length]
                    blocks_yielded += 1

            if blocks_yielded == 0:
                raise RuntimeError("Dataset stream produced no full training blocks")


def stack_batch(examples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([example[key] for example in examples]) for key in examples[0]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("datlas_2.5_scratch_checkpoints"))
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=0, help="Override the step estimate; useful for short test runs")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Validate dataset, tokenizer and model size without training")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow impractically slow/large CPU training")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sequence_length < 128:
        raise SystemExit("--sequence-length must be at least 128")
    if not args.tokenizer_dir.is_dir():
        raise SystemExit(f"Tokenizer directory not found: {args.tokenizer_dir}")
    parquet_files = sorted(str(path) for path in args.dataset_dir.glob("*.parquet"))
    if not parquet_files:
        raise SystemExit(f"No Parquet shards found in {args.dataset_dir}")

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_dir), local_files_only=True)
    config = build_config(tokenizer, args.sequence_length)
    with torch.device("meta"):
        parameter_count = sum(
            parameter.numel()
            for parameter in AutoModelForCausalLM.from_config(config).parameters()
        )
    if parameter_count >= 1_000_000_000:
        raise SystemExit(f"Configured model is not under 1B parameters: {parameter_count:,}")

    stream = PackedReasoningStream(parquet_files, tokenizer, args.sequence_length, args.seed)
    first_block = next(iter(stream))
    if args.dry_run:
        print(f"Dataset shards: {len(parquet_files)}")
        print(f"Tokenizer vocabulary: {len(tokenizer):,}")
        print(f"Randomly initialized parameters: {parameter_count:,} ({parameter_count / 1e6:.1f}M)")
        print(f"First packed block shape: {tuple(first_block['input_ids'].shape)}")
        print("Dry run complete; no model weights were loaded or trained.")
        return

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit(
            "Full 438M-parameter pretraining requires a CUDA GPU with substantial VRAM. "
            "This host is CPU-only; use --dry-run here or pass --allow-cpu only if you accept extremely slow training."
        )

    tokens_per_update = args.sequence_length * args.batch_size * args.gradient_accumulation_steps
    estimated_steps_per_epoch = math.ceil(DATASET_CARD_TOKEN_ESTIMATE / tokens_per_update)
    max_steps = args.max_steps or math.ceil(estimated_steps_per_epoch * args.epochs)
    config.use_cache = False
    model = AutoModelForCausalLM.from_config(config)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(
        output_dir=str(args.checkpoint_dir),
        max_steps=max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.1,
        # warmup_ratio removed to avoid parameter issues
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=bf16,
        fp16=torch.cuda.is_available() and not bf16,
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=torch.cuda.is_available(),
        remove_unused_columns=False,
        report_to="none",
        optim="adamw_torch",
        seed=args.seed,
    )
    print(f"Training {parameter_count:,} randomly initialized parameters for up to {max_steps:,} optimizer steps")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=stream,
        data_collator=stack_batch,
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    model.config.use_cache = True
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"Saved independently trained Datlas 2.5 model to {args.output_dir}")


if __name__ == "__main__":
    main()
