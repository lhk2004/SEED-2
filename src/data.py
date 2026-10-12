"""Training data: tokenized datasets saved with `datasets.save_to_disk`, and padding.

Rows hold `input_ids`, `attention_mask`, and `context_mask` (1 for prompt
tokens); see scripts/data/preprocess_distilled_jsonl.py.
"""

import os
from typing import Any

import torch
from datasets import Dataset, load_from_disk
from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase


def load_preprocessed_dataset(dataset_path: str) -> Dataset:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}.")
    return load_from_disk(dataset_path).with_format("torch")


class PaddingCollator:
    """Pad every example (including its `context_mask`) to `max_length`."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int):
        self.pad = DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.padding_side = tokenizer.padding_side
        self.max_length = max_length

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        context_mask = [f.pop("context_mask", None) for f in features]
        batch = self.pad(features)
        if all(c is not None for c in context_mask):
            context_mask = torch.nn.utils.rnn.pad_sequence(
                context_mask, batch_first=True
            )[..., : self.max_length]
            padding = self.max_length - context_mask.shape[-1]
            batch["context_mask"] = torch.nn.functional.pad(
                context_mask,
                (0, padding) if self.padding_side == "right" else (padding, 0),
            )
        return batch
