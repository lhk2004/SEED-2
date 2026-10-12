"""Pieces shared by the AR and SEED models."""

from dataclasses import dataclass

import torch
from transformers.modeling_outputs import ModelOutput


@dataclass
class LMOutput(ModelOutput):
    """Forward output used by the training loss and metrics.

    ``nlls`` and ``loss_mask`` are aligned with the next-token targets, i.e. they
    have shape (batch, seq_len - 1); ``nlls`` is zero wherever ``loss_mask`` is.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    nlls: torch.FloatTensor | None = None
    loss_mask: torch.LongTensor | None = None


def loss_mask(
    attention_mask: torch.LongTensor,
    context_mask: torch.LongTensor,
    include_context: bool,
) -> torch.LongTensor:
    """Tokens that contribute to the loss: non-padding, and non-prompt unless
    ``include_context``."""
    return attention_mask if include_context else attention_mask * (1 - context_mask)
