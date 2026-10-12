"""Autoregressive baseline: a Hugging Face causal LM trained with next-token loss."""

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM

from src.models.common import LMOutput, loss_mask


class AR(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        num_layers: int = -1,
        keep_top_layers: bool = False,
        from_scratch: bool = False,
        attn_implementation: str = "sdpa",
        torch_dtype: str | None = None,
        train_on_context: bool = False,
    ):
        """
        Args:
            pretrained_model_name_or_path: Hugging Face model id or path.
            num_layers: Number of decoder layers to keep (-1 keeps all).
            keep_top_layers: Keep the last (instead of the first) `num_layers`.
            from_scratch: Randomly initialize a `num_layers`-layer model with the
                pretrained model's architecture, in float32.
            attn_implementation: Hugging Face attention implementation.
            torch_dtype: dtype to load the pretrained weights in.
            train_on_context: Also train on prompt tokens (`context_mask == 1`).
        """
        super().__init__()
        self.train_on_context = train_on_context
        kwargs = {"trust_remote_code": True, "attn_implementation": attn_implementation}
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        if from_scratch:
            config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path, num_hidden_layers=num_layers, **kwargs
            )
            self.lm = AutoModelForCausalLM.from_config(
                config, torch_dtype=torch.float32
            )
        else:
            self.lm = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path, **kwargs
            )
            if num_layers != -1:
                layers = self.lm.model.layers
                self.lm.model.layers = (
                    layers[-num_layers:] if keep_top_layers else layers[:num_layers]
                )

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor | None = None,
        context_mask: torch.LongTensor | None = None,
    ) -> LMOutput:
        labels = input_ids[..., 1:].clone()[..., None]
        input_ids = input_ids[..., :-1]
        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            attention_mask = attention_mask[..., :-1]
        if context_mask is None:
            context_mask = torch.zeros_like(input_ids)
        else:
            context_mask = context_mask[..., :-1]
            # No padding: let the model use its default causal attention.
            if (
                attention_mask is not None
                and (attention_mask == 1).all()
                and (context_mask == 0).all()
            ):
                attention_mask = None
        mask = loss_mask(
            attention_mask
            if attention_mask is not None
            else torch.ones_like(input_ids),
            context_mask,
            include_context=self.training and self.train_on_context,
        )

        logits = self.lm(input_ids, attention_mask=attention_mask).logits
        log_probs = torch.log_softmax(logits, dim=-1)
        nlls = -torch.gather(log_probs, -1, labels).squeeze(-1) * mask
        # Mean over each sequence's tokens, then over sequences.
        loss = (nlls.sum(dim=-1) / mask.sum(dim=-1)).mean()
        return LMOutput(loss=loss, logits=logits, nlls=nlls, loss_mask=mask)

    @torch.no_grad()
    def generate(
        self,
        inputs: torch.LongTensor,
        attention_mask: torch.LongTensor | None = None,
        **kwargs,
    ) -> torch.LongTensor:
        """Standard Hugging Face generation (`kwargs` go to `generate`)."""
        if attention_mask is None:
            attention_mask = torch.ones_like(inputs)
        return self.lm.generate(inputs=inputs, attention_mask=attention_mask, **kwargs)
