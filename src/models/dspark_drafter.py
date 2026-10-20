"""DeepSpec block drafter (DSpark, and DeepSpec-trained DFlash) for Qwen3 targets.

Inference-only port of `Qwen3DSparkModel` from DeepSpec
(https://github.com/deepseek-ai/DeepSpec, commit 005e03b, MIT License,
Copyright (c) DeepSeek-AI): deepspec/modeling/dspark/qwen3/modeling.py,
markov_head.py, and common.py. Training-only code (anchor sampling, flex-attention
masks, losses) is omitted. DeepSpec itself requires transformers 5.

The drafter reads features of the target's hidden states and fills a block that
starts with the last verified token, followed by mask tokens, in one forward
pass; position i predicts the token after position i. A Markov head (DSpark)
adds a bias from the previously drafted token when sampling left to right, and
a confidence head predicts whether each draft token survives verification.
DeepSpec's DFlash drafters have neither head.
"""

import json
import os

import torch
from torch import nn
from transformers import Qwen3Config
from transformers.cache_utils import Cache
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3MLP,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    rotate_half,
)

ARCHITECTURE = "Qwen3DSparkModel"


def is_deepspec_drafter(path: str) -> bool:
    """Whether the drafter checkpoint directory `path` holds this architecture."""
    with open(os.path.join(path, "config.json")) as f:
        return json.load(f).get("architectures") == [ARCHITECTURE]


def load_drafter_config(path: str) -> Qwen3Config:
    """The drafter config, with transformers 5's `rope_parameters` mapped back to
    `rope_theta`/`rope_scaling` (otherwise transformers 4 silently uses
    rope_theta=10000)."""
    config = Qwen3Config.from_pretrained(path)
    rope = getattr(config, "rope_parameters", None)
    if rope is not None:
        config.rope_theta = rope["rope_theta"]
        if rope.get("rope_type", "default") == "default":
            config.rope_scaling = None
        else:
            config.rope_scaling = {k: v for k, v in rope.items() if k != "rope_theta"}
    return config


def _apply_rotary_pos_emb(q, k, cos, sin):
    """Queries cover only the block, the last positions of the keys."""
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    q_len = q.shape[-2]
    q = q * cos[..., -q_len:, :] + rotate_half(q) * sin[..., -q_len:, :]
    k = k * cos + rotate_half(k) * sin
    return q, k


class DSparkAttention(nn.Module):
    """Block queries attend, without a causal mask, to the cached and new target
    features and to the whole block."""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        num_heads, num_kv_heads = config.num_attention_heads, config.num_key_value_heads
        self.num_key_value_groups = num_heads // num_kv_heads  # read by SDPA
        self.scaling = self.head_dim**-0.5
        self.is_causal = False
        bias = config.attention_bias
        self.q_proj = nn.Linear(config.hidden_size, num_heads * self.head_dim, bias)
        self.k_proj = nn.Linear(config.hidden_size, num_kv_heads * self.head_dim, bias)
        self.v_proj = nn.Linear(config.hidden_size, num_kv_heads * self.head_dim, bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, config.hidden_size, bias)
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        context: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_values: Cache,
    ) -> torch.Tensor:
        bsz, q_len = hidden_states.shape[:-1]
        kv_len = context.shape[1] + q_len
        q = self.q_proj(hidden_states).view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k = torch.cat([self.k_proj(context), self.k_proj(hidden_states)], dim=1)
        v = torch.cat([self.v_proj(context), self.v_proj(hidden_states)], dim=1)
        k = self.k_norm(k.view(bsz, kv_len, -1, self.head_dim)).transpose(1, 2)
        v = v.view(bsz, kv_len, -1, self.head_dim).transpose(1, 2)
        q, k = _apply_rotary_pos_emb(q, k, *position_embeddings)
        k, v = past_key_values.update(k, v, self.layer_idx)
        output, _ = sdpa_attention_forward(
            self, q, k, v, None, scaling=self.scaling, is_causal=False
        )
        return self.o_proj(output.reshape(bsz, q_len, -1))


class DSparkDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.self_attn = DSparkAttention(config, layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(self, hidden_states, context, position_embeddings, past_key_values):
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states),
            context,
            position_embeddings,
            past_key_values,
        )
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class MarkovHead(nn.Module):
    """DeepSpec's "vanilla" Markov head: a low-rank bias from the previous token."""

    def __init__(self, vocab_size: int, rank: int):
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab_size, rank)
        self.markov_w2 = nn.Linear(rank, vocab_size, bias=False)

    def forward(self, prev_token_ids: torch.LongTensor) -> torch.Tensor:
        return self.markov_w2(self.markov_w1(prev_token_ids))


class ConfidenceHead(nn.Module):
    """Predicts the logit that a draft token is accepted."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.proj(features).squeeze(-1)


class DSparkDrafter(Qwen3PreTrainedModel):
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.target_layer_ids = [int(i) for i in config.target_layer_ids]
        self.block_size = int(config.block_size)
        self.mask_token_id = int(config.mask_token_id)
        # Frozen copies of the target's input embeddings and LM head.
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=getattr(config, "pad_token_id", None),
        )
        self.layers = nn.ModuleList(
            DSparkDecoderLayer(config, i) for i in range(config.num_hidden_layers)
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.fc = nn.Linear(
            len(self.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        rank = int(config.markov_rank)
        self.markov_head = None
        if rank > 0:
            head_type = getattr(config, "markov_head_type", "vanilla")
            if head_type != "vanilla":
                raise NotImplementedError(f"Markov head type {head_type!r}.")
            self.markov_head = MarkovHead(config.vocab_size, rank)
        self.confidence_head = None
        self.confidence_head_with_markov = False
        if config.enable_confidence_head:
            self.confidence_head_with_markov = bool(config.confidence_head_with_markov)
            self.confidence_head = ConfidenceHead(
                config.hidden_size + rank * self.confidence_head_with_markov
            )
        self.post_init()

    def context_features(self, hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Target features the drafter conditions on (index 0 is the embeddings)."""
        return torch.cat([hidden_states[i + 1] for i in self.target_layer_ids], dim=-1)

    def block_hidden_states(
        self,
        block_ids: torch.LongTensor,
        context: torch.Tensor,
        position_ids: torch.LongTensor,
        past_key_values: Cache,
    ) -> torch.Tensor:
        """Backbone over a block; `context` holds the target features not yet in
        `past_key_values`, and `position_ids` covers them and the block."""
        hidden_states = self.embed_tokens(block_ids)
        context = self.hidden_norm(self.fc(context))
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, context, position_embeddings, past_key_values
            )
        return self.norm(hidden_states)

    def step_logits(
        self, base_logits: torch.Tensor, prev_token_ids: torch.LongTensor
    ) -> torch.Tensor:
        """Draft logits at one position, given the previous (drafted) token."""
        if self.markov_head is None:
            return base_logits
        return base_logits + self.markov_head(prev_token_ids)

    def confidence_logits(
        self, hidden_states: torch.Tensor, prev_token_ids: torch.LongTensor
    ) -> torch.Tensor:
        features = hidden_states
        if self.confidence_head_with_markov:
            prev_embeddings = self.markov_head.markov_w1(prev_token_ids)
            features = torch.cat([hidden_states, prev_embeddings], dim=-1)
        return self.confidence_head(features).float()
