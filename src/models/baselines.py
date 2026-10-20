"""EAGLE-3, DFlash, and DSpark speculative decoding baselines, evaluated like SEED.

Each wraps the authors' released drafters and decoding steps:

- EAGLE-3 (https://github.com/SafeAILab/EAGLE): the official package. Its
  `EaModel.eagenerate` loop is reproduced here to record per-round statistics,
  to size the KV cache for each request, and to stop exactly like Hugging Face
  `generate` (first EOS, or `max_new_tokens`).
- DFlash (https://github.com/z-lab/dflash): the drafter's Hub remote code, with
  the decoding loop of its `spec_generate` (which returns no statistics).
- DSpark (https://github.com/deepseek-ai/DeepSpec): a port of DeepSpec's drafter
  (`src.models.dspark_drafter`) and of its evaluation's decoding loop. It also
  runs DeepSpec's DFlash drafters, trained on the same data as its DSpark ones.

The verifier is the unmodified target model, so greedy decoding reproduces the
target's greedy output (up to floating-point differences in batched
verification). Per-round statistics follow `SEED.generate`: accepted lengths
exclude the target's bonus token; a draft length is the longest verified
candidate (the tree depth for EAGLE-3, `block_size - 1` for DFlash, the
verified block for DSpark).
"""

import os
import time
from contextlib import contextmanager

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from torch import nn
from transformers import AutoModel, AutoModelForCausalLM, GenerationConfig
from transformers.cache_utils import DynamicCache
from transformers.integrations.sdpa_attention import sdpa_attention_forward

from src.models.dspark_drafter import (
    DSparkDrafter,
    is_deepspec_drafter,
    load_drafter_config,
)
from src.models.seed import SpeculativeOutput

# Commits of the public drafters these baselines were checked with (DFlash runs
# its drafter's remote code); any other drafter loads its latest revision.
DRAFT_REVISIONS = {
    "AngelSlim/Qwen3-4B_eagle3": "fd331e59626c8e95c392381a16ee59d518727fbb",
    "AngelSlim/Qwen3-8B_eagle3": "9629dfce7a4a10564dd48d3e5485c3976095653c",
    "deepseek-ai/dflash_qwen3_4b_block7": "02d530b7962ea1412beaf41a05c0b8e36d5f9b1d",
    "deepseek-ai/dflash_qwen3_8b_block7": "9e44dbbb6cb68b0c943abf9c5fc3c17c00897cdf",
    "deepseek-ai/dspark_qwen3_4b_block7": "3457dff1417cb84927f6098a5fcb7cee85c934b7",
    "deepseek-ai/dspark_qwen3_8b_block7": "03326e5043815da1f81b109078b2889737c26017",
    "z-lab/Qwen3-4B-DFlash-b16": "b74e3a329c4d963783143b1e970d95b002be72bd",
    "z-lab/Qwen3-8B-DFlash-b16": "9b41424b7109f9c5413454f481b09a82b85333f4",
}


def _sample(logits: torch.Tensor, temperature: float) -> torch.LongTensor:
    """Greedy for temperature 0, otherwise a sample at that temperature."""
    if temperature == 0:
        return logits.argmax(dim=-1)
    probs = (logits.float() / temperature).softmax(dim=-1)
    return torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(probs.shape[:-1])


def _draft_snapshot(draft_model_path: str, filename: str | None = None) -> str:
    """Local directory of a drafter (with only `filename` if given)."""
    if os.path.isdir(draft_model_path):
        return draft_model_path
    revision = DRAFT_REVISIONS.get(draft_model_path)
    if filename is not None:
        return os.path.dirname(
            hf_hub_download(draft_model_path, filename, revision=revision)
        )
    return snapshot_download(draft_model_path, revision=revision)


def _stop_token_ids(
    generation_config: GenerationConfig, target: nn.Module, device: torch.device
) -> torch.LongTensor:
    eos = generation_config.eos_token_id
    if eos is None:
        eos = target.generation_config.eos_token_id
    return torch.tensor(
        [eos] if isinstance(eos, int) else list(eos), dtype=torch.long, device=device
    )


def _accepted_until_stop(tokens: torch.LongTensor, stop_token_ids) -> int:
    """Accepted draft tokens up to and including the first EOS (later ones are
    discarded, so they do not count)."""
    stops = torch.isin(tokens, stop_token_ids).nonzero()
    return int(stops[0, 0]) + 1 if len(stops) else len(tokens)


def _truncate(
    sequences: torch.LongTensor,
    prompt_len: int,
    max_new_tokens: int,
    stop_token_ids: torch.LongTensor,
) -> torch.LongTensor:
    """Keep at most `max_new_tokens` generated tokens, up to the first EOS."""
    generated = sequences[0, prompt_len : prompt_len + max_new_tokens]
    stops = torch.isin(generated, stop_token_ids).nonzero()
    if len(stops):
        generated = generated[: stops[0, 0] + 1]
    return torch.cat([sequences[0, :prompt_len], generated])[None]


class _Rounds:
    """Per-round draft statistics and timing of one `generate` call."""

    def __init__(self, device: torch.device):
        self.use_cuda_events = device.type == "cuda"
        self.draft_lengths: list[int] = []
        self.accepted_lengths: list[int] = []
        self.draft_spans = []
        self.start = self._mark()

    def _mark(self):
        if not self.use_cuda_events:
            return time.perf_counter()
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def _elapsed(self, start, end) -> float:
        if self.use_cuda_events:
            return start.elapsed_time(end) / 1000.0
        return end - start

    @contextmanager
    def drafting(self):
        start = self._mark()
        yield
        self.draft_spans.append((start, self._mark()))

    def add(self, draft_length: int, accepted: int) -> None:
        self.draft_lengths.append(draft_length)
        self.accepted_lengths.append(accepted)

    def output(self, sequences: torch.LongTensor) -> SpeculativeOutput:
        end = self._mark()
        if self.use_cuda_events:
            torch.cuda.synchronize()
        positions = range(max(self.draft_lengths, default=0))
        return SpeculativeOutput(
            sequences=sequences,
            num_drafted=sum(self.draft_lengths),
            num_accepted=sum(self.accepted_lengths),
            accepted_lengths=self.accepted_lengths,
            draft_lengths=self.draft_lengths,
            position_attempts=[
                sum(n > pos for n in self.draft_lengths) for pos in positions
            ],
            position_accepts=[
                sum(n > pos for n in self.accepted_lengths) for pos in positions
            ],
            drafting_time=sum(self._elapsed(s, e) for s, e in self.draft_spans),
            total_time=self._elapsed(self.start, end),
        )


def _eagle_sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs):
    """Hugging Face SDPA attention (as used by AR, SEED, and DFlash) in place of
    EAGLE's eager attention. EAGLE's masks are additive float32; 0 is visible."""
    if attention_mask is not None:
        attention_mask = attention_mask == 0
    return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)


def _patch_eagle_head_dim() -> None:
    """EAGLE's drafter attention assumes head_dim = hidden_size / num_heads; use
    the config's `head_dim` instead (Qwen3-4B: 128, not 80)."""
    from eagle.model.cnets import LlamaAttention

    if getattr(LlamaAttention, "_config_head_dim", False):
        return
    init = LlamaAttention.__init__

    def patched_init(self, config):
        init(self, config)
        head_dim = getattr(config, "head_dim", None) or self.head_dim
        if head_dim == self.head_dim:
            return
        self.head_dim = head_dim
        q_width = self.num_heads * head_dim
        kv_width = self.num_key_value_heads * head_dim
        self.q_proj = nn.Linear(config.hidden_size * 2, q_width, bias=False)
        self.k_proj = nn.Linear(config.hidden_size * 2, kv_width, bias=False)
        self.v_proj = nn.Linear(config.hidden_size * 2, kv_width, bias=False)
        self.o_proj = nn.Linear(q_width, config.hidden_size, bias=False)
        self._init_rope()
        # `forward` reshapes the attention output to `hidden_size` features.
        self.hidden_size = q_width

    LlamaAttention.__init__ = patched_init
    LlamaAttention._config_head_dim = True


class EAGLE3(nn.Module):
    def __init__(
        self,
        target_model_path: str,
        draft_model_path: str,
        device: torch.device | str,
        total_token: int = 32,
        depth: int = 8,
        top_k: int = 4,
        sdpa: bool = True,
        torch_dtype: str = "bfloat16",
    ):
        """
        Args:
            target_model_path: Target (verifier) model id or path.
            draft_model_path: EAGLE-3 drafter id or path.
            total_token, depth, top_k: Draft tree size, depth, and branching.
                Defaults are those of EAGLE's Qwen3 evaluation script.
            sdpa: Run the target's attention with SDPA instead of EAGLE's
                eager attention (Qwen3 targets only).
        """
        super().__init__()
        from eagle.model import modeling_qwen3_kv
        from eagle.model.ea_model import EaModel

        _patch_eagle_head_dim()
        if sdpa:
            # Looked up as a module global on every attention call.
            modeling_qwen3_kv.eager_attention_forward = _eagle_sdpa_attention_forward
        self.model = EaModel.from_pretrained(
            use_eagle3=True,
            base_model_path=target_model_path,
            ea_model_path=_draft_snapshot(draft_model_path),
            total_token=total_token,
            depth=depth,
            top_k=top_k,
            torch_dtype=torch_dtype,
            # EAGLE builds tree buffers on the weights' device at load time.
            device_map={"": str(device)},
        ).eval()
        self.requires_grad_(False)
        self._kv_cache = None
        self._kv_capacity = 0
        self._rounds = None

        top_k_generate = self.model.ea_layer.topK_genrate

        def timed_top_k_generate(*args, **kwargs):
            with self._rounds.drafting():
                return top_k_generate(*args, **kwargs)

        self.model.ea_layer.topK_genrate = timed_top_k_generate

    def _reset_kv_cache(self, length: int):
        """EAGLE's preallocated target KV cache, grown to hold `length` tokens."""
        from eagle.model.kv_cache import initialize_past_key_values

        if length > self._kv_capacity:
            self._kv_cache = None
            self._kv_capacity = -(-length // 1024) * 1024
            self._kv_cache = initialize_past_key_values(
                self.model.base_model, max_length=self._kv_capacity
            )
        self._kv_cache[2].zero_()  # current lengths
        return self._kv_cache

    @torch.inference_mode()
    def generate(
        self,
        inputs: torch.LongTensor,
        generation_config: GenerationConfig | None = None,
        max_new_tokens: int | None = None,
    ) -> SpeculativeOutput:
        """EAGLE-3 tree speculative decoding of one prompt of shape (1, length)."""
        from eagle.model.utils import (
            evaluate_posterior,
            initialize_tree,
            prepare_logits_processor,
            reset_tree_mode,
            tree_decoding,
            update_inference_inputs,
        )

        generation_config = generation_config or GenerationConfig()
        max_new_tokens = max_new_tokens or generation_config.max_new_tokens
        logits_processor = None
        if generation_config.do_sample:
            if not generation_config.temperature > 0:
                raise ValueError("Sampling requires a temperature > 0.")
            logits_processor = prepare_logits_processor(
                temperature=generation_config.temperature,
                top_p=generation_config.top_p or 0.0,
                top_k=generation_config.top_k or 0,
            )
        model, device = self.model, inputs.device
        prompt_len = inputs.shape[1]
        stop_token_ids = _stop_token_ids(generation_config, model.base_model, device)
        ea_layer = model.ea_layer
        past_key_values, past_key_values_data, current_length_data = (
            self._reset_kv_cache(
                prompt_len + max_new_tokens + ea_layer.total_tokens + ea_layer.depth + 2
            )
        )
        ea_layer.reset_kv()
        reset_tree_mode(model)

        rounds = self._rounds = _Rounds(device)
        input_ids = inputs.clone()
        padding = torch.full((1, 1), -1, dtype=torch.long, device=device)
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, _, _, token = (
            initialize_tree(input_ids, model, past_key_values, logits_processor)
        )
        new_token = 0
        while True:
            model.base_model.model.tree_mask = tree_mask
            logits, hidden_state_new, _ = tree_decoding(
                model,
                draft_tokens.to(device),
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
            )
            candidates = torch.cat((draft_tokens, padding), dim=1)[0, retrieve_indices]
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )
            accept_length = int(accept_length)
            rounds.add(
                retrieve_indices.shape[1] - 1,
                _accepted_until_stop(
                    candidates[best_candidate, 1 : accept_length + 1], stop_token_ids
                ),
            )
            # Appends the accepted path; `token` (the target's next token) is the
            # root of the next draft tree.
            (
                input_ids,
                draft_tokens,
                retrieve_indices,
                tree_mask,
                tree_position_ids,
                new_token,
                _,
                token,
            ) = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                logits_processor,
                new_token,
                past_key_values_data,
                current_length_data,
                model,
                hidden_state_new,
                sample_p,
            )
            sequences = torch.cat([input_ids, token.view(1, 1).to(device)], dim=1)
            generated = sequences[0, prompt_len:]
            if (
                len(generated) >= max_new_tokens
                or torch.isin(generated, stop_token_ids).any()
            ):
                break
        reset_tree_mode(model)
        return rounds.output(
            _truncate(sequences, prompt_len, max_new_tokens, stop_token_ids)
        )


class DFlash(nn.Module):
    def __init__(
        self,
        target_model_path: str,
        draft_model_path: str,
        device: torch.device | str,
        block_size: int | None = None,
        torch_dtype: str = "bfloat16",
    ):
        """
        Args:
            target_model_path: Target (verifier) model id or path.
            draft_model_path: DFlash drafter id or path (Hub remote code).
            block_size: Tokens per drafted block, including the last verified
                token (default: the drafter's training block size).
        """
        super().__init__()
        self.target = AutoModelForCausalLM.from_pretrained(
            target_model_path, attn_implementation="sdpa", torch_dtype=torch_dtype
        )
        self.drafter = AutoModel.from_pretrained(
            draft_model_path,
            revision=DRAFT_REVISIONS.get(draft_model_path),
            trust_remote_code=True,
            attn_implementation="sdpa",
            torch_dtype=torch_dtype,
        )
        self.to(device).eval()
        self.requires_grad_(False)
        self.block_size = block_size or self.drafter.block_size

    def _context(self, hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Target features the drafter conditions on (index 0 is the embeddings)."""
        return torch.cat(
            [hidden_states[i + 1] for i in self.drafter.target_layer_ids], dim=-1
        )

    @torch.inference_mode()
    def generate(
        self,
        inputs: torch.LongTensor,
        generation_config: GenerationConfig | None = None,
        max_new_tokens: int | None = None,
    ) -> SpeculativeOutput:
        """DFlash block speculative decoding of one prompt of shape (1, length).

        The drafter fills a block of mask tokens greedily in one forward pass.
        When sampling, the target samples each position and a draft token is
        accepted while it equals that sample, which is exact for greedy drafts.
        """
        generation_config = generation_config or GenerationConfig()
        max_new_tokens = max_new_tokens or generation_config.max_new_tokens
        temperature = 0.0
        if generation_config.do_sample:
            temperature = generation_config.temperature
            if not temperature > 0:
                raise ValueError("Sampling requires a temperature > 0.")
            if generation_config.top_k or (generation_config.top_p or 1.0) < 1.0:
                raise ValueError("DFlash sampling supports no top-k/top-p.")
        target, device = self.target, inputs.device
        block_size, prompt_len = self.block_size, inputs.shape[1]
        stop_token_ids = _stop_token_ids(generation_config, target, device)
        max_length = prompt_len + max_new_tokens
        output_ids = torch.full(
            (1, max_length + block_size),
            self.drafter.mask_token_id,
            dtype=torch.long,
            device=device,
        )
        position_ids = torch.arange(output_ids.shape[1], device=device)[None]
        target_cache, draft_cache = DynamicCache(), DynamicCache()

        rounds = _Rounds(device)
        output = target(
            inputs,
            position_ids=position_ids[:, :prompt_len],
            past_key_values=target_cache,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=True,
        )
        output_ids[:, :prompt_len] = inputs
        output_ids[:, prompt_len] = _sample(output.logits[:, -1], temperature)
        context = self._context(output.hidden_states)
        # Position `start` holds the last verified token, which heads each block.
        start = prompt_len
        while start < max_length:
            block = output_ids[:, start : start + block_size].clone()
            with rounds.drafting():
                hidden = self.drafter(
                    target_hidden=context,
                    noise_embedding=target.model.embed_tokens(block),
                    position_ids=position_ids[
                        :, draft_cache.get_seq_length() : start + block_size
                    ],
                    past_key_values=draft_cache,
                    use_cache=True,
                    is_causal=False,
                )
                draft_cache.crop(start)
                block[:, 1:] = target.lm_head(hidden[:, 1 - block_size :]).argmax(-1)

            output = target(
                block,
                position_ids=position_ids[:, start : start + block_size],
                past_key_values=target_cache,
                use_cache=True,
                output_hidden_states=True,
            )
            posterior = _sample(output.logits, temperature)
            accepted = int((block[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum())
            output_ids[:, start + 1 : start + accepted + 1] = block[:, 1 : accepted + 1]
            output_ids[:, start + accepted + 1] = posterior[:, accepted]
            rounds.add(
                block_size - 1,
                _accepted_until_stop(block[0, 1 : accepted + 1], stop_token_ids),
            )
            start += accepted + 1
            target_cache.crop(start)
            context = self._context(output.hidden_states)[:, : accepted + 1]
            if torch.isin(output_ids[0, prompt_len : start + 1], stop_token_ids).any():
                break
        return rounds.output(
            _truncate(
                output_ids[:, : start + 1], prompt_len, max_new_tokens, stop_token_ids
            )
        )


def dflash(
    target_model_path: str,
    draft_model_path: str,
    device: torch.device | str,
    block_size: int | None = None,
    torch_dtype: str = "bfloat16",
) -> "DFlash | DSpark":
    """The DFlash baseline: `DFlash` for z-lab's drafters, `DSpark` for DeepSpec's
    (the DSpark architecture without Markov and confidence heads)."""
    if not is_deepspec_drafter(_draft_snapshot(draft_model_path, "config.json")):
        return DFlash(
            target_model_path, draft_model_path, device, block_size, torch_dtype
        )
    if block_size is not None:
        raise ValueError(f"{draft_model_path} drafts blocks of its training size.")
    return DSpark(target_model_path, draft_model_path, device, torch_dtype=torch_dtype)


def _residual_sample(target_probs: torch.Tensor, draft_probs: torch.Tensor):
    """Sample from max(p - q, 0), or from p where that has no mass (DeepSpec)."""
    residual = (target_probs - draft_probs).clamp_min(0.0)
    mass = residual.sum(dim=-1, keepdim=True)
    residual = torch.where(mass <= 1e-8, target_probs, residual)
    return torch.multinomial(residual, 1)[:, 0]


class DSpark(nn.Module):
    def __init__(
        self,
        target_model_path: str,
        draft_model_path: str,
        device: torch.device | str,
        confidence_threshold: float = 0.0,
        torch_dtype: str = "bfloat16",
    ):
        """
        Args:
            target_model_path: Target (verifier) model id or path.
            draft_model_path: DeepSpec drafter id or path (DSpark or DFlash).
            confidence_threshold: Verify draft tokens only up to the first whose
                predicted acceptance probability is below this; 0 verifies the
                whole block, as in DeepSpec's evaluation.
        """
        super().__init__()
        path = _draft_snapshot(draft_model_path)
        if not is_deepspec_drafter(path):
            raise ValueError(f"{draft_model_path} is not a DeepSpec drafter.")
        self.target = AutoModelForCausalLM.from_pretrained(
            target_model_path, attn_implementation="sdpa", torch_dtype=torch_dtype
        )
        self.drafter, info = DSparkDrafter.from_pretrained(
            path,
            config=load_drafter_config(path),
            torch_dtype=torch_dtype,
            output_loading_info=True,
        )
        bad_keys = [k for v in info.values() for k in v]
        if bad_keys:
            raise ValueError(
                f"{draft_model_path} does not match DSparkDrafter: {bad_keys}"
            )
        target_config, draft_config = self.target.config, self.drafter.config
        if (
            target_config.rope_theta != draft_config.rope_theta
            or max(self.drafter.target_layer_ids) >= target_config.num_hidden_layers - 1
            or not torch.equal(
                self.drafter.embed_tokens.weight,
                self.target.get_input_embeddings().weight,
            )
        ):
            raise ValueError(
                f"{draft_model_path} was not trained for {target_model_path}."
            )
        if confidence_threshold > 0 and self.drafter.confidence_head is None:
            raise ValueError(f"{draft_model_path} has no confidence head.")
        self.confidence_threshold = confidence_threshold
        self.to(device).eval()
        self.requires_grad_(False)

    def _draft(self, block, context, position_ids, draft_cache, start, temperature):
        """Draft tokens after `block[:, 0]`, and their logits (DeepSpec's DSpark
        proposal)."""
        drafter = self.drafter
        hidden = drafter.block_hidden_states(
            block,
            context,
            position_ids[:, draft_cache.get_seq_length() : start + block.shape[1]],
            draft_cache,
        )
        draft_cache.crop(start)
        base_logits = drafter.lm_head(hidden)
        tokens, logits = [], []
        prev = block[:, 0]
        for i in range(block.shape[1]):
            logits.append(drafter.step_logits(base_logits[:, i], prev))
            prev = _sample(logits[-1], temperature)
            tokens.append(prev)
        tokens, logits = torch.stack(tokens, dim=1), torch.stack(logits, dim=1)
        if self.confidence_threshold > 0:
            prev_tokens = torch.cat([block[:, :1], tokens[:, :-1]], dim=1)
            confident = drafter.confidence_logits(hidden, prev_tokens).sigmoid()
            below = (confident[0] < self.confidence_threshold).nonzero()
            if len(below):
                tokens, logits = tokens[:, : below[0, 0]], logits[:, : below[0, 0]]
        return tokens, logits

    @torch.inference_mode()
    def generate(
        self,
        inputs: torch.LongTensor,
        generation_config: GenerationConfig | None = None,
        max_new_tokens: int | None = None,
    ) -> SpeculativeOutput:
        """DeepSpec block speculative decoding of one prompt of shape (1, length).

        The target verifies each block by speculative (rejection) sampling with
        the drafter's probabilities; greedy decoding accepts draft tokens while
        they equal the target's argmax.
        """
        generation_config = generation_config or GenerationConfig()
        max_new_tokens = max_new_tokens or generation_config.max_new_tokens
        temperature = 0.0
        if generation_config.do_sample:
            temperature = generation_config.temperature
            if not temperature > 0:
                raise ValueError("Sampling requires a temperature > 0.")
            if generation_config.top_k or (generation_config.top_p or 1.0) < 1.0:
                raise ValueError("DSpark sampling supports no top-k/top-p.")
        target, drafter, device = self.target, self.drafter, inputs.device
        block_size, prompt_len = drafter.block_size, inputs.shape[1]
        stop_token_ids = _stop_token_ids(generation_config, target, device)
        max_length = prompt_len + max_new_tokens
        output_ids = torch.full(
            (1, max_length + block_size + 1),
            drafter.mask_token_id,
            dtype=torch.long,
            device=device,
        )
        position_ids = torch.arange(output_ids.shape[1], device=device)[None]
        target_cache, draft_cache = DynamicCache(), DynamicCache()

        rounds = _Rounds(device)
        output = target(
            inputs,
            position_ids=position_ids[:, :prompt_len],
            past_key_values=target_cache,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=True,
        )
        output_ids[:, :prompt_len] = inputs
        output_ids[:, prompt_len] = _sample(output.logits[:, -1], temperature)
        context = drafter.context_features(output.hidden_states)
        # Position `start` holds the last verified token, which heads each block.
        start = prompt_len
        done = bool(torch.isin(output_ids[0, start], stop_token_ids))
        while not done and start < max_length:
            block = output_ids[:, start : start + block_size].clone()
            block[:, 1:] = drafter.mask_token_id
            with rounds.drafting():
                tokens, logits = self._draft(
                    block, context, position_ids, draft_cache, start, temperature
                )
            num_drafted = tokens.shape[1]
            output = target(
                torch.cat([block[:, :1], tokens], dim=1),
                position_ids=position_ids[:, start : start + num_drafted + 1],
                past_key_values=target_cache,
                use_cache=True,
                output_hidden_states=True,
            )
            if temperature == 0:
                posterior = output.logits.argmax(dim=-1)
                matches = tokens == posterior[:, :num_drafted]
                accepted = int(matches.cumprod(dim=1).sum())
                next_token = posterior[:, accepted]
            else:
                target_probs = (output.logits.float() / temperature).softmax(dim=-1)
                draft_probs = (logits.float() / temperature).softmax(dim=-1)
                p = target_probs[:, :num_drafted].gather(-1, tokens[..., None])[..., 0]
                q = draft_probs.gather(-1, tokens[..., None])[..., 0].clamp_min(1e-8)
                accept = torch.rand_like(p) < (p / q).clamp(max=1.0)
                accepted = int(accept.long().cumprod(dim=1).sum())
                if accepted < num_drafted:
                    next_token = _residual_sample(
                        target_probs[:, accepted], draft_probs[:, accepted]
                    )
                else:
                    next_token = torch.multinomial(target_probs[:, -1], 1)[:, 0]
            output_ids[:, start + 1 : start + accepted + 1] = tokens[:, :accepted]
            output_ids[:, start + accepted + 1] = next_token
            rounds.add(
                num_drafted, _accepted_until_stop(tokens[0, :accepted], stop_token_ids)
            )
            start += accepted + 1
            target_cache.crop(start)
            context = drafter.context_features(output.hidden_states)[:, : accepted + 1]
            done = bool(
                torch.isin(output_ids[0, prompt_len : start + 1], stop_token_ids).any()
            )
        return rounds.output(
            _truncate(
                output_ids[:, : start + 1], prompt_len, max_new_tokens, stop_token_ids
            )
        )
