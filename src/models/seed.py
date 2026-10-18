"""SEED: speculative decoding with a LoRA drafter on top of a frozen target LLM.

The target is the unmodified pretrained causal LM; it verifies drafted tokens.
The drafter shares the target's token embeddings and reuses copies of its last
``num_draft_layers`` decoder layers, final norm, and LM head, each wrapped with
LoRA. Drafter layer ``i`` attends to the target's KV cache at layer ``i`` for
already-verified tokens, plus to its own states for the tokens of the current
draft.

Training uses blocks of ``block_size`` tokens: a drafter position attends to the
target's states of all earlier blocks and causally to drafter states within its
own block. Only the LoRA parameters are trained.
"""

import copy
import math
import time
from dataclasses import dataclass

import torch
from torch import nn
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, GenerationConfig, StoppingCriteriaList
from transformers.cache_utils import DynamicCache
from transformers.generation import EosTokenCriteria, MaxLengthCriteria

from src.models.common import LMOutput, loss_mask
from src.models.lora import LoRALinear, add_lora_adapters, merge_lora_adapters


def additive_mask(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert a 0/1 attention mask to an additive one (0 or the dtype's min)."""
    return torch.where((mask == 0.0).bool(), torch.finfo(dtype).min, 0.0).to(dtype)


def cache_length(past_key_values: DynamicCache) -> int:
    """Longest sequence stored by any layer of the cache."""
    return max(
        (key.shape[-2] for key in past_key_values.key_cache if key.shape[0] > 0),
        default=0,
    )


def drafter_block_mask(seq_len: int, block_size: int) -> torch.BoolTensor:
    """Training attention mask of shape (seq_len, 2 * seq_len).

    Queries are drafter positions. Keys are the target's states (first
    ``seq_len`` columns) followed by the drafter's own states (last ``seq_len``).
    """
    q_idx = torch.arange(seq_len)[:, None]
    kv_idx = torch.arange(seq_len * 2)[None, :]
    is_drafter_kv = kv_idx >= seq_len
    block_q = q_idx // block_size
    block_kv = torch.where(
        is_drafter_kv, (kv_idx - seq_len) // block_size, kv_idx // block_size
    )
    earlier_target_blocks = (block_q > block_kv) & ~is_drafter_kv
    causal_in_own_block = (
        (block_q == block_kv) & is_drafter_kv & (q_idx >= (kv_idx - seq_len))
    )
    return earlier_target_blocks | causal_in_own_block


class Drafter(nn.Module):
    """LoRA copies of the target's top decoder layers, final norm, and LM head.

    The frozen base weights are shared with the target (never copied or
    modified), even when its LM head is tied to the input embedding, until
    `merge_lora` gives the drafter its own merged copy for inference.
    """

    def __init__(
        self,
        target: nn.Module,
        num_layers: int,
        lora_rank: int,
        lora_alpha: float,
        lora_dropout: float,
    ):
        super().__init__()
        total_layers = len(target.model.layers)
        if not 1 <= num_layers <= total_layers:
            raise ValueError(f"num_draft_layers must be between 1 and {total_layers}.")
        shared = {id(p): p for p in target.parameters()}  # deepcopy memo
        self.layers = nn.ModuleList(
            [
                copy.deepcopy(target.model.layers[i], shared)
                for i in range(total_layers - num_layers, total_layers)
            ]
        )
        self.norm = copy.deepcopy(target.model.norm, shared)
        self.rotary_emb = copy.deepcopy(target.model.rotary_emb, shared)
        for layer in self.layers:
            add_lora_adapters(layer, lora_rank, lora_alpha, lora_dropout)
        self.lm_head = LoRALinear(target.lm_head, lora_rank, lora_alpha, lora_dropout)

    def merge_lora(self) -> None:
        """Fold the adapters into separate drafter weights in the target's dtype,
        so drafting runs plain linear layers. For inference only: the drafter
        can no longer be trained or saved as LoRA weights."""
        merge_lora_adapters(self)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        past_key_values: DynamicCache,
        truncate_cache: bool = True,
    ) -> torch.Tensor:
        """Return logits for embedded tokens, attending to ``past_key_values``.

        Each layer appends its keys/values to the cache of the target layer it
        was copied from; ``truncate_cache`` removes them again afterwards.
        """
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            layer_idx = layer.self_attn.layer_idx
            prev_len = (
                past_key_values[layer_idx][0].shape[-2]
                if len(past_key_values) > layer_idx
                else 0
            )
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=False,
                use_cache=True,
                cache_position=position_ids[0],
                position_embeddings=position_embeddings,
            )[0]
            if truncate_cache:
                past_key_values.key_cache[layer_idx] = past_key_values.key_cache[
                    layer_idx
                ][..., :prev_len, :]
                past_key_values.value_cache[layer_idx] = past_key_values.value_cache[
                    layer_idx
                ][..., :prev_len, :]
        return self.lm_head(self.norm(hidden_states))


@dataclass
class SpeculativeOutput:
    """Result of `SEED.generate` (and of the EAGLE-3 / DFlash baselines)."""

    sequences: torch.LongTensor  # prompt followed by the generated tokens
    num_drafted: int  # draft tokens proposed
    num_accepted: int  # draft tokens accepted (incl. reused tree tokens)
    accepted_lengths: list[int]  # accepted tokens per verification round
    draft_lengths: list[int]  # drafted tokens per verification round
    position_attempts: list[int]  # per draft position: rounds that drafted it
    position_accepts: list[int]  # per draft position: rounds that accepted it
    drafting_time: float  # seconds spent drafting
    total_time: float  # seconds for the whole call


class SEED(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        max_seq_length: int,
        block_size: int,
        num_draft_layers: int = 2,
        lora_rank: int = 32,
        lora_alpha: float = 64.0,
        lora_dropout: float = 0.0,
        torch_dtype: str | None = "bfloat16",
        train_on_context: bool = False,
    ):
        """
        Args:
            pretrained_model_name_or_path: Target (verifier) model id or path.
            max_seq_length: Training sequence length (inputs are padded to it).
            block_size: Training block size of the drafter attention mask.
            num_draft_layers: Number of top target layers the drafter reuses.
            lora_rank, lora_alpha, lora_dropout: LoRA hyperparameters.
            torch_dtype: dtype to load the target weights in.
            train_on_context: Also train on prompt tokens (`context_mask == 1`).
        """
        super().__init__()
        if lora_rank < 1 or not math.isfinite(lora_alpha) or lora_alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive.")
        if not 0 <= lora_dropout < 1:
            raise ValueError("LoRA dropout must be in [0, 1).")
        self.max_seq_length = max_seq_length
        self.block_size = block_size
        self.train_on_context = train_on_context
        self.target = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            attn_implementation="sdpa",
            torch_dtype=torch_dtype,
        )
        self.target.requires_grad_(False)
        self.target.eval()
        self.drafter = Drafter(
            self.target, num_draft_layers, lora_rank, lora_alpha, lora_dropout
        )
        self.register_buffer(
            "block_mask",
            drafter_block_mask(max_seq_length, block_size),
            persistent=False,
        )

    def train(self, mode: bool = True) -> "SEED":
        super().train(mode)
        self.target.eval()  # target dropout must never affect the KV cache
        return self

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor | None = None,
        context_mask: torch.LongTensor | None = None,
    ) -> LMOutput:
        """Next-token loss of the drafter, given the target's KV cache."""
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if context_mask is None:
            context_mask = torch.zeros_like(attention_mask)
        if torch.is_floating_point(attention_mask):
            attention_mask = attention_mask.to(torch.int)
            context_mask = context_mask.to(torch.int)
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len).to(input_ids.device)[None, :]

        # The target sees the plain causal mask (as at inference time).
        causal = torch.ones(
            seq_len, seq_len, device=input_ids.device, dtype=torch.bool
        ).tril()
        target_mask = additive_mask(
            causal[None, None] & attention_mask[:, None, None, :].bool(), torch.float
        )
        with torch.no_grad():
            past_key_values = self.target.model(
                input_ids=input_ids,
                attention_mask=target_mask,
                position_ids=position_ids,
                cache_position=position_ids[0],
                past_key_values=None,
                use_cache=True,
            ).past_key_values

        drafter_mask = (
            self.block_mask[None, ...]
            & attention_mask.repeat(1, 2)[:, None, :]
            & attention_mask[..., None]
        )[:, None, ...]
        logits = self.drafter(
            self.target.model.embed_tokens(input_ids),
            attention_mask=additive_mask(drafter_mask, torch.float),
            position_ids=position_ids,
            past_key_values=past_key_values,
        )

        targets = input_ids[:, 1:]
        mask = loss_mask(
            attention_mask,
            context_mask,
            include_context=self.training and self.train_on_context,
        )[:, 1:]
        shifted_logits = logits[:, :-1, :]
        flat_mask = mask.contiguous().view(-1).bool()
        loss = nn.functional.cross_entropy(
            shifted_logits.contiguous().view(-1, logits.size(-1))[flat_mask],
            targets.contiguous().view(-1)[flat_mask],
        )
        with torch.no_grad():
            log_probs = torch.log_softmax(shifted_logits, dim=-1)
            nlls = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1) * mask
        return LMOutput(loss=loss, logits=logits, nlls=nlls, loss_mask=mask)

    def _target_forward(
        self,
        input_ids: torch.LongTensor,
        past_key_values: DynamicCache,
        position_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the target on new tokens, extending the cache; returns hidden states.

        By default the tokens follow the cache and attend causally.
        """
        if position_ids is None:
            start = cache_length(past_key_values)
            position_ids = torch.arange(
                start, start + input_ids.shape[1], device=input_ids.device
            )[None, :]
        return self.target.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=position_ids[0],
            past_key_values=past_key_values,
            use_cache=True,
        ).last_hidden_state

    def _draft_logits(
        self, input_ids: torch.LongTensor, past_key_values: DynamicCache
    ) -> torch.Tensor:
        """Drafter logits for tokens following the cache (which keeps their states)."""
        start = cache_length(past_key_values)
        end = start + input_ids.shape[-1]
        position_ids = torch.arange(start, end, device=input_ids.device)[None, :]
        # The tokens attend to the whole cache and to each other. A single token
        # needs no mask (SDPA then picks its fastest kernel); without a mask,
        # SDPA would treat several tokens as causal.
        attention_mask = None
        if input_ids.shape[1] > 1:
            attention_mask = additive_mask(
                torch.ones(
                    (input_ids.shape[0], 1, input_ids.shape[1], end),
                    device=input_ids.device,
                ),
                self.target.dtype,
            )
        return self.drafter(
            self.target.model.embed_tokens(input_ids),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            truncate_cache=False,
        )

    @staticmethod
    def _speculative_sample(draft_tokens, draft_probs, target_probs):
        """Sample an accepted prefix and one residual/bonus token.

        q has shape (batch, draft_length, vocab); p includes one extra position.
        Implements Algorithm 1 of Leviathan et al. (2023), using fp32 probabilities.
        """
        q = draft_probs.float()
        p = target_probs.float()
        q_token = q.gather(-1, draft_tokens.unsqueeze(-1)).squeeze(-1)
        p_token = p[:, :-1].gather(-1, draft_tokens.unsqueeze(-1)).squeeze(-1)
        # Equivalent to U < min(1, p(x)/q(x)), without division by tiny q(x).
        accepted = torch.rand_like(q_token) * q_token < p_token
        lengths = accepted.long().cumprod(dim=1).sum(dim=1)
        rows = torch.arange(q.shape[0], device=q.device)
        correction_probs = p[rows, lengths].clone()
        rejected = lengths < draft_tokens.shape[1]
        correction_probs[rejected] = (
            correction_probs[rejected] - q[rows[rejected], lengths[rejected]]
        ).clamp_min(0)
        # multinomial normalizes the positive residual; when every proposal is
        # accepted, this is already the target distribution for the bonus token.
        correction = torch.multinomial(correction_probs, 1).squeeze(-1)
        return lengths, correction

    @torch.inference_mode()
    def generate(
        self,
        inputs: torch.LongTensor,
        generation_config: GenerationConfig | None = None,
        stopping_criteria: StoppingCriteriaList | None = None,
        max_new_tokens: int | None = None,
        max_length: int | None = None,
        block_size: int | None = None,
        conf_seg: bool = True,
        tree_attn: bool = True,
        disable_pbar: bool = False,
    ) -> SpeculativeOutput:
        """Speculative decoding: the drafter proposes, the target verifies.

        With ``generation_config.do_sample``, this is exact speculative sampling
        from the target distribution at ``generation_config.temperature``;
        otherwise it matches greedy decoding with the target.

        Args:
            inputs: Prompt token ids of shape (1, prompt_length).
            stopping_criteria: Evaluated on the generated tokens only.
            block_size: Draft length per round when ``conf_seg`` is off
                (default: ``max_new_tokens``).
            conf_seg: End a draft early once drafter confidence drops below 0.7
                (drafts up to 100 tokens).
            tree_attn: Also verify the drafter's top-k alternatives at each
                position (k grows as confidence drops). A verified alternative
                is reused, together with the target's continuation of it.
        """
        generation_config = generation_config or GenerationConfig()
        do_sample = bool(generation_config.do_sample)
        temperature = (
            float(getattr(generation_config, "temperature", 1.0)) if do_sample else 1.0
        )
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError(
                "Stochastic SEED decoding requires a finite temperature > 0."
            )
        if inputs.shape[0] != 1:
            raise ValueError("Batched SEED speculative decoding is not supported.")
        if inputs.shape[1] == 0:
            raise ValueError("SEED speculative decoding requires a nonempty prompt.")
        device = inputs.device
        if max_new_tokens is None:
            max_new_tokens = generation_config.max_new_tokens
        if max_new_tokens is None:
            max_new_tokens = (
                max_length or generation_config.max_length
            ) - inputs.shape[1]
        if block_size is None:
            block_size = max_new_tokens
        if block_size < 1 or max_new_tokens < 0:
            raise ValueError(
                "block_size must be positive and max_new_tokens nonnegative."
            )

        prompt_len = inputs.shape[1]
        limit = prompt_len + max_new_tokens
        # Positions past `current_idx` are only placeholders; they are never read.
        samples = torch.cat([inputs, inputs.new_zeros((1, max_new_tokens))], dim=-1)
        current_idx = prompt_len
        draft_lengths, accepted_lengths = [], []
        attempts, accepts = [0] * block_size, [0] * block_size
        total_drafted = total_accepted = 0
        draft_events = []
        drafting_time = 0.0
        use_cuda_events = device.type == "cuda"
        if use_cuda_events:
            overall_start = torch.cuda.Event(enable_timing=True)
            overall_end = torch.cuda.Event(enable_timing=True)
            overall_start.record()
        else:
            overall_start = time.perf_counter()

        # The cache always holds target states for the tokens before `current_idx - 1`.
        past_key_values = DynamicCache()
        if prompt_len > 1 and max_new_tokens > 0:
            self._target_forward(inputs[:, :-1], past_key_values)

        def probabilities(logits):
            return (logits.float() / temperature).softmax(dim=-1)

        # EOS and max-length criteria are checked on the host from token ids the
        # loop already fetched; any other criterion runs on the generated tokens.
        stop_len, stop_ids = limit, set()
        other_criteria = StoppingCriteriaList()
        for criterion in stopping_criteria or []:
            if isinstance(criterion, EosTokenCriteria):
                stop_ids.update(criterion.eos_token_id.view(-1).tolist())
            elif isinstance(criterion, MaxLengthCriteria):
                stop_len = min(stop_len, prompt_len + criterion.max_length)
            else:
                other_criteria.append(criterion)

        def should_stop(end, last_token):
            """Whether to stop after `samples[:, end - 1]`, which is `last_token`."""
            if end >= stop_len or last_token in stop_ids:
                return True
            return bool(other_criteria) and bool(
                torch.any(
                    other_criteria(input_ids=samples[:, prompt_len:end], scores=None)
                )
            )

        pbar = tqdm(
            total=max_new_tokens, desc="Tokens", leave=True, disable=disable_pbar
        )
        while current_idx < limit:
            start_idx = current_idx
            max_draft_len = min(100 if conf_seg else block_size, limit - start_idx)
            if use_cuda_events:
                draft_start = torch.cuda.Event(enable_timing=True)
                draft_end = torch.cuda.Event(enable_timing=True)
                draft_start.record()
            else:
                draft_start = time.perf_counter()

            # --- Draft: one token at a time, starting from the last verified one.
            draft_probs, draft_tokens, candidates = [], [], []
            for step in range(max_draft_len):
                cur_pos = start_idx + step
                logits = self._draft_logits(
                    samples[:, cur_pos - 1 : cur_pos], past_key_values
                )[:, -1]
                q = probabilities(logits)
                token = (
                    torch.multinomial(q, 1)
                    if do_sample
                    else logits.argmax(dim=-1, keepdim=True)
                )
                samples[:, cur_pos : cur_pos + 1] = token
                draft_probs.append(q)
                # One device-to-host sync per step: the token, then the top-10
                # probabilities (the first is the confidence) and their ids.
                top_probs, top_ids = q.topk(min(10, q.shape[-1]), dim=-1)
                fetched = torch.cat(
                    [token[0].double(), top_probs[0].double(), top_ids[0].double()]
                ).tolist()
                num_top = top_probs.shape[-1]
                token_id, confidence = int(fetched[0]), fetched[1]
                draft_tokens.append(token_id)
                if tree_attn:
                    k = (
                        1
                        if confidence > 0.95
                        else (
                            3 if confidence > 0.8 else (5 if confidence > 0.5 else 10)
                        )
                    )
                    candidates.extend(
                        (step, int(top_id))
                        for top_id in fetched[1 + num_top : 1 + num_top + k]
                        if int(top_id) != token_id
                    )
                if should_stop(cur_pos + 1, token_id) or (
                    conf_seg and confidence < 0.7
                ):
                    break

            draft_len = len(draft_probs)
            total_drafted += draft_len
            # Remove the drafter's speculative entries before verification.
            for layer in range(len(past_key_values)):
                if past_key_values.key_cache[layer].numel() == 0:
                    continue
                past_key_values.key_cache[layer] = past_key_values.key_cache[layer][
                    ..., : start_idx - 1, :
                ]
                past_key_values.value_cache[layer] = past_key_values.value_cache[layer][
                    ..., : start_idx - 1, :
                ]
            if use_cuda_events:
                draft_end.record()
                draft_events.append((draft_start, draft_end))
            else:
                drafting_time += time.perf_counter() - draft_start

            # --- Verify: score the last verified token + draft (+ tree
            # alternatives, each attending to the draft prefix before it).
            main_inputs = samples[:, start_idx - 1 : start_idx + draft_len]
            main_len = main_inputs.shape[1]
            target_inputs, position_ids, attention_mask = main_inputs, None, None
            if candidates:
                cand_tokens, cand_steps = torch.tensor(
                    [
                        [token for _, token in candidates],
                        [step for step, _ in candidates],
                    ],
                    device=device,
                )
                target_inputs = torch.cat([main_inputs, cand_tokens[None]], dim=1)
                past_len = start_idx - 1
                position_ids = torch.cat(
                    [
                        torch.arange(past_len, past_len + main_len, device=device),
                        start_idx + cand_steps,
                    ]
                )[None]
                total_len = target_inputs.shape[1]
                # Over the new tokens: the draft rows are causal; a candidate for
                # step s sees the inputs of steps 0..s (draft columns 0..s) and
                # itself. Everything attends to the whole cache.
                cols = torch.arange(total_len, device=device)
                visible = torch.cat(
                    [
                        cols[:main_len, None] >= cols,
                        (cols <= cand_steps[:, None]) | (cols == cols[main_len:, None]),
                    ]
                )
                attention_mask = torch.zeros(
                    (1, 1, total_len, past_len + total_len),
                    dtype=self.target.dtype,
                    device=device,
                )
                attention_mask[0, 0, :, past_len:].masked_fill_(~visible, float("-inf"))
            hidden_states = self._target_forward(
                target_inputs, past_key_values, position_ids, attention_mask
            )
            target_logits = self.target.lm_head(hidden_states)
            target_scores = target_logits[:, :main_len]
            if do_sample:
                lengths, correction = self._speculative_sample(
                    main_inputs[:, 1:],
                    torch.stack(draft_probs, dim=1),
                    probabilities(target_scores),
                )
            else:
                target_tokens = target_scores.argmax(dim=-1)
                lengths = (
                    (main_inputs[:, 1:] == target_tokens[:, :-1])
                    .long()
                    .cumprod(dim=-1)
                    .sum(dim=-1)
                )
                correction = target_tokens.gather(1, lengths[:, None]).squeeze(1)
            accepted, correction_id = torch.cat([lengths, correction]).tolist()
            final_accepted = accepted
            current_idx = start_idx + accepted
            finished = accepted > 0 and should_stop(
                current_idx, draft_tokens[accepted - 1]
            )
            branch_index = None
            if not finished:
                samples[:, current_idx] = correction
                current_idx += 1
                if accepted < draft_len:
                    branch_index = next(
                        (
                            i
                            for i, (step, token) in enumerate(candidates)
                            if step == accepted and token == correction_id
                        ),
                        None,
                    )
                if branch_index is not None:
                    # The reused branch token counts as accepted; its target
                    # continuation is a bonus token.
                    final_accepted += 1
                finished = should_stop(current_idx, correction_id)
                if branch_index is not None and not finished:
                    branch_scores = target_logits[:, main_len + branch_index]
                    bonus = (
                        torch.multinomial(probabilities(branch_scores), 1)
                        if do_sample
                        else branch_scores.argmax(dim=-1, keepdim=True)
                    )
                    samples[:, current_idx : current_idx + 1] = bonus
                    current_idx += 1
                    finished = should_stop(current_idx, bonus.item())

            # Correction/bonus tokens do not count as accepted (except a reused
            # tree token, above); terminal rounds are included.
            total_accepted += final_accepted
            accepted_lengths.append(final_accepted)
            draft_lengths.append(draft_len)
            while len(attempts) < draft_len:
                attempts.append(0)
                accepts.append(0)
            for pos in range(draft_len):
                attempts[pos] += 1
                accepts[pos] += int(pos < final_accepted)

            # Keep target states for the accepted prefix and any reused tree
            # branch, excluding the final target continuation and all siblings.
            indices = torch.arange(start_idx + accepted, device=device)
            if branch_index is not None:
                indices = torch.cat(
                    [
                        indices,
                        torch.full(
                            (1,), start_idx - 1 + main_len + branch_index, device=device
                        ),
                    ]
                )
            for layer in range(len(past_key_values)):
                past_key_values.key_cache[layer] = past_key_values.key_cache[
                    layer
                ].index_select(-2, indices)
                past_key_values.value_cache[layer] = past_key_values.value_cache[
                    layer
                ].index_select(-2, indices)

            pbar.update(current_idx - start_idx)
            if finished:
                break
        pbar.close()

        if use_cuda_events:
            overall_end.record()
            torch.cuda.synchronize()
            total_time = overall_start.elapsed_time(overall_end) / 1000.0
            drafting_time = (
                sum(start.elapsed_time(end) for start, end in draft_events) / 1000.0
            )
        else:
            total_time = time.perf_counter() - overall_start
        return SpeculativeOutput(
            sequences=samples[:, :current_idx],
            num_drafted=total_drafted,
            num_accepted=total_accepted,
            accepted_lengths=accepted_lengths,
            draft_lengths=draft_lengths,
            position_attempts=attempts,
            position_accepts=accepts,
            drafting_time=drafting_time,
            total_time=total_time,
        )
