"""SEED-LoRA-Diffusion: a SEED drafter that proposes a whole block in one step.

The model is SEED's (frozen target, LoRA drafter on copies of its top layers);
only the drafter's inputs and attention change. Instead of drafting one token at
a time, the drafter fills a block of [MASK] tokens in a single forward pass
(one-step block diffusion):

- A block of ``block_size`` positions holds the last verified token (whose
  target states are not in the KV cache yet), followed by ``block_size - 1``
  [MASK] tokens.
- Drafter positions attend to the target's KV cache for all earlier tokens and
  bidirectionally to each other.
- Each position predicts the token after it, as in SEED: the verified token
  predicts draft token 1, the first [MASK] draft token 2, and so on, so a block
  drafts ``block_size`` tokens.

Training reuses SEED's blocks of the padded sequence (the drafter's keys are the
target's states of earlier blocks followed by the drafter's own states, i.e. a
context of 2 * seq_len) with every token except the first of each block replaced
by [MASK], and SEED's next-token cross-entropy loss. Optionally, as in DFlash
(Eq. 4), the loss of draft token k of a block is weighted by
exp(-(k - 1) / gamma), so the early draft tokens, on which acceptance of all
later ones depends, count more.

[MASK] is the target's frozen embedding row ``mask_token_id`` (default 151669,
DFlash's choice: a row of Qwen3's embedding matrix that its tokenizer never
produces).
"""

import math
import time

import torch
from tqdm.auto import tqdm
from transformers import GenerationConfig, StoppingCriteriaList
from transformers.cache_utils import DynamicCache
from transformers.generation import EosTokenCriteria, MaxLengthCriteria

from src.models.seed import SEED, SpeculativeOutput


def drafter_diffusion_mask(seq_len: int, block_size: int) -> torch.BoolTensor:
    """Training attention mask of shape (seq_len, 2 * seq_len).

    As `drafter_block_mask` (target states of all earlier blocks, then drafter
    states of the own block), except that drafter positions see every drafter
    state of their block, not only the earlier ones.
    """
    q_block = torch.arange(seq_len)[:, None] // block_size
    kv_block = torch.arange(seq_len)[None, :] // block_size
    return torch.cat([q_block > kv_block, q_block == kv_block], dim=1)


def crop_cache(past_key_values: DynamicCache, length: int) -> None:
    """Keep the first ``length`` positions of every nonempty cache layer."""
    for layer in range(len(past_key_values)):
        if past_key_values.key_cache[layer].numel() == 0:
            continue
        past_key_values.key_cache[layer] = past_key_values.key_cache[layer][
            ..., :length, :
        ]
        past_key_values.value_cache[layer] = past_key_values.value_cache[layer][
            ..., :length, :
        ]


class SEEDDiffusion(SEED):
    def __init__(
        self,
        mask_token_id: int = 151669,
        loss_decay_gamma: float | None = None,
        **kwargs,
    ):
        """
        Args:
            mask_token_id: Token whose (frozen) target embedding is the [MASK]
                input of the drafter.
            loss_decay_gamma: DFlash's loss decay: draft token k of a block
                (k = 1 for the prediction at the last verified token) gets
                weight exp(-(k - 1) / gamma). None trains all positions
                equally. DFlash uses 7, 5, 4 for its blocks of 16, 10, 8,
                which draft 15, 9, 7 tokens; a block here drafts ``block_size``.
            kwargs: Arguments of `SEED`; ``block_size`` is the number of tokens
                per training block (the last verified token and its [MASK]s).
        """
        if loss_decay_gamma is not None and not loss_decay_gamma > 0:
            raise ValueError("loss_decay_gamma must be positive (or None).")
        super().__init__(**kwargs)
        num_embeddings = self.target.get_input_embeddings().num_embeddings
        if not 0 <= mask_token_id < num_embeddings:
            raise ValueError(f"mask_token_id must be in [0, {num_embeddings}).")
        self.mask_token_id = mask_token_id
        self.loss_decay_gamma = loss_decay_gamma
        self.block_mask = drafter_diffusion_mask(self.max_seq_length, self.block_size)

    def _drafter_input_ids(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        """Keep the first token of each block (the last verified token at
        inference time) and replace the others with [MASK]."""
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        return input_ids.masked_fill(
            positions % self.block_size != 0, self.mask_token_id
        )

    def _drafter_token_mask(self, attention_mask: torch.LongTensor) -> torch.LongTensor:
        """Every position of a block that starts with a real token. Padding at
        the end of such a block is then [MASK] input like the rest of the block,
        as at inference time, where the drafter always fills a whole block."""
        heads = attention_mask[:, :: self.block_size]
        return heads.repeat_interleave(self.block_size, dim=1)[
            :, : attention_mask.shape[1]
        ]

    def _loss_weights(
        self, num_targets: int, device: torch.device
    ) -> torch.Tensor | None:
        """The output at position p predicts draft token k = p % block_size + 1
        of its block, weighted exp(-(k - 1) / loss_decay_gamma)."""
        if self.loss_decay_gamma is None:
            return None
        offsets = torch.arange(num_targets, device=device) % self.block_size
        return torch.exp(-offsets.float() / self.loss_decay_gamma)

    @torch.inference_mode()
    def generate(
        self,
        inputs: torch.LongTensor,
        generation_config: GenerationConfig | None = None,
        stopping_criteria: StoppingCriteriaList | None = None,
        max_new_tokens: int | None = None,
        max_length: int | None = None,
        block_size: int | None = None,
        disable_pbar: bool = False,
    ) -> SpeculativeOutput:
        """Speculative decoding with one drafter forward pass per round.

        Each round the drafter fills ``[last verified token, [MASK] x
        (block_size - 1)]``, proposing ``block_size`` tokens (fewer near
        ``max_new_tokens``; a draft also ends at its first stop token), and the
        target verifies them in one forward pass, as in `SEED.generate`.

        With ``generation_config.do_sample``, this is exact speculative sampling
        from the target distribution at ``generation_config.temperature``: draft
        token i is sampled from the drafter's distribution at position i, which
        is the proposal probability the acceptance test uses. Otherwise it
        matches greedy decoding with the target.

        Args:
            inputs: Prompt token ids of shape (1, prompt_length).
            stopping_criteria: Evaluated on the generated tokens only.
            block_size: Draft tokens per round (default: the training block size).
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
            block_size = self.block_size
        if block_size < 1 or max_new_tokens < 0:
            raise ValueError(
                "block_size must be positive and max_new_tokens nonnegative."
            )

        prompt_len = inputs.shape[1]
        limit = prompt_len + max_new_tokens
        # Positions past `current_idx` are only placeholders; they are never read.
        samples = torch.cat([inputs, inputs.new_zeros((1, max_new_tokens))], dim=-1)
        current_idx = prompt_len
        # Drafter input: the last verified token (set each round), then [MASK]s.
        block = torch.full_like(inputs[:, :1], self.mask_token_id).repeat(1, block_size)
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
            draft_len = min(block_size, limit - start_idx)
            if use_cuda_events:
                draft_start = torch.cuda.Event(enable_timing=True)
                draft_end = torch.cuda.Event(enable_timing=True)
                draft_start.record()
            else:
                draft_start = time.perf_counter()

            # --- Draft: the whole block in one drafter pass. The full block is
            # always used (as in training), even when fewer tokens are needed.
            block[:, 0] = samples[:, start_idx - 1]
            logits = self._draft_logits(block, past_key_values)[:, :draft_len]
            crop_cache(past_key_values, start_idx - 1)  # drop the drafter's states
            draft_probs = probabilities(logits)
            draft = (
                torch.multinomial(draft_probs[0], 1).T
                if do_sample
                else logits.argmax(dim=-1)
            )
            samples[:, start_idx : start_idx + draft_len] = draft
            # One device-to-host sync; the draft ends at its first stop token.
            draft_tokens = draft[0].tolist()
            for step, token_id in enumerate(draft_tokens):
                if should_stop(start_idx + step + 1, token_id):
                    draft_len = step + 1
                    break
            total_drafted += draft_len
            if use_cuda_events:
                draft_end.record()
                draft_events.append((draft_start, draft_end))
            else:
                drafting_time += time.perf_counter() - draft_start

            # --- Verify: score the last verified token followed by the draft.
            main_inputs = samples[:, start_idx - 1 : start_idx + draft_len]
            target_logits = self.target.lm_head(
                self._target_forward(main_inputs, past_key_values)
            )
            if do_sample:
                lengths, correction = self._speculative_sample(
                    main_inputs[:, 1:],
                    draft_probs[:, :draft_len],
                    probabilities(target_logits),
                )
            else:
                target_tokens = target_logits.argmax(dim=-1)
                lengths = (
                    (main_inputs[:, 1:] == target_tokens[:, :-1])
                    .long()
                    .cumprod(dim=-1)
                    .sum(dim=-1)
                )
                correction = target_tokens.gather(1, lengths[:, None]).squeeze(1)
            accepted, correction_id = torch.cat([lengths, correction]).tolist()
            current_idx = start_idx + accepted
            finished = accepted > 0 and should_stop(
                current_idx, draft_tokens[accepted - 1]
            )
            if not finished:
                samples[:, current_idx] = correction
                current_idx += 1
                finished = should_stop(current_idx, correction_id)

            # Correction/bonus tokens do not count as accepted; terminal rounds
            # are included.
            total_accepted += accepted
            accepted_lengths.append(accepted)
            draft_lengths.append(draft_len)
            for pos in range(draft_len):
                attempts[pos] += 1
                accepts[pos] += int(pos < accepted)

            # Keep target states for the verified token and the accepted draft.
            crop_cache(past_key_values, start_idx + accepted)

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
