"""lm-eval-harness adapter for SEED, EAGLE-3/DFlash baselines, and autoregressive
(Hugging Face) models.

Prompts are rendered zero-shot with the tokenizer's chat template. Generations,
throughput, and draft acceptance statistics are printed; responses are not scored.
"""

import re
from typing import Any

import accelerate
import numpy as np
import torch
from lm_eval.api.model import LM
from tqdm import tqdm
from transformers import (
    GenerationConfig,
    PreTrainedTokenizerBase,
    StoppingCriteria,
    StoppingCriteriaList,
)
from transformers.generation import EosTokenCriteria, MaxLengthCriteria

from src.models.baselines import EAGLE3, DFlash
from src.models.seed import SEED

MATH_INSTRUCTION = (
    "Please reason step by step, and put your final answer within $\\boxed{}$."
)


class RegexStoppingCriteria(StoppingCriteria):
    """Stop once the decoded text contains a match of `pattern`."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, pattern: str):
        self.tokenizer = tokenizer
        self.pattern = pattern

    def __call__(
        self, input_ids: torch.LongTensor, scores=None, **kwargs
    ) -> torch.BoolTensor:
        if input_ids.numel() == 0:
            return torch.tensor([False], device=input_ids.device, dtype=torch.bool)
        texts = (
            self.tokenizer.batch_decode(input_ids)
            if input_ids.ndim > 1
            else [self.tokenizer.decode(input_ids)]
        )
        matches = [len(re.findall(self.pattern, text)) > 0 for text in texts]
        return torch.tensor(matches, device=input_ids.device, dtype=torch.bool)


def build_stopping_criteria(
    names: list[str], tokenizer: PreTrainedTokenizerBase, max_new_tokens: int
) -> StoppingCriteriaList | None:
    """SEED stopping criteria (they see only the generated tokens).

    Names: `eos`, `max_new_tokens`, and `boxed_answer` (stop after the first
    `\\boxed{...}`).
    """
    factories = {
        "eos": lambda: EosTokenCriteria(eos_token_id=tokenizer.eos_token_id),
        "max_new_tokens": lambda: MaxLengthCriteria(max_length=max_new_tokens),
        "boxed_answer": lambda: RegexStoppingCriteria(tokenizer, r"\\boxed\{.*?\}"),
    }
    unknown = set(names) - set(factories)
    if unknown:
        raise ValueError(
            f"Unknown stopping criteria {sorted(unknown)}; use {list(factories)}."
        )
    return (
        StoppingCriteriaList([factories[name]() for name in names]) if names else None
    )


def truncate_response(response: str, until: list[str], eos_token: str) -> str:
    """Cut the response at the task's stop strings and at end-of-turn tokens."""
    for stop in [*until, "<|eot_id|>", eos_token]:
        if stop:
            response = response.split(stop)[0]
    return response


def _task_kind(doc: dict[str, Any]) -> str:
    if (
        "entry_point" in doc
        or "test_list" in doc
        or str(doc.get("task_id", "")).lower().startswith("mbpp")
    ):
        return "code"
    if "answer" in doc:  # GSM8K, MATH-500
        return "math"
    raise ValueError(f"Unsupported task document with fields {sorted(doc)}.")


class GenerationStats:
    """Running totals of generated/accepted tokens, throughput, and draft stats."""

    def __init__(self, throughput_warmup: int):
        self.throughput_warmup = throughput_warmup
        self.throughputs: list[float] = []
        self.generated = self.accepted = 0
        self.accepted_lengths: list[int] = []
        self.rounds = 0
        self.draft_lengths: list[int] = []
        self.position_attempts: list[int] = []
        self.position_accepts: list[int] = []
        self.drafting_time = self.total_time = 0.0

    def add(self, output: Any, num_new_tokens: int) -> None:
        if hasattr(output, "num_drafted"):  # SpeculativeOutput
            self.drafting_time += output.drafting_time
            self.total_time += output.total_time
            self.generated += output.num_drafted
            self.accepted += output.num_accepted
            self.accepted_lengths.extend(output.accepted_lengths)
            self.rounds += len(output.accepted_lengths)
            self.draft_lengths.extend(int(n) for n in output.draft_lengths)
            for totals, counts in (
                (self.position_attempts, output.position_attempts),
                (self.position_accepts, output.position_accepts),
            ):
                totals.extend([0] * (len(counts) - len(totals)))
                for i, count in enumerate(counts):
                    totals[i] += int(count)
        else:  # Autoregressive generation accepts one token per step.
            self.generated += num_new_tokens
            self.accepted += num_new_tokens
            self.accepted_lengths.extend([1] * num_new_tokens)
            self.rounds += num_new_tokens

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.generated if self.generated > 0 else 0.0

    @property
    def average_accepted_length(self) -> float:
        return np.sum(self.accepted_lengths) / self.rounds if self.rounds > 0 else 0.0

    def per_position(self) -> list[dict[str, Any]]:
        return [
            {
                "position": pos,
                "accept_count": int(acc),
                "attempt_count": int(att),
                "acceptance_rate": float(acc / att) if att > 0 else 0.0,
            }
            for pos, (acc, att) in enumerate(
                zip(self.position_accepts, self.position_attempts), start=1
            )
        ]

    def print_running(self, sample_idx: int, last_throughput: float) -> None:
        print(
            f"Total generated tokens: {self.generated}, Total accepted tokens: "
            f"{self.accepted}, Acceptance rate: {self.acceptance_rate:.2%}"
        )
        print(f"Average accepted length: {self.average_accepted_length:.2f}")
        if self.draft_lengths:
            print(f"Running avg draft length: {np.mean(self.draft_lengths):.2f}")
        if self.position_attempts:
            print(
                "Per-position acceptance rate: "
                + ", ".join(
                    f"p{row['position']}:{row['acceptance_rate']:.2%} "
                    f"({row['accept_count']}/{row['attempt_count']})"
                    for row in self.per_position()
                )
            )
        if sample_idx >= self.throughput_warmup:
            print(
                f"Thput (tok/s): {np.mean(self.throughputs):0.2f} +/- "
                f"{np.std(self.throughputs):0.2f}"
            )
        else:
            print(f"Thput (tok/s): {last_throughput:0.2f}")


class LMEvalAdapter(LM):
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        generation_config: GenerationConfig,
        max_new_tokens: int,
        stopping_criteria: StoppingCriteriaList | None = None,
        seed_decoding: dict[str, Any] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        throughput_warmup: int = 100,
        accelerator: accelerate.Accelerator | None = None,
    ):
        """
        Args:
            model: A `SEED`, `EAGLE3`, or `DFlash` model, or any model with
                Hugging Face `generate`.
            generation_config: Sampling settings (`do_sample`, `temperature`, ...).
                Used as given: the model's own generation defaults never apply.
            stopping_criteria: Used by SEED only; the baselines and Hugging Face
                generation stop via `generation_config` (EOS) and `max_new_tokens`.
            seed_decoding: Extra `SEED.generate` options (block size, ...).
            chat_template_kwargs: Passed to `tokenizer.apply_chat_template`.
            throughput_warmup: Samples excluded from the mean throughput.
        """
        super().__init__()
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.generation_config = generation_config
        self.max_new_tokens = max_new_tokens
        self.stopping_criteria = stopping_criteria
        self.seed_decoding = dict(seed_decoding or {})
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.throughput_warmup = throughput_warmup
        self.accelerator = accelerator  # lm-eval syncs ranks through it
        self._rank = accelerator.local_process_index if accelerator else 0
        self._world_size = accelerator.num_processes if accelerator else 1
        self.device = next(model.parameters()).device

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    def loglikelihood(self, requests):
        raise NotImplementedError

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def _prompt(self, context: str, kind: str) -> str:
        if kind == "code":
            user_prompt = context.strip()
        else:  # "Question: ...\nAnswer:" -> instruction + question
            user_prompt = context.replace("Question: ", f"{MATH_INSTRUCTION} ", 1)
            if user_prompt == context:
                user_prompt = f"{MATH_INSTRUCTION} {context}"
            user_prompt = re.sub(r"\nAnswer:\s*$", "", user_prompt).strip()
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
            **self.chat_template_kwargs,
        )

    def _generate(self, input_ids: torch.LongTensor) -> tuple[torch.LongTensor, Any]:
        if isinstance(self.model, SEED):
            output = self.model.generate(
                input_ids,
                generation_config=self.generation_config,
                stopping_criteria=self.stopping_criteria,
                max_new_tokens=self.max_new_tokens,
                disable_pbar=self.rank != 0,
                **self.seed_decoding,
            )
            return output.sequences, output
        if isinstance(self.model, EAGLE3 | DFlash):
            output = self.model.generate(
                input_ids,
                generation_config=self.generation_config,
                max_new_tokens=self.max_new_tokens,
            )
            return output.sequences, output
        sequences = self.model.generate(
            inputs=input_ids,
            attention_mask=torch.ones_like(input_ids),
            generation_config=self.generation_config,
            # Otherwise fields left at their defaults (e.g. do_sample=False) are
            # replaced by the model's generation_config.json (Qwen3 samples).
            use_model_defaults=False,
            max_length=None,
            max_new_tokens=self.max_new_tokens,
        )
        return sequences, None

    def generate_until(self, requests, **kwargs) -> list[str]:
        kind = _task_kind(requests[0].doc) if requests else "math"
        stats = GenerationStats(self.throughput_warmup)
        responses = []
        for i, request in tqdm(
            enumerate(requests),
            desc="Generating",
            total=len(requests),
            disable=self.rank != 0,
        ):
            context, gen_kwargs = request.args[0], request.args[1]
            prompt = self._prompt(context, kind)
            prompt_ids = torch.tensor(
                self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            )
            if self.rank == 0:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            sequences, output = self._generate(prompt_ids[None, :].to(self.device))
            num_new_tokens = int(sequences.shape[-1] - prompt_ids.numel())
            if self.rank == 0 and output is not None and output.total_time > 0:
                print(
                    f"[TIME STATS] Sample Drafting Time: {output.drafting_time:.4f}s / "
                    f"All Time: {output.total_time:.4f}s "
                    f"({output.drafting_time / output.total_time:.2%})"
                )
            stats.add(output, num_new_tokens)
            if self.rank == 0 and output is not None and output.total_time > 0:
                print(
                    f"[TIME STATS] Total Drafting Time: {stats.drafting_time:.4f}s / "
                    f"Total All Time: {stats.total_time:.4f}s "
                    f"({stats.drafting_time / stats.total_time:.2%})"
                )
            throughput = 0.0
            if self.rank == 0:
                end_event.record()
                torch.cuda.synchronize()
                throughput = num_new_tokens / (
                    start_event.elapsed_time(end_event) / 1000
                )
                if i >= self.throughput_warmup:
                    stats.throughputs.append(throughput)

            response = truncate_response(
                self.tokenizer.decode(sequences[0, len(prompt_ids) :]),
                gen_kwargs["until"],
                self.tokenizer.eos_token,
            )
            if kind == "code":
                if self.rank == 0:
                    print("=" * 20)
                    print("prompt: ", prompt)
                    print("generated: ", response)
                    print("=" * 20, end="\n\n")
            elif self.rank == 0:
                print("=" * 20)
                print("Prefix:", prompt)
                print("Generated:", response)
                print("=" * 20, end="\n\n")
            responses.append(response)

            if self.rank == 0:
                print(f"\nCompleted: {len(responses)}/{len(requests)}\n")
                stats.print_running(i, throughput)

        print(f"RANK {self.rank} completed!")
        return responses
