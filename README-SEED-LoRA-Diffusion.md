# SEED-LoRA-Diffusion

SEED-LoRA-Diffusion is a variant of SEED-LoRA in which the drafter proposes a
**whole block of draft tokens in a single forward pass** (one-step block
diffusion over `[MASK]` tokens). SEED-LoRA drafts one token at a time instead.

Everything else stays the same. The target model is frozen and verifies the
drafts. The drafter is the same set of LoRA adapters on copies of the target's
last `N_DECODER_LAYERS` decoder layers, final norm, and LM head, and it attends
to the target's KV cache. Only the drafter's **inputs** (`[MASK]` instead of
real tokens) and its **attention inside a block** (bidirectional instead of
causal) change. No new parameters are added.

|                               | SEED-LoRA                                 | SEED-LoRA-Diffusion                          |
| ----------------------------- | ----------------------------------------- | -------------------------------------------- |
| Drafter forward passes / round | one per draft token                      | **one**                                      |
| Drafter input in a block      | real tokens                               | last verified token, then `[MASK]`s          |
| Attention inside a block      | causal                                    | bidirectional                                |
| Draft length per round        | `BLOCK_SIZE`, or adaptive with `CONF_SEG` | `BLOCK_SIZE`                                 |
| Tree verification             | optional (`TREE_ATTN`)                    | none                                         |
| Trainable parameters          | LoRA adapters                             | the same LoRA adapters                       |
| Training loss                 | next-token cross-entropy                  | the same, with DFlash's position decay       |
| Training context              | 2L (target KV + drafter states)           | the same                                     |

## How a block works

A block has `B = BLOCK_SIZE` positions. The first is the **last verified
token**: the newest token whose target states are not yet in the KV cache (at
inference, the target's correction/bonus token from the previous round). The
other `B - 1` positions are `[MASK]`. As in SEED-LoRA, **every position predicts
the token after it**, so one block drafts `B` tokens:

```
position   s-1          s         s+1       ...   s+B-2
input      x[s-1]       [MASK]    [MASK]    ...   [MASK]
predicts   draft 1      draft 2   draft 3   ...   draft B
           (= x[s])     (= x[s+1])                (= x[s+B-1])
```

Every drafter position attends to

1. the target's KV cache for all tokens before `x[s-1]`, and
2. all `B` drafter positions of its own block (bidirectional attention).

`[MASK]` is the target's frozen embedding row **151669**. DFlash uses the same
row: Qwen3's embedding matrix has 151,936 rows, but its tokenizer only produces
ids up to 151,668. The row is not zero (norm ≈ 0.57 on Qwen3-8B vs. ≈ 1.7 for
real tokens), so `[MASK]` positions are a distinct, fixed input that the LoRA
adapters learn to read. To use another row, set `MASK_TOKEN_ID` (or
`model.mask_token_id`).

## Training

Training keeps SEED-LoRA's block-wise pattern. A training sequence of length
`L` is split into blocks of `B` tokens (positions `bB … bB+B-1`), and each
block is treated exactly like one inference round:

1. **Target (frozen, no grad):** one causal pass over the clean sequence gives
   the KV cache of every layer.
2. **Drafter inputs:** the first token of every block is kept (it plays the
   last verified token) and every other token is replaced by `[MASK]`.
3. **Drafter attention** (an `L × 2L` mask; keys are the `L` target states
   followed by the `L` drafter states). A query in block `b` sees the target
   states of blocks `< b` and every drafter state of block `b`:

   ```
                    keys: target states     keys: drafter states
   queries          blk0   blk1   blk2      blk0   blk1   blk2
   block 0           ·      ·      ·         ■      ·      ·
   block 1           ■      ·      ·         ·      ■      ·
   block 2           ■      ■      ·         ·      ·      ■
   ```

   SEED-LoRA uses the same mask except that its drafter block (■ on the right)
   is lower-triangular (causal).
4. **Loss:** next-token cross-entropy (negative log-likelihood) on response
   tokens, as in SEED-LoRA. The output at position `i` is scored
   against token `i + 1`, so each block is trained to produce its `B` drafts.
   As in DFlash (Eq. 4), draft token `k` of a block (`k = 1` for the
   prediction at the last verified token) is weighted by
   `exp(-(k - 1) / γ)` (`model.loss_decay_gamma`; `null` weights all
   positions equally), and the loss is the weighted mean `Σ w·CE / Σ w` over
   the loss tokens. A rejected early draft token makes all later ones
   worthless, so the early tokens count more. `eval/loss` (which picks the
   best checkpoint) is weighted too. `eval/NLL` is not, so it can be compared
   across values of `γ`.

Two details keep training identical to inference:

- **Padding at the end of a response.** The last block of a response is
  usually cut short by padding. Because attention inside a block is
  bidirectional, the real positions of that block would otherwise see fewer
  `[MASK]`s than at inference, where the drafter always fills a whole block. So
  in a block that starts with a real token, the padded positions act as
  ordinary `[MASK]` positions. They still never count toward the loss.
- **Prompt tokens** inside a block are masked like any other token. Only
  response tokens count toward the loss, so this affects just the one block per
  example that spans the prompt/response boundary.

The block size is fixed at training time (`model.block_size`, saved in the
run's `config.yaml`).

## Inference (`SEEDDiffusion.generate`)

Each round:

1. **Draft:** one drafter pass over `[last verified token, [MASK] × (B-1)]`
   gives `B` next-token distributions. Draft token `i` is sampled from
   distribution `i` (or is its argmax when decoding greedily). The draft ends
   early at its first EOS, or when `max_new_tokens` is reached. The drafter's
   own key/value states are then removed from the cache.
2. **Verify:** one target pass over `[last verified token, drafts]`, exactly as
   in SEED-LoRA (there are no tree candidates).
3. **Accept:** the longest accepted prefix is kept, plus one token from the
   target (the correction token, or a bonus token if every draft was accepted).
   That token becomes the next round's "last verified token".

So one round costs **one drafter pass over `B` tokens plus one target pass over
`B + 1` tokens**. SEED-LoRA makes one drafter pass per draft token.

**The output distribution is exact.**
- *Greedy* (`DO_SAMPLE=false`): the output equals the target's own greedy
  decoding.
- *Sampling* (`DO_SAMPLE=true`): the speculative sampling rule of Leviathan et
  al. (2023), as in SEED-LoRA, which samples from the target at
  `TEMPERATURE`. Each draft token is drawn from its own position's distribution
  and does not depend on the earlier drafts, and the acceptance test uses that
  same probability. This is a valid proposal for the algorithm, so the output
  still follows the target distribution exactly.

**Block size at evaluation.** `BLOCK_SIZE` defaults to the run's training
block size. It can be set to any value, but a block larger than the training
block contains more `[MASK]`s than the drafter ever saw in training. Smaller
blocks are the safer direction.

## Running it

All commands run from any directory and are configured with environment
variables. Extra Hydra overrides can be appended at the end, as with the other
scripts.

### Train

```bash
# Defaults: Qwen/Qwen3-8B, MAX_SEQ_LEN=4096, BLOCK_SIZE=8, N_DECODER_LAYERS=2,
# LORA_RANK=512, LR=5e-5, BATCH_SIZE=32, 3 epochs.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash bash_scripts/run_train_seed_tulu3_lora_diffusion.sh

# For example, a different model, block size, and learning rate:
CUDA_VISIBLE_DEVICES=0,1,2,3 MODEL_NAME_OR_PATH=Qwen/Qwen3-4B BLOCK_SIZE=16 LR=2e-4 \
  bash bash_scripts/run_train_seed_tulu3_lora_diffusion.sh

# Uniform loss weights (no decay):
LOSS_DECAY_GAMMA=null bash bash_scripts/run_train_seed_tulu3_lora_diffusion.sh
```

The script is `run_train_seed_tulu3_lora.sh` with `model=seed_diffusion`. It
takes the same environment variables (`MODEL_NAME_OR_PATH`, `MAX_SEQ_LEN`,
`BLOCK_SIZE`, `N_DECODER_LAYERS`, `LORA_RANK`, `LORA_ALPHA`, `LORA_DROPOUT`,
`LR`, `BATCH_SIZE`, `MICRO_BATCH_SIZE`, `MAX_DURATION`, …), uses the same
self-distilled Tulu 3 data, and adds two more:

| Variable           | Default                  | Meaning                                              |
| ------------------ | ------------------------ | ---------------------------------------------------- |
| `MASK_TOKEN_ID`    | `151669`                 | Embedding row used as the `[MASK]` input             |
| `LOSS_DECAY_GAMMA` | DFlash's γ for the block | Loss decay `exp(-(k-1)/γ)`; `null` = uniform weights |

DFlash (Appendix A.1) uses γ = 7, 5, 4 for its blocks of 16, 10, 8 tokens,
which draft 15, 9, 7 tokens. A block here drafts `BLOCK_SIZE` tokens, so the
script uses γ = 4 for `BLOCK_SIZE` 7 or 8, 5 for 9 or 10, and 7 for 15 or 16.
For any other block size it stops and asks for `LOSS_DECAY_GAMMA`.

Runs are saved to
`${OUTPUT_ROOT}/tulu3_<len>_seed_lora_diffusion_block<B>_<model>_dec<N>_rank<r>_gamma<γ>_lr<lr>_bsz<bsz>_<timestamp>`.
The LoRA weights always start fresh.

### Evaluate on MATH-500

```bash
MODEL_PATH=/data/shared_data/hankun/outputs/<diffusion run> \
  bash bash_scripts/evaluate_math500_lora_diff.sh

# Smaller blocks, greedy decoding, first 50 problems:
MODEL_PATH=<run> BLOCK_SIZE=4 DO_SAMPLE=false MATH500_NUM_SAMPLES=50 \
  bash bash_scripts/evaluate_math500_lora_diff.sh
```

| Variable              | Default                     | Meaning                                    |
| --------------------- | --------------------------- | ------------------------------------------ |
| `MODEL_PATH`          | (required)                  | A SEED-LoRA-Diffusion run directory        |
| `BLOCK_SIZE`          | `null` (= training block)   | Draft tokens per round                     |
| `CKPT`                | `best`                      | Checkpoint: `<CKPT>-rank0.pt`              |
| `MAX_NEW_TOKENS`      | `4096`                      | Generation length limit                    |
| `DO_SAMPLE`           | `true`                      | Speculative sampling (`false`: greedy)     |
| `TEMPERATURE`         | `1.0`                       | Sampling temperature                       |
| `MATH500_NUM_SAMPLES` | `null` (all 500)            | Number of problems                         |
| `QWEN_MODEL`          | the run's target model      | Tokenizer                                  |

`CONF_SEG` and `TREE_ATTN` do not exist here. The script reads the target
model from the run's `config.yaml`, so the run name does not need to contain
`4b`/`8b`. It also refuses runs that are not SEED-LoRA-Diffusion.

The printed statistics are the same as for SEED-LoRA. *Acceptance rate* is
accepted ÷ drafted tokens. *Average accepted length* is accepted draft tokens
per round, excluding the target's correction/bonus token (so tokens per
round = average accepted length + 1). *Per-position acceptance* gives the rate
for each draft position 1…B. *Thput* is tokens per second.

To evaluate another benchmark, copy the script and change `task=math500` /
`task.limit` the same way the other `evaluate_*_lora.sh` scripts differ.

## What changed in the code

New files:

| File | Contents |
| ---- | -------- |
| `src/models/seed_diffusion.py` | `SEEDDiffusion` (a subclass of `SEED`): the training mask, `[MASK]` inputs, padding handling, and `generate` |
| `configs/model/seed_diffusion.yaml` | Model config (`model=seed_diffusion`) |
| `bash_scripts/run_train_seed_tulu3_lora_diffusion.sh` | Training script |
| `bash_scripts/evaluate_math500_lora_diff.sh` | MATH-500 evaluation script |

Small changes to existing files (SEED-LoRA behaves exactly as before):

- `src/models/seed.py`: `SEED.forward` now asks two methods for the drafter's
  training inputs: `_drafter_input_ids` (which tokens the drafter embeds) and
  `_drafter_token_mask` (which positions take part in attention). In `SEED`
  they return `input_ids` and `attention_mask` unchanged, which gives the same
  computation as before. `SEEDDiffusion` overrides both.
- `src/models/__init__.py`: also exports `SEEDDiffusion`.
- `configs/eval.yaml`: a comment noting that `SEEDDiffusion.generate` only
  takes `block_size`.

Evaluation reuses `scripts/eval.py` and the lm-eval adapter unchanged.
`SEEDDiffusion` is a `SEED`, so its LoRA adapters are merged for inference and
it is called like SEED. Its checkpoints load with the existing
`load_model_from_run`. Running `evaluate_math500_lora.sh` on a diffusion run
fails with `unexpected keyword argument 'conf_seg'`; use
`evaluate_math500_lora_diff.sh` instead.

## Checks that were run

On Qwen3-1.7B in float32, with random non-zero LoRA weights:

- **No information leakage in training:** changing token `j` only changes
  training logits in blocks whose first token is at or after `j`.
- **Training matches inference:** the training logits of every block equal
  those of an inference drafting pass over `[last verified token, [MASK]…]` on
  a fresh target cache (max difference 9e-5). Right-padding does not change the
  logits of real positions (difference 0).
- **Gradients:** only the drafter's LoRA tensors are trainable, and all of them
  receive gradients.
- **Greedy decoding equals the target's greedy decoding.** This holds for the
  random drafter (block sizes 1, 3, 8, 12) and for oracle drafters that are
  always right or wrong on 15% or 50% of tokens (block sizes 4, 8, 16). With an
  always-right oracle, every draft is accepted. `max_new_tokens` limits that
  fall inside a block are also handled correctly.
- **Sampling is exact:** for a toy two-token target with independent
  per-position proposals, the empirical joint distribution of 2M samples
  matched the target to within 3e-4 (Monte Carlo noise).

End to end, on Qwen3-1.7B:

- **Both new scripts run:** training → checkpoint → MATH-500 evaluation with
  sampling, with greedy decoding, and with a `BLOCK_SIZE` override. The
  existing `run_train_seed_tulu3_lora.sh` still trains as before.
- **A short training run learns every block position.** 300 steps (batch 16,
  LoRA rank 64, LR 3e-4) brought the eval loss from 14.7 to 5.98. Per-position
  eval NLL (position 1 is the last verified token, 2–8 are `[MASK]`s):

  | Block position      | 1     | 2     | 3     | 4     | 5     | 6     | 7     | 8     |
  | ------------------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
  | Untrained LoRA      | 14.68 | 15.60 | 15.52 | 15.56 | 15.53 | 15.68 | 15.49 | 15.63 |
  | After 300 steps     | 4.07  | 5.12  | 5.28  | 5.34  | 5.35  | 5.41  | 5.37  | 5.45  |

  On 5 MATH-500 problems (greedy, block size 8), this drafter's acceptance was
  37.7% at position 1, 2.9% at position 2, and 0% at positions 3–8. That run
  is far shorter than the real recipe (3 epochs at rank 512), so it only shows
  that the pipeline works and that later positions are harder. It does not
  predict the acceptance of a fully trained drafter.
