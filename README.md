# SEED

Speculative decoding with a LoRA drafter on top of a frozen target LLM. The
target (verifier) is an unmodified pretrained causal LM. The drafter reuses its
last few decoder layers, final norm, and LM head with LoRA adapters, and attends
to the target's KV cache for already-verified tokens. Only the adapters are
trained. An autoregressive (AR) baseline is included for comparison.

## Layout

```
bash_scripts/   entry points (see below)
configs/        Hydra configs: train.yaml, eval.yaml, model/, task/
scripts/        train.py, eval.py, data/ (dataset generation + tokenization)
src/
  models/       seed.py (SEED, Drafter), lora.py, ar.py, common.py
  data.py       tokenized datasets and the padding collator
  training.py   Composer wrapper, metrics, best-checkpoint callback
  checkpoint.py loading trained runs (including runs saved by older versions)
  evaluation/   lm-eval-harness adapter, GSM8K / MATH-500 answer scoring
```

## Setup

```bash
conda env create -f environment.yml  # creates seed-env
source setup_env.sh                  # activates it and sets PYTHONPATH / HF_HOME
```

## Scripts

All scripts can be run from any directory and are configured with environment
variables (see the top of each script); extra Hydra overrides can be appended.

| Script | Purpose |
| --- | --- |
| `generate_tulu3_selfdistill_advanced.sh` | Self-distill Tulu 3 responses with vLLM |
| `run_train_seed_tulu3_lora.sh` | Tokenize the distilled data (if needed) and train the SEED drafter |
| `evaluate_{gsm8k,math500,humaneval,mbpp}_lora.sh` | Evaluate a SEED run (`MODEL_PATH=<run dir>`) |
| `evaluate_{gsm8k,math500,humaneval,mbpp}.sh` | Evaluate the untuned model with plain autoregressive decoding |

HumanEval and MBPP are generation-only (no generated code is executed).
