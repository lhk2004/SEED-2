# SEED-2

Speculative decoding with a LoRA drafter on top of a frozen target LLM. The
target (verifier) is an unmodified pretrained causal LM. The drafter reuses its
last few decoder layers, final norm, and LM head with LoRA adapters, and attends
to the target's KV cache for already-verified tokens. Only the adapters are
trained. An autoregressive (AR) baseline and the EAGLE-3 and DFlash speculative
decoding baselines are included for comparison.

## Layout

```
bash_scripts/   entry points (see below)
configs/        Hydra configs: train.yaml, eval.yaml, model/, task/
scripts/        train.py, eval.py, data/ (dataset generation + tokenization)
src/
  models/       seed.py (SEED, Drafter), lora.py, ar.py, common.py,
                baselines.py (EAGLE-3 and DFlash, evaluated like SEED)
  data.py       tokenized datasets and the padding collator
  training.py   Composer wrapper, metrics, best-checkpoint callback
  checkpoint.py loading trained runs (including runs saved by older versions)
  evaluation/   lm-eval-harness adapter, GSM8K / MATH-500 answer scoring
```

## Setup

```bash
conda env create -f environment.yml  # creates seed-env
source setup_env.sh                  # activates it and sets PYTHONPATH / HF_HOME
# EAGLE-3 baseline only (--no-deps: its setup.py pins torch 2.0 / transformers 4.46)
pip install --no-deps "git+https://github.com/SafeAILab/EAGLE.git@cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b"
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
| `evaluate_math500_{eagle3,dflash}.sh` | Evaluate the EAGLE-3 / DFlash baselines (public Qwen3-4B drafters) |

HumanEval and MBPP are generation-only (no generated code is executed).
