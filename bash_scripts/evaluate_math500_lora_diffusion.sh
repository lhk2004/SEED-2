#!/bin/bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."
source setup_env.sh
set -u

# A run directory of bash_scripts/run_train_seed_tulu3_lora_diffusion.sh
MODEL_PATH=${MODEL_PATH:}
# Refuse other runs (e.g. SEED-LoRA: use evaluate_math500_lora.sh), and take the
# tokenizer from the run's target model.
QWEN_MODEL=${QWEN_MODEL:-${RUN_TARGET_MODEL}}
CKPT=${CKPT:-best}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-4096}
BLOCK_SIZE=${BLOCK_SIZE:-null}  # draft tokens per round; null: the run's training block size
DO_SAMPLE=${DO_SAMPLE:-false}
TEMPERATURE=${TEMPERATURE:-1.0}
MATH500_NUM_SAMPLES=${MATH500_NUM_SAMPLES:-null}  # null for all, or an integer

# SEEDDiffusion.generate drafts whole blocks: no conf_seg / tree_attn options.
accelerate launch scripts/eval.py \
  task=math500 \
  task.limit="${MATH500_NUM_SAMPLES}" \
  model_path="${MODEL_PATH}" \
  ckpt_file="${CKPT}-rank0.pt" \
  tokenizer.pretrained_model_name_or_path="${QWEN_MODEL}" \
  max_new_tokens="${MAX_NEW_TOKENS}" \
  generation_config.do_sample="${DO_SAMPLE}" \
  generation_config.temperature="${TEMPERATURE}" \
  'stopping_criteria=[eos,max_new_tokens]' \
  seed_decoding.block_size="${BLOCK_SIZE}" \
  '~seed_decoding.conf_seg' \
  '~seed_decoding.tree_attn' \
  "$@"
