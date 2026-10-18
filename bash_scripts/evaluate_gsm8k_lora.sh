#!/bin/bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."
source setup_env.sh
set -u

# Qwen3-4B drafter fine-tuned for 3 epochs, with an eval loss of 1.02
MODEL_PATH=${MODEL_PATH:-/data/shared_data/hankun/outputs/tulu3_4096_seed_lora_block8_4b_dec2_rank512_lr2e-4_bsz32_20261010_001519}
QWEN_MODEL=${QWEN_MODEL:-Qwen/Qwen3-4B}
CKPT=${CKPT:-best}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
BLOCK_SIZE=${BLOCK_SIZE:-4}
CONF_SEG=${CONF_SEG:-true}
TREE_ATTN=${TREE_ATTN:-true}
DO_SAMPLE=${DO_SAMPLE:-false}
TEMPERATURE=${TEMPERATURE:-1.0}
GSM8K_NUM_SAMPLES=${GSM8K_NUM_SAMPLES:-null}  # null for all, or an integer

accelerate launch scripts/eval.py \
  task=gsm8k \
  task.limit="${GSM8K_NUM_SAMPLES}" \
  model_path="${MODEL_PATH}" \
  ckpt_file="${CKPT}-rank0.pt" \
  tokenizer.pretrained_model_name_or_path="${QWEN_MODEL}" \
  max_new_tokens="${MAX_NEW_TOKENS}" \
  generation_config.do_sample="${DO_SAMPLE}" \
  generation_config.temperature="${TEMPERATURE}" \
  'stopping_criteria=[eos,max_new_tokens]' \
  seed_decoding.block_size="${BLOCK_SIZE}" \
  seed_decoding.conf_seg="${CONF_SEG}" \
  seed_decoding.tree_attn="${TREE_ATTN}" \
  "$@"
