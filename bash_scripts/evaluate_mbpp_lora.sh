#!/bin/bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."
source setup_env.sh
set -u

# Match the checkpoint/tokenizer defaults in evaluate_math500_lora.sh.
MODEL_PATH=${MODEL_PATH:-/data/shared_data/hankun/outputs/tulu3_4096_seed_lora_block8_1.7b_dec2_rank512_lr6e-4_bsz32_20260926_204506}
QWEN_MODEL=${QWEN_MODEL:-Qwen/Qwen3-1.7B}
CKPT=${CKPT:-best}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
BLOCK_SIZE=${BLOCK_SIZE:-4}
CONF_SEG=${CONF_SEG:-true}
TREE_ATTN=${TREE_ATTN:-true}
DO_SAMPLE=${DO_SAMPLE:-false}
TEMPERATURE=${TEMPERATURE:-1.0}
MBPP_NUM_SAMPLES=${MBPP_NUM_SAMPLES:-null}  # null for all, or an integer

# Zero-shot non-thinking chat prompt; stopping criteria see only generated tokens.
accelerate launch scripts/eval.py \
  task=mbpp \
  task.limit="${MBPP_NUM_SAMPLES}" \
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
