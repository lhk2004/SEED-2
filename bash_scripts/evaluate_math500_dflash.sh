#!/bin/bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."
source setup_env.sh
set -u

DRAFT_MODEL=${DRAFT_MODEL:-z-lab/Qwen3-8B-DFlash-b16}
# DRAFT_MODEL=${DRAFT_MODEL:-deepseek-ai/dflash_qwen3_8b_block7}
if [[ "${DRAFT_MODEL,,}" == *"4b"* ]]; then
  QWEN_MODEL=${QWEN_MODEL:-Qwen/Qwen3-4B}
elif [[ "${DRAFT_MODEL,,}" == *"8b"* ]]; then
  QWEN_MODEL=${QWEN_MODEL:-Qwen/Qwen3-8B}
else
  echo "Error: DRAFT_MODEL must contain either '4B' or '8B' to determine the QWEN model."
  exit 1
fi
BLOCK_SIZE=${BLOCK_SIZE:-null}  # null: the drafter's training block size (16)
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-4096}
DO_SAMPLE=${DO_SAMPLE:-false}
TEMPERATURE=${TEMPERATURE:-1.0}
MATH500_NUM_SAMPLES=${MATH500_NUM_SAMPLES:-null}  # null for all, or an integer

accelerate launch scripts/eval.py \
  task=math500 \
  task.limit="${MATH500_NUM_SAMPLES}" \
  baseline=dflash \
  model_path="${QWEN_MODEL}" \
  baseline.draft_model_path="${DRAFT_MODEL}" \
  baseline.block_size="${BLOCK_SIZE}" \
  tokenizer.pretrained_model_name_or_path="${QWEN_MODEL}" \
  max_new_tokens="${MAX_NEW_TOKENS}" \
  generation_config.do_sample="${DO_SAMPLE}" \
  generation_config.temperature="${TEMPERATURE}" \
  +generation_config.top_k=0 \
  +generation_config.top_p=1.0 \
  '+generation_config.eos_token_id=${eos_token_id}' \
  '+generation_config.pad_token_id=${eos_token_id}' \
  "$@"
