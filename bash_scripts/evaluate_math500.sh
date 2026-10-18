#!/bin/bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."
source setup_env.sh
set -u

QWEN_MODEL=${QWEN_MODEL:-Qwen/Qwen3-4B}
MODEL_PATH=${MODEL_PATH:-${QWEN_MODEL}}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-4096}
DO_SAMPLE=${DO_SAMPLE:-true}
TEMPERATURE=${TEMPERATURE:-1.0}
MATH500_NUM_SAMPLES=${MATH500_NUM_SAMPLES:-null}  # null for all, or an integer

accelerate launch scripts/eval.py \
  task=math500 \
  task.limit="${MATH500_NUM_SAMPLES}" \
  model_path="${MODEL_PATH}" \
  torch_dtype=bfloat16 \
  tokenizer.pretrained_model_name_or_path="${QWEN_MODEL}" \
  max_new_tokens="${MAX_NEW_TOKENS}" \
  generation_config.do_sample="${DO_SAMPLE}" \
  generation_config.temperature="${TEMPERATURE}" \
  +generation_config.top_k=0 \
  +generation_config.top_p=1.0 \
  '+generation_config.eos_token_id=${eos_token_id}' \
  '+generation_config.pad_token_id=${eos_token_id}' \
  "$@"
