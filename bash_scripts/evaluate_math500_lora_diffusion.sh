#!/bin/bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."
source setup_env.sh
set -u

# 3 epochs

# MODEL_PATH=${MODEL_PATH:-/data/shared_data/hankun/outputs/tulu3_4096_seed_lora_diffusion_block16_8b_dec2_rank512_lr8e-5_bsz32_20261019_211322}
# MODEL_PATH=${MODEL_PATH:-/data/shared_data/hankun/outputs/tulu3_4096_seed_lora_diffusion_block7_8b_dec5_rank256_lr1e-4_bsz32_20261031_143130}

# 6 epochs
MODEL_PATH=${MODEL_PATH:-/data/shared_data/hankun/outputs/tulu3_4096_seed_lora_diffusion_block7_8b_dec5_rank256_lr1e-4_bsz32_20261114_172856}

if [[ "${MODEL_PATH}" == *"4b"* ]]; then
  QWEN_MODEL=${QWEN_MODEL:-Qwen/Qwen3-4B}
elif [[ "${MODEL_PATH}" == *"8b"* ]]; then
  QWEN_MODEL=${QWEN_MODEL:-Qwen/Qwen3-8B}
else
  echo "Error: MODEL_PATH must contain either '4b' or '8b' to determine the QWEN model."
  exit 1
fi
CKPT=${CKPT:-best}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-4096}
# Positions per drafter block (1 head + [MASK]s); null: the run's training block
# size. Drafts per round: BLOCK_SIZE, or BLOCK_SIZE - 1 with MERGE_DRAFT_VERIFY.
BLOCK_SIZE=${BLOCK_SIZE:-null}
DO_SAMPLE=${DO_SAMPLE:-true}
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
