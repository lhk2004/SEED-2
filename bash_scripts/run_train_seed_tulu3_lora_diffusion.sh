#!/bin/bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."
source setup_env.sh
set -u

PYTHON=${PYTHON:-python}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-Qwen/Qwen3-8B}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-4096}
DISTILL_DATA_ROOT=${DISTILL_DATA_ROOT:-/data/shared_data/hankun/datasets/tulu3_qwen3_8b_selfdistill_maxlen${MAX_SEQ_LEN}}
DISTILL_FILE_STEM=${DISTILL_FILE_STEM:-tulu3_qwen3_8b}
DISTILL_RAW_FILE=${DISTILL_RAW_FILE:-${DISTILL_DATA_ROOT}/${DISTILL_FILE_STEM}.jsonl}
DISTILL_TOKENIZER_NAME_OR_PATH=${DISTILL_TOKENIZER_NAME_OR_PATH:-Qwen/Qwen3-8B}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/shared_data/hankun/outputs}
BLOCK_SIZE=${BLOCK_SIZE:-16}  # tokens per block: the last verified token + BLOCK_SIZE - 1 [MASK]s
MASK_TOKEN_ID=${MASK_TOKEN_ID:-151669}  # [MASK] embedding row (DFlash's; unused by Qwen3's tokenizer)
N_DECODER_LAYERS=${N_DECODER_LAYERS:-2}
LORA_RANK=${LORA_RANK:-512}
LORA_ALPHA=${LORA_ALPHA:-1024}
LORA_DROPOUT=${LORA_DROPOUT:-0.0}
LR=${LR:-3e-4}
WARMUP_DURATION=${WARMUP_DURATION:-0.01dur}
ALPHA_F=${ALPHA_F:-0.5}
MAX_DURATION=${MAX_DURATION:-3ep}
BATCH_SIZE=${BATCH_SIZE:-32}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
NUM_WORKERS=${NUM_WORKERS:-0}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000ba}
EVAL_INTERVAL=${EVAL_INTERVAL:-1000ba}

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  NUM_VISIBLE_DEVICES=${NUM_VISIBLE_DEVICES:-1}
else
  IFS=',' read -r -a DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
  NUM_VISIBLE_DEVICES=${NUM_VISIBLE_DEVICES:-${#DEVICES[@]}}
fi
if (( BATCH_SIZE < NUM_VISIBLE_DEVICES * MICRO_BATCH_SIZE || BATCH_SIZE % (NUM_VISIBLE_DEVICES * MICRO_BATCH_SIZE) != 0 )); then
  echo "BATCH_SIZE must be divisible by NUM_VISIBLE_DEVICES * MICRO_BATCH_SIZE." >&2
  exit 1
fi
if (( MAX_SEQ_LEN % BLOCK_SIZE != 0 )); then
  echo "MAX_SEQ_LEN must be divisible by BLOCK_SIZE." >&2
  exit 1
fi

if [[ ! -d "${DISTILL_DATA_ROOT}/train_preprocessed" || ! -d "${DISTILL_DATA_ROOT}/eval_preprocessed" ]]; then
  if [[ ! -f "${DISTILL_RAW_FILE}" ]]; then
    echo "ERROR: Missing self-distilled Tulu3 JSONL: ${DISTILL_RAW_FILE}" >&2
    exit 1
  fi
  echo "[data] Preprocessing available Tulu3 responses in ${DISTILL_DATA_ROOT}."
  "${PYTHON}" scripts/data/preprocess_distilled_jsonl.py \
    --input-jsonl "${DISTILL_RAW_FILE}" \
    --data-root "${DISTILL_DATA_ROOT}" \
    --tokenizer-name-or-path "${DISTILL_TOKENIZER_NAME_OR_PATH}" \
    --max-length "${MAX_SEQ_LEN}" \
    --eval-ratio "${EVAL_RATIO:-0.01}" \
    --seed "${SPLIT_SEED:-42}" \
    --num-proc "${PREPROCESS_NUM_PROC:-4}"
else
  echo "[data] Reusing Tulu3 cache: ${DISTILL_DATA_ROOT}"
fi

# Prevent accidentally reusing a cache prepared for a different context length.
if [[ -f "${DISTILL_DATA_ROOT}/preprocessing_metadata.json" ]]; then
  "${PYTHON}" - "${DISTILL_DATA_ROOT}/preprocessing_metadata.json" "${MAX_SEQ_LEN}" <<'PY_CHECK'
import json
import sys
with open(sys.argv[1]) as handle:
    metadata = json.load(handle)
if metadata["max_length"] != int(sys.argv[2]):
    raise SystemExit("Cache max_length differs from MAX_SEQ_LEN; use a matching preprocessed dataset.")
PY_CHECK
fi

# set model tag if "1.7B" or "4B" is in the model name
if [[ "${MODEL_NAME_OR_PATH}" == *"1.7B"* ]]; then
  MODEL_TAG="1.7b"
elif [[ "${MODEL_NAME_OR_PATH}" == *"4B"* ]]; then
  MODEL_TAG="4b"
elif [[ "${MODEL_NAME_OR_PATH}" == *"8B"* ]]; then
  MODEL_TAG="8b"
else
  MODEL_TAG="unknown"
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_NAME=${RUN_NAME:-tulu3_${MAX_SEQ_LEN}_seed_lora_diffusion_block${BLOCK_SIZE}_${MODEL_TAG}_dec${N_DECODER_LAYERS}_rank${LORA_RANK}_lr${LR}_bsz${BATCH_SIZE}_${TIMESTAMP}}
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
mkdir -p "${RUN_DIR}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Only the drafter's LoRA parameters are trainable; the target model is frozen.
# The drafter fills each block of [MASK]s in one pass (src/models/seed_diffusion.py).
composer -n "${NUM_VISIBLE_DEVICES}" scripts/train.py \
  run_name="${RUN_NAME}" \
  pretrained_model_name_or_path="${MODEL_NAME_OR_PATH}" \
  train_dataset.dataset_path="${DISTILL_DATA_ROOT}/train_preprocessed" \
  eval_dataset.dataset_path="${DISTILL_DATA_ROOT}/eval_preprocessed" \
  max_seq_length="${MAX_SEQ_LEN}" \
  model=seed_diffusion \
  model.block_size="${BLOCK_SIZE}" \
  model.mask_token_id="${MASK_TOKEN_ID}" \
  model.num_draft_layers="${N_DECODER_LAYERS}" \
  model.lora_rank="${LORA_RANK}" \
  model.lora_alpha="${LORA_ALPHA}" \
  model.lora_dropout="${LORA_DROPOUT}" \
  model.train_on_context=false \
  training.global_batch_size="${BATCH_SIZE}" \
  training.grad_accum="$(( BATCH_SIZE / NUM_VISIBLE_DEVICES / MICRO_BATCH_SIZE ))" \
  composer.optimizer.lr="${LR}" \
  composer.trainer.precision=amp_bf16 \
  composer.trainer.max_duration="${MAX_DURATION}" \
  composer.trainer.eval_interval="${EVAL_INTERVAL}" \
  composer.trainer.save_interval="${SAVE_INTERVAL}" \
  composer.trainer.save_num_checkpoints_to_keep=1 \
  composer.lr_scheduler.t_warmup="${WARMUP_DURATION}" \
  composer.lr_scheduler.alpha_f="${ALPHA_F}" \
  composer.loggers.name="${RUN_NAME}" \
  train_dataloader.num_workers="${NUM_WORKERS}" \
  eval_dataloader.batch_size=1 \
  hydra.run.dir="${RUN_DIR}" \
  "$@"
