#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

# ------------------------ CONFIGURABLE PARAMETERS ------------------------ #
FILENAME="pre_ChiangSaen_spi_6"
TARGET="spi_6"
MODEL_NAME="GBRT"  # Có thể là: ARIMA, ETS, SVR, GBRT, etc.

# Dataset config
SEQ_LEN=192
LABEL_LEN=36
PRED_LEN=6
BATCH_SIZE=4
EMBED_TYPE="monthSine"

# Paths
ROOT_PATH="./dataset/DataSet_raw"
CHECKPOINT_DIR="./checkpoints"
LOG_DIR="logs"
# ------------------------------------------------------------------------ #

# ------------------------ SETUP ENVIRONMENT ------------------------ #
# Giả sử script đang được chạy từ thư mục /home/hung-tran-nam/SWAT_AI/MODELS
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"  # => /home/hung-tran-nam/SWAT_AI
VENV_DIR="$PROJECT_ROOT/.venv"          # => /home/hung-tran-nam/SWAT_AI/.venv

if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment not found at: $VENV_DIR"
    exit 1
fi
source "$VENV_DIR/bin/activate"

mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$LOG_DIR"

# ------------------------ RUN TRAINING ------------------------ #
START_TIME=$(date +%s)
LOG_FILE="${LOG_DIR}/train_${FILENAME}_${MODEL_NAME}_$(date +%Y%m%d_%H%M%S).log"

echo -e "\nLogging to: $LOG_FILE"
echo -e "Training Model: $MODEL_NAME on file: ${FILENAME}.csv\n"

python -u run_stats.py \
  --is_training 1 \
  --root_path $ROOT_PATH \
  --data_path "${FILENAME}.csv" \
  --model_id $FILENAME \
  --model $MODEL_NAME \
  --target $TARGET \
  --features MS \
  --seq_len $SEQ_LEN \
  --label_len $LABEL_LEN \
  --pred_len $PRED_LEN \
  --embed $EMBED_TYPE \
  --itr 1 \
  --des "Stats_Exp" \
  | tee "$LOG_FILE"

END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))

echo -e "\nTraining completed in $ELAPSED_TIME seconds."
echo "Full log saved at: $LOG_FILE"
