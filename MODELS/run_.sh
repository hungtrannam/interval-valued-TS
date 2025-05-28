#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

# ------------------------ CONFIGURABLE PARAMETERS ------------------------ #
FILENAME="pre_LuangPhrabang_spi_6"
TARGET="spi_6"
MODEL_NAME="FEDformer"

# Input sequence configuration
SEQ_LEN=192
LABEL_LEN=128
PRED_LEN=6

# Model architecture
E_LAYERS=6
D_LAYERS=4
D_MODEL=128
D_FF=1024
FACTOR=3
MOV_AVG=3

# Training settings
BATCH_SIZE=8
LEARNING_RATE=0.0001
DROPOUT=0.1
TRAIN_EPOCHS=100
PATIENCE=8

# Dataset paths
ROOT_PATH="/home/hung-tran-nam/SWAT_AIv2v/dataset/DataSet"
CHECKPOINT_DIR="./checkpoints"
USE_DTW=1
EMBED_TYPE="timeF"
# ---------------------------------------------------------------------------- #

# ------------------------ SETUP ENVIRONMENT ------------------------ #
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_ROOT/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment not found at: $VENV_DIR"
    exit 1
fi
source "$VENV_DIR/bin/activate"

mkdir -p "$CHECKPOINT_DIR"

# ------------------------ START TRAINING ------------------------ #
START_TIME=$(date +%s)
export CUDA_VISIBLE_DEVICES=1

LOG_FILE="logs/train_${FILENAME}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs
echo -e "\nLogging to: $LOG_FILE"
echo -e "Training Model: $MODEL_NAME on $FILENAME.csv\n"

python -u run.py \
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
  --e_layers $E_LAYERS \
  --d_layers $D_LAYERS \
  --d_model $D_MODEL \
  --d_ff $D_FF \
  --factor $FACTOR \
  --enc_in 5 \
  --dec_in 5 \
  --c_out 1 \
  --moving_avg $MOV_AVG \
  --batch_size $BATCH_SIZE \
  --dropout $DROPOUT \
  --des "Exp" \
  --embed $EMBED_TYPE \
  --use_dtw $USE_DTW \
  --lradj type3 \
  --learning_rate $LEARNING_RATE \
  --train_epochs $TRAIN_EPOCHS \
  --itr 1 \
  --patience $PATIENCE \
  | tee "$LOG_FILE"

END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))

echo -e "\nTraining completed in $ELAPSED_TIME seconds."
echo "Full log saved at: $LOG_FILE"
