
#!/bin/bash
# export CUDA_VISIBLE_DEVICES=0
set -e

#  Autoformer, FEDformer, LSTM, PatchTST, TimeNet, DLinear, Nonstationary_Transformer
MODELS=(
'LSTM'
# 'DLinear'
# 'FEDformer'
# Nonstationary_Transformer
)


DATASET_NAME="Temp"
TARGET='Low,High'

ROOT_PATH="$(dirname "$(dirname "$(realpath "$0")")")/dataset"
EMBED='timeF'

VENV_DIR="$(dirname "$(dirname "$(realpath "$0")")")/.venv"
LOG_DIR="./logs/optuna"

# Check if the virtual environment directory exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment not found at: $VENV_DIR"
    exit 1
fi
source "$VENV_DIR/bin/activate"

mkdir -p "$LOG_DIR"

# --------------------------- LOOP THROUGH MODELS --------------------------- #
for MODEL_NAME in "${MODELS[@]}"; do
    echo ""
    echo "Starting Optuna tuning for model: $MODEL_NAME"
    START_TIME=$(date +%s)

    LOG_FILE="${LOG_DIR}/optuna_${MODEL_NAME}_${DATASET_NAME}_$(date +%Y%m%d_%H%M%S).log"
    echo "Logging to: $LOG_FILE"

    # Run tuning
    "$VENV_DIR/bin/python" -u run_optun.py \
      --model $MODEL_NAME \
      --data_path "${DATASET_NAME}.csv" \
      --target $TARGET \
      --root_path $ROOT_PATH \
      --embed $EMBED \
      | tee "$LOG_FILE"

    END_TIME=$(date +%s)
    ELAPSED_TIME=$((END_TIME - START_TIME))

echo ""
echo "Optuna tuning completed in $ELAPSED_TIME seconds."
echo "Full log saved at: $LOG_FILE"
done    