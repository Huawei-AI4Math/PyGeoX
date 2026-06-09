#!/bin/bash

# ============================================================================
# MS-Swift SFT Training Script for Dense Models (Full Parameter Fine-Tuning)
# 
# This script performs Supervised Fine-Tuning (SFT) using ModelScope Swift
# framework with full parameter training (not LoRA).
# Uses dense reward dataset (reward > 2, with weights).
#
# Usage:
#   bash sft_dense.sh
#   or
#   ./sft_dense.sh
#
# Requirements:
#   - ms-swift installed and configured
#   - CUDA-compatible GPUs (full parameter training requires more GPU memory)
#   - Training dataset in JSONL format
# ============================================================================

# ==================== Configuration ====================
# GPU Configuration
export CUDA_VISIBLE_DEVICES=3

# Model and Data Paths (SFT training data in data/Qwen_32B_teacher_data)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL_PATH="YOUR_BASE_MODEL_PATH"  # Base model path
# For dense training, uses dataset with reward > 2 and sample weights
DATASET_PATH="${PROJECT_ROOT}/data/Qwen_32B_teacher_data/rs_dense.jsonl"
OUTPUT_DIR="${PROJECT_ROOT}/model_training/sft_training/SFT_dense"

# Training Hyperparameters (Full Parameter Training)
LEARNING_RATE=1e-5         # Learning rate (typically 5e-5 to 1e-5 for full parameter training)
WEIGHT_DECAY=0.01          # Weight decay for L2 regularization (0.01-0.1, helps prevent overfitting)
LR_SCHEDULER_TYPE="cosine" # Learning rate scheduler type: "linear", "cosine", "constant", "constant_with_warmup"
                           # - cosine: Smoothly decreases LR following cosine curve (recommended for SFT)
                           # - linear: Linear decay from max LR to 0
                           # - constant: Keeps LR constant throughout training
                           # - constant_with_warmup: Constant LR after warmup period
WARMUP_RATIO=0.1           # Warmup ratio (0.0-1.0): fraction of training steps for warmup
NUM_EPOCHS=2               # Number of training epochs
BATCH_SIZE=2               # Batch size per device (full parameter training uses more memory, adjust based on GPU)
GRADIENT_ACCUMULATION_STEPS=32  # Effective batch size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * num_gpus
MAX_LENGTH=18196            # Maximum sequence length
# Evaluation and Saving
EVAL_STEPS=50              # Evaluate every N steps
SAVE_STEPS=50             # Save checkpoint every N steps
SAVE_TOTAL_LIMIT=12        # Maximum number of checkpoints to keep
LOGGING_STEPS=5            # Log training metrics every N steps

# Other Settings
SPLIT_DATASET_RATIO=0.05    # Validation set ratio (5% for validation, 95% for training)

# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}

# Build swift sft command (Full Parameter Training - no LoRA parameters)
SWIFT_CMD="swift sft \
    --model ${MODEL_PATH} \
    --dataset ${DATASET_PATH} \
    --split_dataset_ratio ${SPLIT_DATASET_RATIO} \
    --output_dir ${OUTPUT_DIR} \
    --torch_dtype bfloat16 \
    --learning_rate ${LEARNING_RATE} \
    --weight_decay ${WEIGHT_DECAY} \
    --lr_scheduler_type ${LR_SCHEDULER_TYPE} \
    --warmup_ratio ${WARMUP_RATIO} \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --gradient_checkpointing true \
    --max_length ${MAX_LENGTH} \
    --eval_steps ${EVAL_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --save_total_limit ${SAVE_TOTAL_LIMIT} \
    --logging_steps ${LOGGING_STEPS} \
    --attn_impl flash_attn \
    --loss_scale dataset_weighted \
    --dataloader_num_workers 2"

# Run ms-swift SFT training
eval ${SWIFT_CMD}

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Training completed successfully!"
    echo "Check output directory: ${OUTPUT_DIR}"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "✗ Training failed! Check logs above for errors."
    echo "=========================================="
    exit 1
fi
