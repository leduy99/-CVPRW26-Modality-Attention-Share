#!/bin/bash

# FSC-147 Counting Training Script with MAS Loss for Ovis
# Make sure to run prepare_fsc147_training.py first!

export CUDA_VISIBLE_DEVICES=0
export WANDB_DISABLED=true

# Model and training parameters
MODEL_NAME="Ovis2.5-2B"
OUTPUT_DIR="./checkpoints/fsc147_counting_mas"
DATA_INFO="ovis/train/dataset/fsc147_datainfo.json"

# Training hyperparameters (reduced for memory)
BATCH_SIZE=2  # Minimum batch size
GRADIENT_ACCUMULATION_STEPS=32
LEARNING_RATE=1e-5
NUM_EPOCHS=10
MAX_LENGTH=1024

# MAS Loss parameters
MAS_LOSS_WEIGHT=0.1
MAS_TAU=0.4
MAS_APPLY_LAYERS="20,21,22,23"  # Last 8 layers for Ovis2.5-2B

# Create output directory
mkdir -p $OUTPUT_DIR

# Training command with MAS Loss
python -m ovis.train.train \
    --ovis_pretrained_path $MODEL_NAME \
    --data_info_version fsc147_datainfo \
    --data_name fsc147_counting \
    --data_type conversation \
    --train_modules llm \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate $LEARNING_RATE \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --save_steps 500 \
    --save_total_limit 3 \
    --eval_strategy no \
    --dataloader_num_workers 8 \
    --dataloader_pin_memory True \
    --multimodal_max_length $MAX_LENGTH \
    --bf16 True \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to none \
    --remove_unused_columns False \
    --ddp_find_unused_parameters False \
    --use_mas_loss true \
    --mas_loss_weight $MAS_LOSS_WEIGHT \
    --mas_tau $MAS_TAU \
    --mas_apply_layers $MAS_APPLY_LAYERS \
    --mas_only false \
    --freeze_vision_encoder true \
    --freeze_text_encoder true

echo "Training with MAS Loss completed! Checkpoints saved to: $OUTPUT_DIR"
