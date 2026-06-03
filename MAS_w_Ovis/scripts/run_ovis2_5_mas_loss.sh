#!/bin/bash

set -e

# Experiment name
EXPNAME="ovis2_5_mas_loss_experiment"
OVIS_CKPT_DIR="Ovis2.5-2B"
data_name="your_dataset_name"

CMDARG="--deepspeed scripts/zero_configs/zero1_cp.json \
  --stage 3 \
  --data_info_version your_datainfo \
  --data_name ${data_name} \
  --data_type conversation \
  --data_seed 5171 \
  --accepts_loss_kwargs True \
  --ovis_pretrained_path ${OVIS_CKPT_DIR} \
  --attn_implementation flash_attention_2 \
  --single_image_min_pixels 200704 \
  --single_image_max_pixels 3211264 \
  --min_frames 10 \
  --max_frames 10 \
  --train_modules all \
  --multimodal_max_length 4096 \
  --text_max_length 4096 \
  --use_mas_loss True \
  --mas_loss_weight 1.0 \
  --mas_tau 0.3 \
  --mas_apply_layers all \
  --output_dir ./checkpoints/$EXPNAME \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 3 \
  --eval_strategy no \
  --save_strategy steps \
  --save_steps 500 \
  --save_total_limit 5 \
  --learning_rate 2e-5 \
  --max_grad_norm 1.0 \
  --weight_decay 0.01 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type cosine \
  --logging_steps 10 \
  --tf32 True \
  --bf16 True \
  --dataloader_num_workers 4 \
  --dataloader_drop_last True \
  --dataloader_persistent_workers True \
  --gradient_checkpointing True \
  --report_to tensorboard \
  --run_name $EXPNAME"

echo "🚀 Training with MAS Loss"
echo "Parameters:"
echo "  - MAS Loss Weight: 1.0"
echo "  - MAS Tau (threshold): 0.3"
echo "  - Apply to layers: all"
echo ""
echo "Training arguments:"
echo "$CMDARG"
echo ""

# Run training
torchrun --nproc_per_node=1 ovis/train/train.py $CMDARG

