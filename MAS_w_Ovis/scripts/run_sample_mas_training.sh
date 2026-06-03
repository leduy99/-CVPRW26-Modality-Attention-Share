#!/bin/bash

set -e

# Experiment name
EXPNAME="sample_mas_training"
OVIS_CKPT_DIR="Ovis2.5-2B"
data_name="sample_dataset"

echo "🚀 Starting Sample MAS Loss Training"
echo "=================================="
echo "Experiment: $EXPNAME"
echo "Model: $OVIS_CKPT_DIR"
echo "Dataset: $data_name (10 samples)"
echo ""

# Check if dataset exists
if [ ! -f "sample_dataset/dataset.json" ]; then
    echo "❌ Dataset not found. Please run: python create_sample_dataset.py"
    exit 1
fi

# Check if model exists
if [ ! -d "$OVIS_CKPT_DIR" ]; then
    echo "❌ Model not found: $OVIS_CKPT_DIR"
    echo "Please make sure the model is downloaded."
    exit 1
fi

CMDARG="--deepspeed scripts/zero_configs/zero1_cp.json \
  --stage 3 \
  --data_info_version sample_datainfo \
  --data_name ${data_name} \
  --data_type conversation \
  --data_seed 42 \
  --accepts_loss_kwargs True \
  --ovis_pretrained_path ${OVIS_CKPT_DIR} \
  --attn_implementation flash_attention_2 \
  --single_image_min_pixels 200704 \
  --single_image_max_pixels 1000000 \
  --min_frames 1 \
  --max_frames 1 \
  --train_modules all \
  --multimodal_max_length 2048 \
  --text_max_length 2048 \
  --use_mas_loss True \
  --mas_loss_weight 0.5 \
  --mas_tau 0.3 \
  --mas_apply_layers all \
  --output_dir ./checkpoints/$EXPNAME \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 2 \
  --eval_strategy no \
  --save_strategy steps \
  --save_steps 5 \
  --save_total_limit 3 \
  --learning_rate 1e-5 \
  --max_grad_norm 1.0 \
  --weight_decay 0.01 \
  --warmup_ratio 0.2 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --tf32 True \
  --bf16 True \
  --dataloader_num_workers 2 \
  --dataloader_drop_last False \
  --dataloader_persistent_workers False \
  --gradient_checkpointing True \
  --report_to tensorboard \
  --run_name $EXPNAME"

echo "🔧 Training Parameters:"
echo "  - Batch size: 1"
echo "  - Gradient accumulation: 4"  
echo "  - Epochs: 2"
echo "  - Learning rate: 1e-5"
echo "  - MAS Loss weight: 0.5"
echo "  - MAS Tau: 0.3"
echo "  - Max tokens: 2048"
echo ""

echo "📝 Full command:"
echo "torchrun --nproc_per_node=1 ovis/train/train.py $CMDARG"
echo ""

read -p "Continue with training? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚀 Starting training..."
    torchrun --nproc_per_node=1 ovis/train/train.py $CMDARG
else
    echo "❌ Training cancelled."
    exit 1
fi

