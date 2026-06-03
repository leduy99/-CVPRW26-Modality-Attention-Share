# VLM Feature Probe

A comprehensive framework for comparing feature representations from different Vision-Language Models (VLMs). This project enables systematic evaluation of encoder vs projector features using lightweight object detection on Pascal VOC.

## Features

- **Multiple VLM Support**: Qwen-2.5-VL, Qwen-3-VL, Ovis-2.5, LLaVA-1.5, InternVL-2.5, InternVL-3, InternVL-3.5
- **Feature Extraction**: Both encoder and projector feature extraction
- **Lightweight Detection**: Center-based object detection head
- **Automatic Dataset Download**: Pascal VOC 2007 with automatic download
- **Performance Comparison**: AP@0.5 evaluation and comparison

## Installation

### 1. Create Conda Environment

```bash
conda create -n improve_vlm python=3.10 -y
conda activate improve_vlm
```

### 2. Install Dependencies

```bash
# Update pip
python -m pip install --upgrade pip

# Install PyTorch (CUDA 12.1)
pip install "torch==2.4.*" torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install "transformers>=4.56" pillow einops accelerate timm sentencepiece

# Optional: Flash attention for faster training
pip install "flash-attn==2.7.0.post2" --no-build-isolation || true
```

### 3. Optional: HuggingFace Login

For gated models:
```bash
huggingface-cli login
```

## Quick Start

### Basic Usage

```bash
cd vlm_feature_probe

# Compare encoder vs projector features on Qwen-2.5-VL
python train_compare.py --model qwen25 --epochs 10 --bs 4

# Try different models
python train_compare.py --model qwen3 --epochs 10 --bs 4
python train_compare.py --model ovis25 --epochs 10 --bs 2  # Note: smaller batch for 9B model
python train_compare.py --model llava15 --epochs 10 --bs 4  # LLaVA 1.5 (7B)
python train_compare.py --model internvl25 --epochs 10 --bs 4  # InternVL 2.5 (2B)
python train_compare.py --model internvl35 --epochs 10 --bs 4  # InternVL 3.5 (2B)

# Custom feature taps (faster experimentation)
python train_compare.py --model qwen25 --epochs 10 --taps llm  # Only LLM features
python train_compare.py --model llava15 --epochs 10 --taps projector,llm  # Projector + LLM only
python train_compare.py --model internvl35 --epochs 10 --taps encoder  # Only encoder features

# Select model size for fair comparison
python train_compare.py --model qwen3 --epochs 10  # Qwen3-VL 30B-A3B (3.3B activated, default)
python train_compare.py --model qwen3 --param 235b --epochs 1 --bs 1  # Qwen3-VL 235B-A22B (22B activated, huge!)
```

### Command Line Options

- `--model`: Model to use (`qwen25`, `qwen3`, `ovis25`, `llava15`, `internvl25`, `internvl3`, `internvl35`)
- `--epochs`: Number of training epochs (default: 10)
- `--bs`: Batch size (default: 4)
- `--lr`: Learning rate (default: 2e-3)
- `--hidden`: Hidden layer size in detection head (default: 128)
- `--head`: Detection head type (`center`, `detr`, `yolo`, `classification`)
- `--taps`: Feature taps to train (default: `all`)
  - `all`: Train all taps (encoder, projector, llm)
  - `encoder`: Only encoder features
  - `projector`: Only projector features
  - `llm`: Only LLM features
  - `projector,llm`: Custom combination (comma-separated)
- `--param`: Model parameter size (default: `auto`)
  - `auto`: Use model's default size
  - `2b`, `3b`, `7b`, `8b`: Select specific size (for models with variants)
  - Supported by: `qwen3` (3b=30B-A3B MoE / 235b=235B-A22B MoE)

## Project Structure

```
vlm_feature_probe/
├── adapters/           # VLM model adapters
│   ├── base.py        # Base adapter interface
│   ├── qwen25.py      # Qwen-2.5-VL adapter
│   ├── qwen3.py       # Qwen-3-VL adapter
│   ├── ovis25.py      # Ovis-2.5 adapter
│   ├── llava15.py     # LLaVA-1.5 adapter (7B)
│   ├── internvl25.py  # InternVL-2.5 adapter (2B)
│   └── internvl35.py  # InternVL-3.5 adapter (2B, Latest)
├── datasets/           # Dataset loaders
│   └── voc_center.py  # Pascal VOC center detection dataset
├── probes/            # Detection heads and utilities
│   └── center_head.py # Center detection head + training utils
├── train_compare.py   # Main training and comparison script
├── requirements.txt   # Dependencies
└── README.md         # This file
```

## Supported Models

### Default Models (2-4B - Fair Comparison)

| Model Key | Model ID | Params | Vision Encoder | Notes |
|-----------|----------|--------|----------------|-------|
| `ovis25` | AIDC-AI/Ovis2.5-2B | **2B** | SigLIP | Codebook architecture |
| `internvl25` | OpenGVLab/InternVL2_5-2B | **2B** | InternViT | Compact |
| `qwen25` | Qwen/Qwen2.5-VL-3B-Instruct | **3B** | Qwen2.5-VL | Efficient |
| `qwen3` | Qwen/Qwen3-VL-30B-A3B-Instruct | **3.3B** activated | Qwen3-VL | MoE (30.5B total) |

### Exception: CLIP Model

| Model Key | Model ID | Params | Vision Encoder | Notes |
|-----------|----------|--------|----------------|-------|
| `llava15` | llava-hf/llava-1.5-7b-hf | **7B** ⚠️ | **CLIP ViT-L** | Exception: Only CLIP |

### All Default Models (2-4B)

| Model Key | Model ID | Params | Vision Encoder | Notes |
|-----------|----------|--------|----------------|-------|
| `ovis25` | AIDC-AI/Ovis2.5-2B | **2B** | SigLIP | Codebook architecture |
| `internvl25` | OpenGVLab/InternVL2_5-2B | **2B** | InternViT | Standard VLM |
| `internvl35` | OpenGVLab/InternVL3_5-2B | **2B** | InternViT | Latest (Aug 2025), ViR ✨ |
| `qwen25` | Qwen/Qwen2.5-VL-3B-Instruct | **3B** | Custom | Standard VLM |
| `qwen3` | Qwen/Qwen3-VL-30B-A3B-Instruct | **3.3B** | Custom | MoE (30.5B total) |
| `llava15` | llava-hf/llava-1.5-7b-hf | **7B** | CLIP | Exception (only CLIP) |

### Model Selection Tips

**⚠️ Default = 2-4B models for fair comparison!**

- **Fair Comparison (Recommended)**: Use default 2-4B models
  - `ovis25` (2B), `internvl25` (2B), `qwen25` (3B), `qwen3` (3.3B activated MoE)
- **CLIP Features?** Use `llava15` (7B) - **Exception**: Only CLIP option available
- **Latest Tech?** Use `internvl35` (2B) - Cascade RL & ViR ✨
- **Fairest Comparison?** Use 3×2B models: `ovis25`, `internvl25`, `internvl35`

**Rules**:
- ✅ Default models: 2-4B activated params (always use smallest variant!)
- 🔶 LLaVA exception: 7B allowed (only CLIP encoder available)
- 📊 Best comparison: 3×2B models (ovis25, internvl25, internvl35)

### ⚠️ Important Note on Ovis Architecture

**Ovis uses a different architecture** than other VLMs:
- **Other models** (Qwen, LLaVA, InternVL): Continuous ViT features
- **Ovis**: Discrete codebook (65k vocabulary) + visual embedding table

Our adapter correctly handles this by computing: `F = softmax(logits) @ EmbeddingTable`

See `OVIS_ARCHITECTURE.md` for detailed explanation. This is critical for fair comparison!

## How It Works

### 1. Feature Extraction

The framework extracts features from two key points in VLM architectures:

- **Encoder Features**: Raw features from the vision encoder
- **Projector Features**: Features after the multimodal alignment layer

### 2. Detection Head

A lightweight center-based detection head is trained on each feature type:

- **Heatmap Branch**: Predicts object centers
- **Size Branch**: Predicts object width/height
- **Offset Branch**: Predicts sub-pixel center refinement

### 3. Evaluation

Performance is measured using AP@0.5 on Pascal VOC 2007 test set. The comparison reveals which feature representation is more suitable for object detection tasks.

## Expected Results

The framework will output:

```
=== FINAL COMPARISON - QWEN25 ===
Feature Source   AP@0.5    Improvement
----------------------------------------
Encoder          0.234     -
Projector        0.198     -0.036

✓ Encoder features are BETTER for detection (+0.036 AP@0.5)
```

## Troubleshooting

### Memory Issues

- Reduce batch size: `--bs 2` or `--bs 1`
- Use smaller models: Try Qwen-3-VL 4B instead of Qwen-2.5-VL 7B
- Enable gradient checkpointing in model loading

### Model Loading Issues

- Ensure you have sufficient disk space for model weights
- Check HuggingFace authentication for gated models
- Verify CUDA compatibility

### Performance Issues

- Install flash-attention for faster training
- Use mixed precision training (already enabled)
- Consider using multiple GPUs with `device_map="auto"`

## Extending the Framework

### Adding New Models

1. Create a new adapter in `adapters/` following the base interface
2. Implement `encode_grid()` and `project_tokens()` methods
3. Add model selection in `train_compare.py`

### Adding New Datasets

1. Create a new dataset class in `datasets/`
2. Implement `__getitem__()` method returning (image, boxes)
3. Update data loading in `train_compare.py`

### Adding New Detection Tasks

1. Create a new detection head in `probes/`
2. Implement training and evaluation functions
3. Update the main training loop

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{vlm_feature_probe,
  title={VLM Feature Probe: Comparing Encoder vs Projector Features},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/vlm-feature-probe}
}
```

## License

MIT License - see LICENSE file for details.


