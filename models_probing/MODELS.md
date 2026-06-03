# VLM Models Guide

This guide provides detailed information about all supported Vision-Language Models, including model specifications, architecture details, and usage examples.

---

## Supported Models Overview

We support **7 VLM models** across 4 families, all optimized for fair comparison with 2-4B activated parameters.

| Model Key | Model ID | Activated Params | Total Params | Vision Encoder | Status |
|-----------|----------|------------------|--------------|----------------|--------|
| `qwen25` | Qwen/Qwen2.5-VL-3B-Instruct | **3B** | 3B | Qwen-VL-ViT | ✅ Working |
| `qwen3` | Qwen/Qwen3-VL-4B-Instruct | **4B** | 4B | Qwen3-ViT | ✅ Working |
| `ovis25` | AIDC-AI/Ovis2.5-2B | **2B** | 2B | SigLIP-400M | ✅ Working |
| `internvl25` | OpenGVLab/InternVL2_5-2B | **2B** | 2B | InternViT-300M | ✅ Working |
| `internvl3` | OpenGVLab/InternVL3-2B | **2B** | 2B | InternViT-300M | ✅ Working |
| `internvl35` | OpenGVLab/InternVL3_5-2B | **2B** | 2B | InternViT-300M | ✅ Working |
| `llava15` | llava-hf/llava-1.5-7b-hf | **7B** | 7B | CLIP ViT-L | ✅ Working |

**→ Fair Comparison**: 6 models in 2-4B range + 1 CLIP-based model (7B)

---

## Memory Requirements (BF16)

| Model | Params | Memory (Model) | Memory (BS=8) | Recommended GPU |
|-------|--------|----------------|---------------|-----------------|
| ovis25 | 2B | ~4GB | ~8GB | 12GB+ |
| internvl25 | 2B | ~4GB | ~8GB | 12GB+ |
| internvl3 | 2B | ~4GB | ~8GB | 12GB+ |
| internvl35 | 2B | ~4GB | ~8GB | 12GB+ |
| qwen25 | 3B | ~6GB | ~10GB | 16GB+ |
| qwen3 | 4B | ~8GB | ~12GB | 16GB+ |
| llava15 | 7B | ~14GB | ~18GB | 24GB+ |

**✅ All models fit comfortably on consumer GPUs (12-24GB VRAM)**

---

## Quick Start Commands

### Fair Comparison (2-4B Models)
```bash
# All 2B models (Most Fair!)
python train_compare.py --model ovis25 --epochs 10      # 2B - SigLIP
python train_compare.py --model internvl25 --epochs 10  # 2B - InternViT
python train_compare.py --model internvl3 --epochs 10   # 2B - InternViT (V3)
python train_compare.py --model internvl35 --epochs 10  # 2B - InternViT (Latest!)

# Include 3-4B models
python train_compare.py --model qwen25 --epochs 10      # 3B
python train_compare.py --model qwen3 --epochs 10       # 4B (Regular, not MoE!)
```

### CLIP Baseline
```bash
# LLaVA 1.5 (only CLIP-based model)
python train_compare.py --model llava15 --epochs 10 --bs 4
```

---

## Detailed Model Specifications

### 1. Qwen 2.5-VL (3B)

**Architecture:**
- **Vision Encoder**: Qwen-VL-ViT (custom ViT)
- **Projector**: MLP
- **LLM**: Qwen2.5-3B
- **Total Parameters**: 3B

**Features:**
- Dynamic resolution support
- Efficient visual token compression
- Strong multilingual capabilities

**Usage:**
```bash
python train_compare.py --model qwen25 --epochs 10 --bs 4
```

**Expected Performance:**
- **Memory Usage**: ~10GB VRAM (batch size 8)
- **Training Speed**: ~2-3 minutes per epoch (Pascal VOC)
- **Feature Dimension**: Variable spatial grid → 3584 (LLM hidden dim)

---

### 2. Qwen 3-VL (4B)

**Architecture:**
- **Vision Encoder**: Qwen3-ViT (improved)
- **Projector**: MLP with enhanced alignment
- **LLM**: Qwen3-4B
- **Total Parameters**: 4B (Regular, **not MoE**)

**Features:**
- Latest Qwen3 architecture
- Improved visual grounding
- Better multimodal reasoning
- **Regular model** (not the 30B-A3B or 235B-A22B MoE variants)

**Usage:**
```bash
python train_compare.py --model qwen3 --epochs 10 --bs 4
```

**Expected Performance:**
- **Memory Usage**: ~12GB VRAM (batch size 8)
- **Training Speed**: ~2-3 minutes per epoch
- **Feature Dimension**: Variable spatial grid → 2560 (LLM hidden dim)

**Note**: We use the **4B regular model**, not the MoE variants (30B-A3B with 60GB memory or 235B-A22B with >100GB).

---

### 3. Ovis 2.5 (2B)

**Architecture:**
- **Vision Encoder**: SigLIP-400M (Google)
- **Projector**: Visual Tokenizer (VTE)
- **LLM**: Qwen2.5-1.5B
- **Total Parameters**: 2B

**Features:**
- SigLIP vision encoder (alternative to CLIP)
- Compact and efficient
- Strong visual grounding

**Usage:**
```bash
python train_compare.py --model ovis25 --epochs 10 --bs 4
```

**Expected Performance:**
- **Memory Usage**: ~8GB VRAM (batch size 8)
- **Training Speed**: ~2-3 minutes per epoch
- **Feature Map Size**: Variable grid (depends on input)
- **Feature Dimension**: 896 (vision) → 1536 (after VTE)

---

### 4. InternVL 2.5 (2B)

**Architecture:**
- **Vision Encoder**: InternViT-300M (448×448)
- **Projector**: MLP (mlp1)
- **LLM**: InternLM2-1.8B
- **Total Parameters**: 2B

**Features:**
- High-quality InternViT features
- 448×448 input resolution (higher than most)
- Efficient and fast

**Usage:**
```bash
python train_compare.py --model internvl25 --epochs 10 --bs 4
```

**Expected Performance:**
- **Memory Usage**: ~8GB VRAM (batch size 8)
- **Training Speed**: ~1-2 minutes per epoch
- **Feature Map Size**: 32×32 grid (typically)
- **Feature Dimension**: 1024 (vision) → 2048 (after projector)

---

### 5. InternVL 3.0 (2B)

**Architecture:**
- **Vision Encoder**: InternViT-300M-V2.5 (448×448)
- **Projector**: MLP (mlp1)
- **LLM**: Qwen2.5-1.5B
- **Total Parameters**: 2B

**Features:**
- Native Multimodal Pre-Training (unified language + vision)
- Variable Visual Position Encoding (V2PE) for longer contexts
- Mixed Preference Optimization (MPO) for better reasoning

**Usage:**
```bash
python train_compare.py --model internvl3 --epochs 10 --bs 4
```

**Expected Performance:**
- **Memory Usage**: ~8GB VRAM (batch size 8)
- **Training Speed**: ~1-2 minutes per epoch
- **Feature Map Size**: 32×32 grid
- **Feature Dimension**: 1024 (vision) → 1536 (after projector)

---

### 6. InternVL 3.5 (2B)

**Architecture:**
- **Vision Encoder**: InternViT-300M (448×448)
- **Projector**: MLP (mlp1)
- **LLM**: InternLM2-1.8B
- **Total Parameters**: 2B

**Features:**
- Most recent InternVL version
- Enhanced visual understanding
- Best InternVL performance for 2B size

**Usage:**
```bash
python train_compare.py --model internvl35 --epochs 10 --bs 4
```

**Expected Performance:**
- **Memory Usage**: ~8GB VRAM (batch size 8)
- **Training Speed**: ~1-2 minutes per epoch
- **Feature Map Size**: 32×32 grid
- **Feature Dimension**: 1024 (vision) → 2048 (after projector)

---

### 7. LLaVA 1.5 (7B)

**Architecture:**
- **Vision Encoder**: CLIP ViT-L/14 (336×336)
- **Projector**: 2-layer MLP
- **LLM**: Vicuna-7B (LLaMA-based)
- **Total Parameters**: 7B

**Features:**
- Well-established architecture
- **CLIP vision encoder** (only CLIP-based model in suite)
- Strong baseline performance
- Good balance between size and performance

**Usage:**
```bash
# Basic training
python train_compare.py --model llava15 --epochs 10 --bs 4

# Lower batch size for memory constraints
python train_compare.py --model llava15 --epochs 10 --bs 2
```

**Expected Performance:**
- **Memory Usage**: ~14-18GB VRAM (batch size 4)
- **Training Speed**: ~2-3 minutes per epoch
- **Feature Map Size**: 24×24 grid
- **Feature Dimension**: 1024 (vision) → 4096 (after projector)

**Special Considerations:**
- Uses prompt format: `USER: <image>\n{prompt}\nASSISTANT:`
- CLIP features pre-trained on large image-text datasets
- Allowed despite 7B size because it's the **only CLIP option** for research

---

## Model Comparison Table

| Model | Size | Vision Encoder | Resolution | Feature Grid | LLM Hidden | Best For |
|-------|------|----------------|------------|--------------|------------|----------|
| **qwen25** | 3B | Qwen-VL-ViT | Dynamic | Variable | 3584 | Dynamic resolution, multilingual |
| **qwen3** | 4B | Qwen3-ViT | Dynamic | Variable | 2560 | Latest Qwen, best reasoning |
| **ovis25** | 2B | SigLIP-400M | Variable | Variable | 1536 | SigLIP features, efficient |
| **internvl25** | 2B | InternViT-300M | 448×448 | 32×32 | 2048 | Fast, high resolution |
| **internvl3** | 2B | InternViT-V2.5 | 448×448 | 32×32 | 1536 | Native multimodal, V2PE |
| **internvl35** | 2B | InternViT-300M | 448×448 | 32×32 | 2048 | Latest InternVL, best 2B |
| **llava15** | 7B | CLIP ViT-L | 336×336 | 24×24 | 4096 | CLIP baseline, established |

---

## Training Tips

### Memory Optimization

If you encounter OOM (Out of Memory) errors:

1. **Reduce batch size**:
   ```bash
   python train_compare.py --model llava15 --bs 2
   python train_compare.py --model internvl35 --bs 2
   ```

2. **Use smaller models first**:
   - Start with 2B models (ovis25, internvl25, internvl3, internvl35)
   - Then try 3-4B models (qwen25, qwen3)
   - Finally test 7B model (llava15)

3. **Close other GPU processes**:
   ```bash
   nvidia-smi  # Check GPU usage
   kill <pid>  # Kill unnecessary processes
   ```

### Performance Optimization

For best detection performance:

1. **Increase training epochs**:
   ```bash
   python train_compare.py --model qwen25 --epochs 20
   ```

2. **Try different detection heads**:
   ```bash
   python train_compare.py --model internvl35 --head center  # Fast, simple
   python train_compare.py --model internvl35 --head yolo    # Balanced
   python train_compare.py --model internvl35 --head detr    # Best accuracy
   ```

3. **Adjust learning rate**:
   ```bash
   python train_compare.py --model qwen3 --lr 0.001  # Lower LR for stability
   python train_compare.py --model internvl35 --lr 0.005  # Higher LR for faster convergence
   ```

### Comparison Experiments

To compare all models side-by-side:

```bash
# Run all models with same settings
for model in qwen25 qwen3 ovis25 internvl25 internvl3 internvl35 llava15; do
    python train_compare.py --model $model --epochs 10 --bs 4 --head yolo
done

# Compare results from logs/
ls -la logs/
```

---

## Troubleshooting

### Common Issues

#### 1. Model Download Fails

**Problem**: HuggingFace connection timeout or authentication error.

**Solution**:
```bash
# Login to HuggingFace (if model is gated)
huggingface-cli login

# Or download manually
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct
huggingface-cli download OpenGVLab/InternVL2_5-2B
```

#### 2. CUDA Out of Memory

**Problem**: OOM error during training.

**Solution**:
- Reduce batch size: `--bs 2` or `--bs 1`
- Use smaller models (2B models)
- Close other GPU processes

#### 3. Import Errors

**Problem**: Module not found errors.

**Solution**:
```bash
pip install --upgrade transformers>=4.56
pip install pillow torchvision qwen-vl-utils
```

#### 4. Slow Training

**Problem**: Training is very slow.

**Solution**:
- Use smaller models (internvl35, ovis25)
- Reduce image resolution in adapter
- Check GPU utilization: `nvidia-smi`

#### 5. Poor Detection Performance

**Problem**: AP@0.5 is very low (<0.1).

**Solution**:
- Increase training epochs: `--epochs 20`
- Try different detection heads: `--head detr`
- Verify dataset is loaded properly
- Check feature extraction is working

---

## Advanced Usage

### Custom Feature Extraction

Extract features for your own analysis:

```python
from adapters.qwen25vl import Qwen25VLAdapter
from PIL import Image

# Initialize adapter
adapter = Qwen25VLAdapter()

# Load image
image = Image.open("example.jpg")

# Extract encoder features
encoder_features = adapter.encode_grid(image)
print(f"Encoder grid shape: {encoder_features['grid'].shape}")

# Extract projector features
projector_features = adapter.project_tokens(image)
print(f"Projector tokens shape: {projector_features['tokens'].shape}")

# Extract LLM features (after language model processing)
llm_features = adapter.extract_llm_features(image)
print(f"LLM tokens shape: {llm_features['tokens'].shape}")
```

### Custom Detection Head

Implement your own detection head:

```python
import torch.nn as nn

class MyDetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes=20):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 128, 3, padding=1)
        self.head = nn.Conv2d(128, num_classes + 4, 1)
    
    def forward(self, x):
        x = self.conv(x)
        return self.head(x)

# Use in training loop
head = MyDetectionHead(in_channels=2048)
```

---

## Next Steps

1. **Choose Your Model**:
   - For quick experiments: `internvl35` or `ovis25` (2B, fast)
   - For best performance: `qwen3` (4B, latest)
   - For CLIP baseline: `llava15` (7B, CLIP-based)

2. **Run Basic Training**:
   ```bash
   python train_compare.py --model internvl35 --epochs 10 --bs 4
   ```

3. **Compare Features**:
   - Check `logs/` directory for training progress
   - Compare AP@0.5 across encoder/projector/LLM features
   - Identify which feature level works best

4. **Experiment & Report**:
   - Try different detection heads
   - Adjust hyperparameters
   - Share insights with the community

Happy experimenting! 🚀

