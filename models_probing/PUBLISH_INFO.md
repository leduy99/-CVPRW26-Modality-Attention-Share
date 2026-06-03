# Publication Package

This folder contains a clean, publishable version of the VLM Feature Probe codebase.

## What's Included

### Core Code
- `train_compare.py` - Main training and comparison script
- `verify_correctness.py` - Feature extraction verification script
- `download_voc_kaggle.py` - Pascal VOC dataset download utility

### Modules
- `adapters/` - VLM model adapters (Qwen, InternVL, LLaVA, Ovis)
- `probes/` - Detection heads (YOLO, Center, DETR, Classification)
- `datasets/` - Dataset loaders (Pascal VOC)
- `tools/` - Utility scripts

### Documentation
- `README.md` - Main documentation
- `MODELS.md` - Detailed model information
- `FAIRNESS_ANALYSIS.md` - Fairness analysis documentation
- `IMPLEMENTATION_GUIDE.md` - Implementation details
- `LICENSE` - License file
- `requirements.txt` - Python dependencies

## What's Excluded

The following have been excluded for a clean publication:
- Checkpoint files (`.pth`, `.pt`, `.ckpt`)
- Log files (`logs/` directory)
- Visualization outputs (`visualizations/` directory)
- Evaluation result files (`.txt` output files)
- Debug scripts (`debug_*.py`)
- Visualization scripts (`visualize_results.py`)
- Data directories (`vocdata/`)
- Cache files (`__pycache__/`)

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download dataset:
```bash
python download_voc_kaggle.py
```

3. Run training:
```bash
python train_compare.py --model qwen25 --epochs 10 --bs 4
```

See `README.md` for detailed usage instructions.

## File Structure

```
publish/
├── train_compare.py          # Main script
├── verify_correctness.py     # Verification
├── download_voc_kaggle.py    # Dataset download
├── adapters/                 # Model adapters
│   ├── base.py
│   ├── qwen25vl.py
│   ├── qwen3vl.py
│   ├── internvl25.py
│   ├── internvl3.py
│   ├── internvl35.py
│   ├── llava15.py
│   ├── llava.py
│   └── ovis25.py
├── probes/                   # Detection heads
│   ├── yolo_head.py
│   ├── center_head.py
│   ├── detr_head.py
│   ├── classification_head.py
│   ├── feature_adapter.py
│   ├── bottleneck.py
│   ├── neck.py
│   └── ema.py
├── datasets/                 # Dataset loaders
│   ├── voc_custom.py
│   └── voc_center.py
├── tools/                    # Utilities
│   └── inspect_internvl25.py
├── README.md                 # Main documentation
├── MODELS.md                 # Model details
├── FAIRNESS_ANALYSIS.md      # Fairness analysis
├── IMPLEMENTATION_GUIDE.md   # Implementation guide
├── LICENSE                   # License
├── requirements.txt          # Dependencies
└── .gitignore               # Git ignore rules
```

## License

See `LICENSE` file for details.

