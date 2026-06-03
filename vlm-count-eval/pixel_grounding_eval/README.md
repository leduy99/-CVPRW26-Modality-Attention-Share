# Pixel Grounding Evaluation

This repository contains tools for evaluating pixel grounding accuracy in vision-language models.

## Overview

The evaluation pipeline in `eval.ipynb` measures how well a model can ground (locate) objects or concepts mentioned in text within an image at the pixel level.

## Dataset

The evaluation uses the `patrickamadeus/vizwiz336-p14-grounding-1k` dataset, which contains:
- Images with 336x336 resolution
- Patch-level ground truth annotations (14x14 patches)
- Text descriptions referring to specific objects/regions in the images

## Evaluation Pipeline

The notebook provides a complete pipeline for:

1. **Loading the dataset** using the Hugging Face datasets library
2. **Visualizing samples** with matplotlib to inspect images and their ground truth annotations
3. **Running inference** on vision-language models to generate pixel-level predictions
4. **Computing metrics** such as:
   - Precision
   - Recall
   - F1 score
   - Attention-adapted-IoU (Intersection over Union for attention scores)