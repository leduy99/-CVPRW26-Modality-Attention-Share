# Counting from Words, Not Worlds: Are VLMs Really Seeing What They Count?

Official implementation for **"Counting from Words, Not Worlds: Are VLMs Really Seeing What They Count?"**

This repository contains the benchmark generation, evaluation, probing, and training code used to study counting failures in Vision-Language Models (VLMs). Our goal is to diagnose whether VLMs genuinely use visual evidence when answering counting questions, or whether their apparent counting ability comes from language priors and shortcut reasoning.

> **Note:** This repository is organized as a modular codebase. Each sub-folder contains its own environment requirements and usage instructions.

---

## Overview

Vision-Language Models have made strong progress in multimodal reasoning, yet they still struggle with seemingly simple object counting. In this project, we investigate this failure mode through three complementary components:

1. **Controlled Counting Evaluation**
   We construct controlled counting scenarios to evaluate how robustly VLMs count objects under different visual and linguistic conditions.

2. **Internal Representation Probing**
   We probe intermediate representations of frozen VLMs to study where count-relevant visual signals appear or disappear across the vision encoder, projector, and language model.

3. **Modality Attention Share (MAS)**
   We introduce an attention-based regularization strategy to encourage better allocation of attention toward visual evidence during counting-related reasoning.

---

## Highlights

* **Counting-Tricks Benchmark**
  A controlled benchmark for diagnosing object-counting behavior in VLMs.
* **Attention and Spatial Signal Analysis**
  Tools for analyzing whether models attend to count-relevant visual regions.
* **Layer-wise Probing Framework**
  Lightweight detection heads trained on frozen VLM representations to inspect count-relevant information across model components.
* **MAS Training Pipeline**
  Fine-tuning code for applying Modality Attention Share regularization to the Ovis model.

---

## Dataset

The benchmark dataset is available on Hugging Face:

* [patrickamadeus/modality-imbalens-circles](https://huggingface.co/datasets/patrickamadeus/modality-imbalens-circles)

---

## Installation

Each module is self-contained and may require a different Python environment.

Please follow the installation instructions inside the corresponding sub-folder:

* [`vlm-count-eval/`](./vlm-count-eval/) for benchmark generation and VLM evaluation.
* [`models_probing/`](./models_probing/) for representation probing.
* [`MAS_w_Ovis/`](./MAS_w_Ovis/) for MAS training and Ovis fine-tuning.

A typical setup is:

```bash
git clone https://github.com/leduy99/project_cv801_code.git
cd project_cv801_code
```

Then navigate to the module you want to run and follow its local `README.md`.

---

## Usage

### 1. Counting Evaluation

Use the `vlm-count-eval/` module to generate controlled counting samples and evaluate VLMs on counting accuracy.

```bash
cd vlm-count-eval
# Follow the module-specific README for setup and evaluation commands.
```

### 2. Representation Probing

Use the `models_probing/` module to train and evaluate lightweight detection heads on frozen VLM representations.

```bash
cd models_probing
# Follow the module-specific README for probing instructions.
```

### 3. MAS Training

Use the `MAS_w_Ovis/` module to fine-tune Ovis with Modality Attention Share regularization.

```bash
cd MAS_w_Ovis
# Follow the module-specific README for training instructions.
```

---

## Method Summary

Our analysis is built around the following question:

> Are VLMs truly grounding their counting answers in visual evidence, or are they relying on language-stage shortcuts?

To answer this, we combine controlled evaluation, attention analysis, representation probing, and attention regularization. The resulting pipeline allows us to study not only whether a model answers correctly, but also where count-relevant visual information is encoded and whether it is used during reasoning.

---

## TODO

* [ ] Add paper link
* [ ] Add project page
* [ ] Add poster
* [ ] Add benchmark download instructions
* [ ] Add pretrained checkpoints
* [ ] Add full evaluation commands
* [ ] Add citation entry

---

## Citation

If you find this repository useful, please consider citing our work:

```bibtex
@inproceedings{le2026counting,
  title     = {Counting from Words, Not Worlds: Are VLMs Really Seeing What They Count?},
  author    = {Patrick Irawan and Anh Duy Le Dinh and Tuan Van Vo},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  year      = {2026}
}
```

---

## Acknowledgements

This repository was originally developed as part of the CV801 Advanced Computer Vision project.

We thank the instructors, teaching assistants, and collaborators who provided feedback and support throughout the project.

---

## Contact

For questions or collaboration, please contact:

**Anh Duy Le Dinh**

Email: [duy.le@mbzuai.ac.ae](mailto:duy.le@mbzuai.ac.ae)

Website: https://leduy99.github.io/
