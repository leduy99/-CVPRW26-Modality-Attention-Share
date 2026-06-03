# Ovis
<div align="center">
  <img src=docs/ovis_logo.png width="30%"/>
</div>
<br>

<p align="center">
  <a href="https://arxiv.org/abs/2508.11737"><img src="https://img.shields.io/badge/📖_Technical_Report-Ovis2.5-b31b1b.svg" alt="technical report"></a>
  <a href="https://huggingface.co/spaces/AIDC-AI/Ovis2.5-9B"><img src="https://img.shields.io/badge/🎨_HF_Spaces-AIDC--AI/Ovis2.5--9B-lightblack" alt="demo"></a>
  <a href="https://huggingface.co/collections/AIDC-AI/ovis25-689ec1474633b2aab8809335"><img src="https://img.shields.io/badge/🤗_Models-AIDC--AI/Ovis2.5-yellow" alt="models"></a>
</p>

## Introduction

Ovis (Open VISion) is a novel Multimodal Large Language Model (MLLM) architecture, designed to structurally align visual and textual embeddings.

<div style="text-align: center;">
  <img style="max-width: 100%;" src="docs/Ovis25_arch.png" alt="Ovis Illustration"/>
</div>

## Contents
- [Ovis: Structural Embedding Alignment for Multimodal Large Language Model](#ovis-structural-embedding-alignment-for-multimodal-large-language-model)
  - [Release](#release)
  - [Contents](#contents)
  - [Model](#model)
  - [Performance](#performance)
  - [Install](#install)
  - [Inference](#inference)
  - [Model Fine-tuning](#model-fine-tuning)
  - [Citation](#citation)
  - [Team](#team)
  - [🔥 We are hiring!](#we-are-hiring)
  - [License](#license)
  - [Disclaimer](#disclaimer)

## Model
Ovis can be instantiated with popular LLMs. We provide the following Ovis MLLMs:

| Ovis MLLMs |           ViT           |          LLM          |                      Model Weights                      |                           Demo                           |
|:-----------|:-----------------------:|:---------------------:|:-------------------------------------------------------:|:--------------------------------------------------------:|
| Ovis2.5-2B   | siglip2-so400m-patch16-512 | Qwen3-1.7B | [Huggingface](https://huggingface.co/AIDC-AI/Ovis2.5-2B)  | [Space](https://huggingface.co/spaces/AIDC-AI/Ovis2.5-2B) |
| Ovis2.5-9B   | siglip2-so400m-patch16-512  |  Qwen3-8B  | [Huggingface](https://huggingface.co/AIDC-AI/Ovis2.5-9B)  | [Space](https://huggingface.co/spaces/AIDC-AI/Ovis2.5-9B) |


## Performance
Ovis2.5 demonstrates strong results on general multimodal benchmarks, complex chart analysis, and reasoning tasks, achieving leading performance among open-source models under 40B parameters.


![performance-Ovis2_5](docs/performance/Ovis2_5_performance.png)


![OC-Ovis2_5](docs/performance/Ovis2_5_OC.png)

![REASON-Ovis2_5](docs/performance/Ovis2_5_reason.png)

## Install
Ovis has been tested with Python 3.10, Torch 2.4.0, Transformers 4.51.3, and DeepSpeed 0.15.4. For a comprehensive list of package dependencies, please consult the `requirements.txt` file.
```bash
git clone git@github.com:AIDC-AI/Ovis.git
conda create -n ovis python=3.10 -y
conda activate ovis
cd Ovis
pip install -r requirements.txt
pip install -e .
```

For `vLLM`:

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
VLLM_USE_PRECOMPILED=1 uv pip install .
```

## Inference

We provide inference examples using both **transformers** and **vLLM**.

### transformers

In `ovis/serve` we provide three example files:

* **`ovis/serve/infer_think_demo.py`**  
  Demonstrates how to enable the model’s *reflective reasoning* via  
  `enable_thinking` and to control the reasoning phase length with `thinking_budget`.

* **`ovis/serve/infer_basic_demo.py`**  
  Provides inference examples for single-image, multi-image, video, and pure-text inputs.

* **`ovis/serve/web_ui.py`**
  Provides a **Gradio-based Web UI** demo.
  Example run:

  ```bash
  python ovis/serve/web_ui.py --model-path AIDC-AI/Ovis2.5-9B --port 8001
  ```

### vLLM

Start the vLLM server:

```bash
vllm serve AIDC-AI/Ovis2.5-9B \
     --trust-remote-code \
     --port 8000
```

Call the model using the **OpenAI Python SDK**:

```python
from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

chat_response = client.chat.completions.create(
    model="AIDC-AI/Ovis2.5-9B",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://cdn-uploads.huggingface.co/production/uploads/637aebed7ce76c3b834cea37/kh-1dhZRAduP-P4SkIhXr.png"
                    },
                },
                {"type": "text", "text": "Recognize the table content"},
            ],
        },
    ],    
    extra_body={
        "chat_template_kwargs": {
            "enable_thinking": True,
        },
        "mm_processor_kwargs": {
            "images_kwargs": {
                "min_pixels": 1048576,   # 1024 * 1024
                "max_pixels": 3211264    # 1792 * 1792
            }
        }
    }
)

print("Chat response:\n", chat_response.choices[0].message.content)
```

#### Explanation of `extra_body` parameters:

* **`chat_template_kwargs.enable_thinking`**
  Enables *thinking mode* (reflective reasoning).

* **`mm_processor_kwargs.images_kwargs.min_pixels / max_pixels`**
  Controls the resolution range of input images (in total pixel count), balancing accuracy and GPU memory usage.


## Model Fine-tuning

Ovis can be fine-tuned using either the provided training code in this repository or via [ms-swift](https://github.com/modelscope/ms-swift).


### 1. Fine-tuning with in-repo code

#### Data Format

The training dataset is stored as a **JSON list**, where each element corresponds to a single sample.
Example dataset JSON:

```jsonc
[
    {
        "id": 1354,
        "image": "1354.png",
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nIn the figure, the vertices of quadrilateral ABCD intersect square EFGH and divide its sides into segments with measures that have a ratio of 1:2. Find the ratio between the areas of ABCD and EFGH."
            },
            {
                "from": "gpt",
                "value": "5:9"
            }
        ]
    }
]
```

#### Dataset Information

Datasets are referenced via **datainfo JSON files**, e.g. `ovis/train/dataset/ovis2_5_sft_datainfo.json`:

```json
{
    "geometry3k_local": {
        "meta_file": "path/to/geometry3k_local.json",
        "storage_type": "hybrid",
        "data_format": "conversation",
        "image_dir": "path/to/images/"
    }
}
```

* `meta_file`: path to the converted dataset JSON file (a list of samples).
* `storage_type`: usually set to `"hybrid"`.
* `data_format`: usually set to `"conversation"`.
* `image_dir`: directory path containing the referenced images.

#### Training Script

We provide example training scripts under `scripts/`.
For instance, to fine-tune Ovis2.5 with SFT:

```bash
bash scripts/run_ovis2_5_sft.sh
```

This script configures the DeepSpeed engine, dataset paths, and model checkpoint initialization. Modify it to match your own dataset and environment.

#### Training with MAS (Modality Attention Share) Loss

MAS Loss is a novel training objective that encourages the model to pay more attention to visual tokens, especially when making incorrect predictions. It measures the fraction of attention that lands on vision tokens and applies a constraint to ensure the model looks at images appropriately.

**Key Features:**
- Only penalizes the model when predictions are wrong (task-aware)
- Encourages higher attention to visual tokens for incorrect predictions
- Supports task-specific validation (e.g., counting tasks)
- Can be used alongside standard language modeling loss or as the sole training objective

**MAS Loss Parameters:**
- `--use_mas_loss`: Enable MAS loss (default: `false`)
- `--mas_loss_weight`: Weight for MAS loss component (default: `1.0`)
- `--mas_tau`: Threshold for MAS constraint (default: `0.5`)
  - Lower values (e.g., 0.3-0.4) encourage more attention to vision
  - Higher values (e.g., 0.5-0.6) are more lenient
- `--mas_apply_layers`: Which layers to apply MAS to
  - `"all"`: Apply to all transformer layers
  - `"last"`: Apply only to the last layer
  - `"20,21,22,23"`: Apply to specific layer indices (comma-separated)
- `--mas_only`: Use MAS loss exclusively, disabling standard language modeling loss (default: `false`)
- `--freeze_vision_encoder`: Freeze vision encoder during training (default: `false`)
- `--freeze_text_encoder`: Freeze text encoder during training (default: `false`)

**Example: Training with MAS Loss**

For standard training with both language modeling and MAS loss:

```bash
python -m ovis.train.train \
    --ovis_pretrained_path AIDC-AI/Ovis2.5-2B \
    --data_info_version your_datainfo \
    --data_name your_dataset \
    --data_type conversation \
    --train_modules llm \
    --output_dir ./checkpoints/ovis_mas \
    --num_train_epochs 10 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --learning_rate 1e-5 \
    --use_mas_loss true \
    --mas_loss_weight 0.1 \
    --mas_tau 0.4 \
    --mas_apply_layers "20,21,22,23" \
    --mas_only false \
    --freeze_vision_encoder true \
    --freeze_text_encoder true \
    --bf16 True \
    --gradient_checkpointing True
```

**Example: MAS-Only Training**

For training with MAS loss only (useful for fine-tuning attention patterns without changing language generation):

```bash
python -m ovis.train.train \
    --ovis_pretrained_path AIDC-AI/Ovis2.5-2B \
    --data_info_version your_datainfo \
    --data_name your_dataset \
    --data_type conversation \
    --train_modules llm \
    --output_dir ./checkpoints/ovis_mas_only \
    --num_train_epochs 10 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --learning_rate 1e-5 \
    --use_mas_loss true \
    --mas_loss_weight 1.0 \
    --mas_tau 0.4 \
    --mas_apply_layers "20,21,22,23" \
    --mas_only true \
    --freeze_vision_encoder true \
    --freeze_text_encoder true \
    --bf16 True \
    --gradient_checkpointing True
```

**Example Training Scripts:**

We provide example scripts in `scripts/`:
- `scripts/train_fsc147_with_mas_loss.sh`: Training with MAS loss for counting tasks
- `scripts/run_ovis2_5_mas_loss.sh`: General MAS loss training template

**How MAS Loss Works:**

1. **MAS Score Calculation**: For each layer, compute the fraction of attention that lands on vision tokens:
   ```
   MAS = (attention to vision tokens) / (total attention)
   ```

2. **Constraint Loss**: Apply a hinge loss that penalizes when MAS is below threshold:
   ```
   L_mas = max(0, τ_mas - MAS)
   ```
   This encourages the model to have at least `τ_mas` fraction of attention on vision tokens.

3. **Task-Aware Application**: For counting and other structured tasks, MAS loss is only applied when the model makes incorrect predictions, making training more efficient and targeted.

4. **Layer Selection**: You can apply MAS to all layers, last layer only, or specific layers. Typically, applying to the last few layers is sufficient and more efficient.

**Tips for Using MAS Loss:**

- Start with `mas_tau=0.4` and `mas_loss_weight=0.1` for balanced training
- For counting tasks, use `mas_apply_layers` to target the last few layers (e.g., `"20,21,22,23"`)
- When using `mas_only=true`, you may need to adjust learning rate (typically lower, e.g., `1e-5`)
- Monitor training logs to see MAS scores - they should increase over time if the loss is working
- For memory efficiency, consider freezing vision/text encoders when training with MAS loss

### 2. Fine-tuning with ms-swift

Alternatively, Ovis models can be fine-tuned using [ms-swift](https://github.com/modelscope/ms-swift), a flexible training framework for LLMs.

