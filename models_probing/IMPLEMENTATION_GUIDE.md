# VLM Feature Extraction Guide

## Overview

This guide documents the generalized approach for extracting features at different architectural levels from Vision-Language Models (VLMs).

## Three Feature Extraction Levels

All VLMs have 3 distinct feature extraction points:

### 1. **Encoder Level** (`encode_grid`)
- **What**: Raw visual features from the vision encoder (before alignment with text)
- **Examples**: CLIP, SigLIP, InternViT features
- **Implementation**: Hook the output of the vision encoder
- **Output shape**: `(B, Hf, Wf, C_vision)` where `C_vision` varies by encoder

### 2. **Projector Level** (`project_tokens`)
- **What**: Visual features after projector/alignment layer (aligned with text space, but no context yet)
- **Examples**: Multi-modal projector output, MLP output
- **Implementation**: Hook the projector output OR apply projector to encoder features
- **Output shape**: `(B, N, D_text)` where `D_text` matches LLM hidden dim, `N = Hf * Wf`

### 3. **LLM Level** (`extract_llm_features`)
- **What**: Visual tokens AFTER being processed by LLM attention layers (with textual context)
- **Key insight**: Extract from middle/late LLM layers (e.g., layer 15, 20, or last layer)
- **Implementation**: Hook LLM layer output, extract visual tokens using their position
- **Output shape**: `(B, N, D_text)` - same as projector but semantically enriched

---

## Critical Insight: Visual Token Positions are FIXED

**Key Discovery**: In transformer-based LLMs, attention does NOT change token positions in the sequence. Visual tokens remain at their original positions throughout all LLM layers.

### Why This Matters:
- You don't need to "search" for visual tokens at each layer
- Position found at LLM layer 0 input = position at ANY layer
- Simplifies implementation significantly

### Finding Visual Token Position (One-Time):

```python
# Method 1: Compare projector output with LLM layer 0 input
# 1. Hook projector output (reference)
# 2. Hook LLM layer 0 input  
# 3. Find where projector tokens appear in layer 0 input
# 4. Use this position for ALL layers

# Method 2: Use known position from model architecture
# - LLaVA 1.5: position 5 (after "<s>USER: ")
# - Can vary based on prompt template
```

---

## Generalized Implementation Pattern

### Step 1: Find Visual Token Position (Debug Script)

Create a debug script to find the position once:

```python
import torch
from transformers import AutoModel, AutoProcessor

# Load model
model = AutoModel.from_pretrained(...)
processor = AutoProcessor.from_pretrained(...)

# Prepare input
inputs = processor(text=prompt, images=image, return_tensors="pt")

# Hook projector and LLM layer 0
projector_out = {}
layer0_input = {}

def hook_projector(mod, inp, out):
    projector_out['tokens'] = out.detach()

def hook_layer0(mod, inp, out):
    if isinstance(inp, tuple):
        layer0_input['hidden'] = inp[0].detach()

# Register hooks
h1 = model.projector.register_forward_hook(hook_projector)
h2 = model.llm.layers[0].register_forward_hook(hook_layer0)

# Forward pass
_ = model(**inputs)
h1.remove()
h2.remove()

# Find position by comparing
proj = projector_out['tokens'][0, 0, :]  # First visual token
layer0 = layer0_input['hidden'][0, :, :]  # Full sequence

for i in range(layer0.shape[0]):
    dist = torch.norm(layer0[i] - proj).item()
    if dist < 1e-4:
        print(f"Visual tokens start at position {i}")
        break
```

### Step 2: Implement `extract_llm_features`

```python
@torch.no_grad()
def extract_llm_features(self, image, prompt, llm_layer=15):
    """
    Extract visual tokens from LLM layer output.
    
    Args:
        image: PIL Image
        prompt: Text prompt
        llm_layer: Which LLM layer to extract from (default: middle layer)
    
    Returns:
        dict: {'tokens': (1, N, D), 'meta': {...}}
    """
    # 1. Get spatial dimensions from encoder
    enc = self.encode_grid(image)
    Hf, Wf = enc['meta']['Hf'], enc['meta']['Wf']
    N_visual = Hf * Wf
    
    # 2. Prepare inputs
    inputs = self.proc(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
    
    # 3. Hook target LLM layer
    llm_output = {}
    
    def hook_llm_layer(mod, inp, out):
        if isinstance(out, tuple):
            llm_output['hidden'] = out[0].detach()
        else:
            llm_output['hidden'] = out.detach()
    
    # Find LLM layers (varies by architecture)
    llm_layers = self._find_llm_layers()  # Helper to find layers
    if llm_layer >= len(llm_layers):
        llm_layer = len(llm_layers) - 1
    
    handle = llm_layers[llm_layer].register_forward_hook(hook_llm_layer)
    
    # 4. Forward pass
    _ = self.model(**inputs)
    handle.remove()
    
    # 5. Extract visual tokens using KNOWN position
    # NOTE: Find this position once using debug script above
    visual_start = 5  # Example for LLaVA
    visual_end = visual_start + N_visual
    
    hidden = llm_output['hidden']  # (1, seq_len, D)
    visual_tokens = hidden[:, visual_start:visual_end, :].contiguous()
    
    return {
        'tokens': visual_tokens,
        'meta': {'Hf': Hf, 'Wf': Wf, 'llm_layer': llm_layer, 'position': visual_start}
    }
```

### Step 3: Helper to Find LLM Layers

Different models have different paths to LLM layers:

```python
def _find_llm_layers(self):
    """Find LLM layers in model architecture."""
    # Try common paths
    candidates = [
        self.model.language_model.layers,           # LLaVA
        self.model.language_model.model.layers,     # Some variants
        self.model.llm.layers,                      # Ovis, InternVL
        self.model.llm.model.layers,                # Qwen variants
    ]
    
    for candidate in candidates:
        if hasattr(self.model, ...):  # Check if path exists
            return candidate
    
    raise RuntimeError("Cannot find LLM layers")
```

---

## Model-Specific Notes

### LLaVA 1.5
- **Visual tokens position**: 5 (after `"<s>USER: "`)
- **Encoder**: `model.vision_tower` (CLIP)
- **Projector**: `model.multi_modal_projector`
- **LLM layers**: `model.language_model.layers`
- **Total layers**: 32 (LLaMA-7B backbone)

### Ovis 2.5
- **Special architecture**: Uses codebook + VTE embedding
- **Encoder**: `visual_tokenizer.vit.vision_model.post_layernorm` (SigLIP)
- **Projector**: After VTE embedding lookup
- **LLM layers**: `llm.model.layers`
- **Note**: Must convert codebook logits → continuous features via VTE

### InternVL 2.5 / 3.5
- **Encoder**: `model.vision_model` (InternViT)
- **Projector**: `mlp1` (with dynamic resolution handling)
- **LLM layers**: `model.language_model.layers`
- **Note**: InternVL 3.5 compresses spatial resolution in projector

### Qwen3-VL
- **MoE architecture**: 30.5B total params, 3.3B activated
- **Encoder**: Vision encoder (varies by version)
- **Projector**: Multi-modal projector
- **LLM layers**: `model.language_model.model.layers`
- **Note**: Check `text_config.hidden_size` for MoE configs

---

## Best Practices

### 1. Always verify visual token position
- Don't hardcode positions without verification
- Create debug scripts to find positions for each model
- Document findings in model-specific adapters

### 2. Choose appropriate LLM layer
- **Early layers (0-5)**: Close to projector features, minimal context
- **Middle layers (10-20)**: Balanced - some LLM processing with context
- **Late layers (25-31)**: Heavily processed, rich context
- **Default recommendation**: Layer 15 (middle) or last layer

### 3. Handle edge cases
- Validate sequence length before extraction
- Handle variable prompt templates
- Check for special tokens (CLS, BOS, etc.)

### 4. Test thoroughly
- Test all 3 levels (encoder, projector, llm)
- Verify output shapes match expectations
- Check that features are semantically meaningful (not all zeros/NaNs)

---

## Debugging Checklist

When implementing for a new model:

- [ ] Find vision encoder module path
- [ ] Find projector module path
- [ ] Find LLM layers path
- [ ] Determine number of LLM layers
- [ ] Find visual token position (debug script)
- [ ] Verify position doesn't change across layers
- [ ] Test encoder extraction
- [ ] Test projector extraction
- [ ] Test LLM extraction (multiple layers)
- [ ] Verify shapes are consistent
- [ ] Document findings in adapter docstring

---

## Common Pitfalls

### ❌ Wrong: Searching for tokens at every layer
```python
# Don't do this - wastes computation
for each_layer:
    find_visual_tokens_by_comparison()
```

### ✅ Right: Find once, use everywhere
```python
# Find position once (or use known position)
visual_pos = 5  # Known from architecture

# Use same position for any layer
visual_tokens = llm_hidden[:, visual_pos:visual_pos+N, :]
```

### ❌ Wrong: Using projector output as "LLM features"
```python
# This is projector level, NOT LLM level
def extract_llm_features(self, image):
    return self.project_tokens(image)  # Wrong!
```

### ✅ Right: Extract from actual LLM layers
```python
# Extract from LLM layer output
def extract_llm_features(self, image, llm_layer=15):
    # Hook LLM layer 15
    # Extract visual tokens at their position
    return visual_tokens_from_llm
```

---

## Summary

**Key Takeaways:**
1. **3 levels**: Encoder (raw) → Projector (aligned) → LLM (contextualized)
2. **Position is fixed**: Visual tokens don't move through LLM layers
3. **Find position once**: Use debug script to find, then hardcode or store
4. **Hook the right layer**: Hook LLM layer output, not just projector
5. **Verify everything**: Test shapes and positions for each new model

This approach works for ALL transformer-based VLMs!

