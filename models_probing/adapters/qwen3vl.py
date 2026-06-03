"""
Qwen2-VL adapter for feature extraction.
NOTE: Qwen3-VL only has 235B variant (too large).
Using Qwen2-VL as closest alternative with 2B/7B options.
"""
import math
import types
import torch
import torch.nn as nn
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, AutoModelForImageTextToText
from .base import VLAdapter


def _find_vision(model):
    """Find vision model in Qwen-3-VL architecture."""
    for name in ["vision_model", "vision_tower", "visual"]:
        if hasattr(model, name):
            return getattr(model, name)
    
    if hasattr(model, "model"):
        for name in ["vision_model", "vision_tower", "visual"]:
            if hasattr(model.model, name):
                return getattr(model.model, name)
    
    raise RuntimeError("No vision module found")


def _find_projector(model):
    """Find projector module in Qwen-3-VL architecture."""
    for name in ["multi_modal_projector", "visual_projector", "mm_projector", 
                 "vision_proj", "image_projection"]:
        if hasattr(model, name):
            return getattr(model, name)
    
    # Fallback: Linear layer with output size matching hidden_size
    # Handle MoE config (may not have hidden_size attribute)
    hidden_size = None
    if hasattr(model.config, 'hidden_size'):
        hidden_size = model.config.hidden_size
    elif hasattr(model.config, 'text_config') and hasattr(model.config.text_config, 'hidden_size'):
        hidden_size = model.config.text_config.hidden_size
    
    if hidden_size:
        for _, mod in model.named_modules():
            if isinstance(mod, nn.Linear) and mod.out_features == hidden_size:
                return mod
    
    print("Warning: No projector found explicitly")
    return None


def _hw_from_tokens(T):
    """
    Infer grid height and width from token count.
    
    Args:
        T: Total token count
        
    Returns:
        tuple: (Hf, Wf, had_cls)
    """
    # Try perfect square
    for t in [T, T - 1]:
        r = int(math.isqrt(t))
        if r * r == t:
            return r, r, (t != T)
    
    raise RuntimeError("Cannot infer grid dimensions")


class Qwen3VLAdapter(VLAdapter):
    """
    Adapter for Qwen3-VL model (Regular 4B architecture).
    Uses Qwen3-VL-4B-Instruct (4B regular, not MoE).
    """
    
    def __init__(self, model_id="Qwen/Qwen3-VL-4B-Instruct", param_size="4b"):
        """
        Initialize Qwen3-VL adapter.
        
        Args:
            model_id: HuggingFace model identifier
            param_size: Model size ("4b" default, "30b" for MoE variant)
        """
        # Map size to model ID for Qwen3-VL
        size_map = {
            "4b": "Qwen/Qwen3-VL-4B-Instruct",  # Default: 4B regular
            "30b": "Qwen/Qwen3-VL-30B-A3B-Instruct",  # MoE: 30.5B total, 3.3B activated
            "235b": "Qwen/Qwen3-VL-235B-A22B-Instruct"  # MoE: 235B total, 22B activated
        }
        
        if param_size.lower() in size_map:
            model_id = size_map[param_size.lower()]
        
        print(f"Loading {model_id}...")
        
        is_moe = "A3B" in model_id or "A22B" in model_id
        if is_moe:
            print(f"NOTE: Using Qwen3-VL MoE model")
        
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id, 
                torch_dtype=torch.bfloat16, 
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        except Exception as e:
            print(f"Failed to load {model_id}: {e}")
            print("Trying fallback: Qwen/Qwen3-VL-30B-A3B-Instruct")
            self.model = AutoModelForImageTextToText.from_pretrained(
                "Qwen/Qwen3-VL-30B-A3B-Instruct", 
                torch_dtype=torch.bfloat16, 
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        
        try:
            self.proc = AutoProcessor.from_pretrained(model_id)
        except Exception as e:
            print(f"Failed to load processor for {model_id}: {e}")
            print("Using fallback processor")
            self.proc = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-30B-A3B-Instruct")
        self.vision = _find_vision(self.model)
        self.projector = _find_projector(self.model)
        print(f"Model loaded on {self.model.device}")

    @torch.no_grad()
    def encode_grid(self, image):
        """
        Extract grid features from vision encoder.
        
        CRITICAL FIX: Hook vision encoder output, NOT LLM output.
        Previous bug: used hidden_states[-1] which is LLM output (2560 dim).
        Correct: Hook visual model output (1024 dim from vision encoder).
        
        Args:
            image: PIL Image
            
        Returns:
            dict: {'grid': (1, Hf, Wf, C), 'meta': {'Hf': Hf, 'Wf': Wf, 'stride': stride}}
        """
        # Use image placeholder tokens for Qwen3-VL
        inputs = self.proc(images=image, text="<|image_pad|>", return_tensors="pt")
        # Ensure input_ids is LongTensor
        if "input_ids" in inputs:
            inputs["input_ids"] = inputs["input_ids"].long()
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Hook encoder output from blocks[-1] (PRE-normalization)
        # LayerNorm wipes magnitude (std~0.4), but InternVL uses pre-norm (std~1.3)
        # We'll z-score normalize to match InternVL's dynamic range
        vision_output = {}
        def hook_vision_blocks(mod, inp, out):
            # Output from last transformer block (before merger)
            # Shape: (N, C) without batch dimension
            vision_output['features'] = out.detach()
        
        # Hook blocks[-2] to get encoder output with more spatial details
        # blocks[-1] is too abstract, blocks[-2] preserves more low-level features
        if not hasattr(self.model, 'visual') or not hasattr(self.model.visual, 'blocks'):
            raise RuntimeError("Cannot find visual.blocks in Qwen 3-VL")
        
        last_block = self.model.visual.blocks[-2]  # Changed from [-1] to [-2]
        handle = last_block.register_forward_hook(hook_vision_blocks)
        
        try:
            # Forward to trigger hook
            _ = self.model(
                **inputs,
                output_hidden_states=False,
                return_dict=True
            )
        finally:
            handle.remove()
        
        # Get hooked vision features
        if 'features' not in vision_output:
            raise RuntimeError("Failed to hook vision encoder output")
        
        hs = vision_output['features']  # (N, C_vision) - NO batch dimension!
        
        # Add batch dimension if missing
        if len(hs.shape) == 2:
            hs = hs.unsqueeze(0)  # (1, N, C_vision)
        
        # Global normalization (preserve per-channel variance like InternVL)
        # LayerNorm wipes magnitude (std~0.4), but InternVL keeps it (std~1.3)
        # Global normalization preserves channel importance (some channels have high std, some low)
        mean = hs.mean()  # Global mean
        std = hs.std() + 1e-6  # Global std
        hs = (hs - mean) / std  # Global normalization - preserves per-channel variance!
        
        B, T, C = hs.shape
        
        # Infer grid dimensions with error handling
        try:
            Hf, Wf, had_cls = _hw_from_tokens(T)
            
            # Validate dimensions
            if Hf <= 0 or Wf <= 0 or Hf > 64 or Wf > 64:
                raise ValueError(f"Invalid grid dimensions: Hf={Hf}, Wf={Wf}")
                
        except Exception as e:
            print(f"Warning: Failed to infer grid dimensions: {e}")
            # Fallback to square grid assumption
            import math
            Hf = Wf = int(math.sqrt(T))
            had_cls = False
            if Hf * Wf != T:
                print(f"Warning: Token count {T} is not a perfect square, using Hf={Hf}, Wf={Wf}")
        
        # Remove CLS token if present
        if had_cls:
            hs = hs[:, 1:, :]
        
        # Reshape to grid
        grid = hs.view(1, Hf, Wf, C).contiguous()
        
        # Calculate stride with float precision
        H, W = inputs["pixel_values"].shape[-2:]
        stride_h = float(H) / float(Hf)
        stride_w = float(W) / float(Wf)
        
        return {
            "grid": grid, 
            "meta": {"Hf": Hf, "Wf": Wf, "stride": stride_h}
        }

    @torch.no_grad()
    def project_tokens(self, image):
        """
        Extract tokens after projector/alignment layer.
        
        CRITICAL FIX: Use visual merger/projector directly on encoder output.
        Previous bug: Hooked wrong projector or LLM input (2560 dim).
        Correct: Apply visual.merger to encoder output (1024→2560 dim).
        
        Args:
            image: PIL Image
            
        Returns:
            dict: {'tokens': (1, N, D), 'meta': {'Hf': Hf, 'Wf': Wf}}
        """
        # Hook merger output (visual.merger is the projector)
        # It does: spatial merging + projection
        merger_output = {}
        def hook_merger(mod, inp, out):
            # Merger output: (N_merged, C_proj) without batch dimension  
            merger_output['tokens'] = out.detach()
        
        if not hasattr(self.model, 'visual') or not hasattr(self.model.visual, 'merger'):
            raise RuntimeError("Cannot find visual.merger in Qwen 3-VL architecture")
        
        handle = self.model.visual.merger.register_forward_hook(hook_merger)
        
        # Prepare inputs
        inputs = self.proc(images=image, text="<|image_pad|>", return_tensors="pt")
        if "input_ids" in inputs:
            inputs["input_ids"] = inputs["input_ids"].long()
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        try:
            # Forward to trigger hook
            _ = self.model(**inputs, output_hidden_states=False, return_dict=True)
        finally:
            handle.remove()
        
        if 'tokens' not in merger_output:
            raise RuntimeError("Failed to hook merger output")
        
        tokens = merger_output['tokens']  # (N_merged, C_proj)
        
        # Add batch dimension if missing
        if len(tokens.shape) == 2:
            tokens = tokens.unsqueeze(0)  # (1, N_merged, C_proj)
        
        # Infer spatial dimensions from token count
        N = tokens.shape[1]
        Hf = Wf = int(N ** 0.5)
        
        return {
            "tokens": tokens, 
            "meta": {"Hf": Hf, "Wf": Wf}
        }

    @torch.no_grad()
    def _project_tokens_fallback_hook(self, image):
        """
        DEPRECATED: Old fallback method using hooks. Keeping for reference.
        Use project_tokens() instead which applies visual.merger directly.
        """
        # Prepare minimal input to trigger vision pipeline
        inputs = self.proc(images=image, text="<|image_pad|>", return_tensors="pt")
        if "input_ids" in inputs:
            inputs["input_ids"] = inputs["input_ids"].long()
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Hook projector output (or LLM layer 0 input as fallback)
        proj_output = {}
        handle = None
        
        # Try to hook projector directly
        if self.projector is not None:
            def hook_proj(mod, inp, out):
                proj_output['tokens'] = out.detach()
            handle = self.projector.register_forward_hook(hook_proj)
        else:
            # Fallback: hook LLM layer 0 input (post-projector embeddings)
            for name, mod in self.model.named_modules():
                if 'language_model.model.layers.0' in name or 'model.layers.0' in name:
                    def hook_llm_input(mod, inp, out):
                        if isinstance(inp, tuple) and len(inp) > 0:
                            proj_output['tokens'] = inp[0].detach()
                    handle = mod.register_forward_hook(hook_llm_input)
                    break
        
        if handle is None:
            raise RuntimeError("Cannot hook projector or LLM layer 0")
        
        # Forward to trigger hooks
        _ = self.model(**inputs, output_hidden_states=True, return_dict=True)
        handle.remove()
        
        if 'tokens' not in proj_output:
            raise RuntimeError("Failed to capture projector output")
        
        hidden = proj_output['tokens']  # (B, seq_len, D), (B, N, D), or (N, D)
        
        # Handle different output shapes
        if len(hidden.shape) == 2:
            # Direct projector output without batch: (N, D)
            N, D = hidden.shape
            hidden = hidden.unsqueeze(0)  # (1, N, D)
            Hf = Wf = int(N ** 0.5)
            return {
                "tokens": hidden, 
                "meta": {"Hf": Hf, "Wf": Wf}
            }
        elif len(hidden.shape) == 3:
            B, N_or_seq, D = hidden.shape
            # If it's direct projector output (reasonable N), return as is
            if self.projector is not None and N_or_seq < 10000:  # Reasonable token count
                Hf = Wf = int(N_or_seq ** 0.5)
                return {
                    "tokens": hidden, 
                    "meta": {"Hf": Hf, "Wf": Wf}
                }
        
        # Otherwise extract from LLM input sequence
        B, seq_len, D = hidden.shape
        
        # Get expected dimensions from image
        p = getattr(getattr(self.model.config, "vision_config", None), "patch_size", 14)
        H, W = inputs["pixel_values"].shape[-2:]
        T_expected = (H // p) * (W // p)
        Hf = Wf = int(T_expected ** 0.5)
        
        # Extract visual tokens (heuristic: after initial text tokens)
        visual_start = min(5, seq_len // 4)
        visual_end = visual_start + T_expected
        visual_end = min(visual_end, seq_len - 3)
        
        if visual_end - visual_start >= T_expected:
            visual_tokens = hidden[:, visual_start:visual_start + T_expected, :]
        else:
            visual_tokens = hidden[:, visual_start:visual_end, :]
            # Pad if needed
            if visual_tokens.shape[1] < T_expected:
                pad = torch.zeros(B, T_expected - visual_tokens.shape[1], D, 
                                device=visual_tokens.device, dtype=visual_tokens.dtype)
                visual_tokens = torch.cat([visual_tokens, pad], dim=1)
        
        return {
            "tokens": visual_tokens, 
            "meta": {"Hf": Hf, "Wf": Wf}
        }

    @torch.no_grad()
    def extract_llm_features(self, image, prompt="Describe the objects in this image.", llm_layer=None):
        """
        Extract features after LLM processing (hook LLM layer output).
        
        Args:
            image: PIL Image
            prompt: Text prompt for LLM
            llm_layer: Which LLM layer to extract from.
                      If None, auto-detect and use layer -2 (second to last).
                      Can be negative index (e.g., -2) or positive (e.g., 20).
            
        Returns:
            dict: {'tokens': (1, N, D), 'meta': {'Hf': Hf, 'Wf': Wf}}
        """
        # Get grid features first to know spatial dimensions
        G = self.encode_grid(image)
        B, Hf, Wf, C = G["grid"].shape
        N_visual = Hf * Wf
        
        # Prepare inputs for full model with image tokens
        inputs = self.proc(
            text=prompt + "<|image_pad|>",
            images=image, 
            return_tensors="pt"
        )
        # Ensure input_ids is LongTensor
        if "input_ids" in inputs:
            inputs["input_ids"] = inputs["input_ids"].long()
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Find LLM layers - Qwen3-VL structure: model.language_model.layers
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'language_model') and hasattr(self.model.model.language_model, 'layers'):
            llm_layers = self.model.model.language_model.layers
        else:
            raise RuntimeError("Cannot find LLM layers in Qwen3-VL architecture")
        
        # Auto-detect layer if not specified
        num_layers = len(llm_layers)
        if llm_layer is None:
            llm_layer = num_layers - 2  # Second to last layer
            if not hasattr(self, '_llm_layer_logged'):
                print(f"  Auto-detected {num_layers} LLM layers, using layer {llm_layer}")
                self._llm_layer_logged = True
        elif llm_layer < 0:
            llm_layer = num_layers + llm_layer
            if not hasattr(self, '_llm_layer_logged'):
                print(f"  Using LLM layer {llm_layer} (from {num_layers} total layers)")
                self._llm_layer_logged = True
        
        # Validate
        if llm_layer >= num_layers:
            llm_layer = num_layers - 1
            if not hasattr(self, '_llm_layer_logged'):
                print(f"  Warning: layer index too high, using last layer {llm_layer}")
                self._llm_layer_logged = True
        
        # Hook target LLM layer output
        llm_output = {}
        
        def hook_llm_layer(mod, inp, out):
            if isinstance(out, tuple):
                llm_output['hidden_states'] = out[0].detach()
            else:
                llm_output['hidden_states'] = out.detach()
        
        handle = llm_layers[llm_layer].register_forward_hook(hook_llm_layer)
        
        # Forward pass
        try:
            _ = self.model(**inputs, return_dict=True)
        finally:
            handle.remove()
        
        if 'hidden_states' not in llm_output:
            raise RuntimeError(f"Failed to capture LLM layer {llm_layer} output")
        
        hidden_states = llm_output['hidden_states']  # (1, seq_len, hidden_dim)
        
        # Find image token positions in the sequence
        input_ids = inputs["input_ids"][0]  # (seq_len,)
        
        # Look for image token markers (Qwen3-VL specific)
        image_start = None
        image_end = None
        
        for i, token_id in enumerate(input_ids):
            if token_id.item() in [151644, 151645, 151646]:  # Qwen3-VL image tokens
                if image_start is None:
                    image_start = i + 1  # Skip the marker token
                else:
                    image_end = i
                    break
        
        # Fallback: if we can't find markers, use heuristic
        if image_start is None or image_end is None:
            image_start = 1  # Skip first special token
            image_end = image_start + N_visual
        
        # Extract image tokens
        image_tokens = hidden_states[0, image_start:image_end, :]  # (N, D)
        
        # Apply final normalization layer to match model output
        # This normalizes LLM features to similar magnitude as encoder/projector
        if hasattr(self.model.language_model, 'norm'):
            image_tokens = self.model.language_model.norm(image_tokens)
        
        # Reshape to match spatial dimensions
        if image_tokens.shape[0] == N_visual:
            tokens = image_tokens.unsqueeze(0)  # (1, N, D)
        else:
            # Handle mismatch by padding or truncating
            if image_tokens.shape[0] > N_visual:
                tokens = image_tokens[:N_visual].unsqueeze(0)
            else:
                # Pad with zeros
                padding = torch.zeros(
                    N_visual - image_tokens.shape[0], 
                                    image_tokens.shape[1], 
                                    device=image_tokens.device, 
                    dtype=image_tokens.dtype
                )
                tokens = torch.cat([image_tokens, padding], dim=0).unsqueeze(0)
        
        return {
            "tokens": tokens,
            "meta": {"Hf": Hf, "Wf": Wf, "llm_layer": llm_layer}
        }

