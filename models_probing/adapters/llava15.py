"""
LLaVA 1.5 adapter for feature extraction.
Supports both encoder and projector feature extraction.
Uses LLaVA-1.5-7B model with CLIP vision encoder.
"""
import math
import types
import torch
import torch.nn as nn
from transformers import AutoProcessor, LlavaForConditionalGeneration
from .base import VLAdapter


def _find_vision(model):
    """Find vision model in LLaVA architecture."""
    # LLaVA uses vision_tower
    for name in ["vision_tower", "vision_model", "visual"]:
        if hasattr(model, name):
            return getattr(model, name)
    
    # Check model.model
    if hasattr(model, "model"):
        for name in ["vision_tower", "vision_model", "visual"]:
            if hasattr(model.model, name):
                return getattr(model.model, name)
    
    raise RuntimeError("No vision module found in LLaVA")


def _find_projector(model):
    """Find projector module in LLaVA architecture."""
    # LLaVA typically uses multi_modal_projector
    for name in ["multi_modal_projector", "mm_projector", "visual_projector"]:
        if hasattr(model, name):
            return getattr(model, name)
    
    # Check model.model
    if hasattr(model, "model"):
        for name in ["multi_modal_projector", "mm_projector", "visual_projector"]:
            if hasattr(model.model, name):
                return getattr(model.model, name)
    
    raise RuntimeError("No projector found in LLaVA")


def _hw_from_tokens(T):
    """
    Infer grid height and width from token count.
    
    Args:
        T: Total token count
        
    Returns:
        tuple: (Hf, Wf, had_cls)
    """
    # Try perfect square (with and without CLS token)
    for t in [T, T - 1]:
        r = int(math.isqrt(t))
        if r * r == t:
            return r, r, (t != T)  # (Hf, Wf, had_cls)
    
    raise RuntimeError(f"Cannot infer grid dimensions from {T} tokens")


class Llava15Adapter(VLAdapter):
    """Adapter for LLaVA 1.5 model."""
    
    def __init__(self, model_id="llava-hf/llava-1.5-7b-hf"):
        """
        Initialize LLaVA 1.5 adapter.
        
        Args:
            model_id: HuggingFace model identifier
        """
        print(f"Loading {model_id}...")
        try:
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_id, 
                torch_dtype=torch.bfloat16, 
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        except Exception as e:
            print(f"Failed to load {model_id}: {e}")
            print("Trying fallback model: llava-hf/llava-1.5-7b-hf")
            self.model = LlavaForConditionalGeneration.from_pretrained(
                "llava-hf/llava-1.5-7b-hf", 
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
            self.proc = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        
        self.vision = _find_vision(self.model)
        self.projector = _find_projector(self.model)
        print(f"Model loaded on {self.model.device}")

    @torch.no_grad()
    def encode_grid(self, image):
        """
        Extract grid features from CLIP vision encoder.
        
        Args:
            image: PIL Image
            
        Returns:
            dict: {'grid': (1, Hf, Wf, C), 'meta': {'Hf': Hf, 'Wf': Wf, 'stride': stride}}
        """
        # LLaVA uses specific prompt format
        prompt = "USER: <image>\nDescribe this image.\nASSISTANT:"
        inputs = self.proc(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Extract vision features directly from vision tower
        pixel_values = inputs["pixel_values"]
        
        # Get vision tower output
        vision_outputs = self.vision(pixel_values, output_hidden_states=True)
        
        # Use last hidden state from vision encoder
        hs = vision_outputs.last_hidden_state  # (1, T, C)
        B, T, C = hs.shape
        
        # Infer grid dimensions
        try:
            Hf, Wf, had_cls = _hw_from_tokens(T)
            
            # Validate dimensions
            if Hf <= 0 or Wf <= 0 or Hf > 64 or Wf > 64:
                raise ValueError(f"Invalid grid dimensions: Hf={Hf}, Wf={Wf}")
                
        except Exception as e:
            print(f"Warning: Failed to infer grid dimensions: {e}")
            # Fallback to square grid assumption
            Hf = Wf = int(math.sqrt(T))
            had_cls = False
            if Hf * Wf != T:
                print(f"Warning: Token count {T} is not a perfect square, using Hf={Hf}, Wf={Wf}")
        
        # Remove CLS token if present
        if had_cls:
            hs = hs[:, 1:, :]
            T = T - 1
        
        # Reshape to grid
        grid = hs.view(1, Hf, Wf, C).contiguous()
        
        # Calculate stride
        H, W = pixel_values.shape[-2:]
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
        
        Args:
            image: PIL Image
            
        Returns:
            dict: {'tokens': (1, N, D), 'meta': {'Hf': Hf, 'Wf': Wf}}
        """
        # Get grid features first
        G = self.encode_grid(image)
        B, Hf, Wf, C = G["grid"].shape
        
        # Apply projector to get text-aligned tokens
        grid_flat = G["grid"].view(1, Hf * Wf, C)  # (1, N, C)
        
        # LLaVA projector is typically a 2-layer MLP
        # Apply projector
        tokens = self.projector(grid_flat)  # (1, N, D_text)
        
        return {
            "tokens": tokens, 
            "meta": {"Hf": Hf, "Wf": Wf}
        }

    @torch.no_grad()
    def extract_llm_features(self, image, prompt="USER: <image>\nDescribe the objects in this image.\nASSISTANT:", llm_layer=None):
        """
        Extract LLM features: Hook LLM layer output to get visual tokens after LLM processing.
        
        This extracts visual tokens AFTER they've been processed by LLM attention layers,
        so they have contextual information from text tokens.
        
        Visual tokens position is FIXED in LLaVA: they start at position 5 (after "<s>USER: ")
        and don't change position through LLM layers (attention doesn't change token positions).
        
        Args:
            image: PIL Image
            prompt: Text prompt for LLM
            llm_layer: Which LLM layer to extract from.
                      If None, auto-detect and use layer -2 (second to last).
                      Can be negative index (e.g., -2) or positive (e.g., 15).
            
        Returns:
            dict: {'tokens': (1, N, D), 'meta': {'Hf': Hf, 'Wf': Wf}}
        """
        # Get grid features first to know spatial dimensions
        G = self.encode_grid(image)
        B, Hf, Wf, C = G["grid"].shape
        N_visual = Hf * Wf
        
        # Prepare inputs
        inputs = self.proc(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Hook target LLM layer output
        llm_output = {}
        
        def hook_llm_layer(mod, inp, out):
            # Output of LLM layer (after attention + FFN)
            if isinstance(out, tuple):
                hidden_states = out[0].detach()
            else:
                hidden_states = out.detach()
            llm_output['hidden_states'] = hidden_states
        
        # Find LLM layers - LLaVA structure: model.language_model.layers
        # (not model.language_model.model.layers)
        if hasattr(self.model.language_model, 'layers'):
            llm_layers = self.model.language_model.layers
        elif hasattr(self.model.language_model, 'model') and hasattr(self.model.language_model.model, 'layers'):
            llm_layers = self.model.language_model.model.layers
        else:
            raise RuntimeError("Cannot find LLM layers in model architecture")
        
        # Auto-detect layer if not specified
        num_layers = len(llm_layers)
        if llm_layer is None:
            llm_layer = num_layers - 2  # Second to last layer
            if not hasattr(self, '_llm_layer_logged'):
                print(f"  Auto-detected {num_layers} LLM layers, using layer {llm_layer}")
                self._llm_layer_logged = True
        elif llm_layer < 0:
            # Convert negative index to positive
            llm_layer = num_layers + llm_layer
            if not hasattr(self, '_llm_layer_logged'):
                print(f"  Using LLM layer {llm_layer} (from {num_layers} total layers)")
                self._llm_layer_logged = True
        
        # Validate layer index
        if llm_layer >= num_layers:
            llm_layer = num_layers - 1
            if not hasattr(self, '_llm_layer_logged'):
                print(f"  Warning: layer index too high, using last layer {llm_layer}")
                self._llm_layer_logged = True
        
        # Hook the specified target LLM layer
        handle = llm_layers[llm_layer].register_forward_hook(hook_llm_layer)
        
        # Forward pass
        _ = self.model(**inputs, return_dict=True)
        
        handle.remove()
        
        # Check capture
        if 'hidden_states' not in llm_output:
            raise RuntimeError(f"Failed to capture LLM layer {llm_layer} output")
        
        llm_hidden = llm_output['hidden_states']  # (1, seq_len, D)
        
        # Visual tokens start at position 5 (after "<s>USER: ")
        # This is fixed for LLaVA 1.5 with this prompt format
        visual_start = 5
        visual_end = visual_start + N_visual
        
        # Validate
        if visual_end > llm_hidden.shape[1]:
            raise RuntimeError(
                f"Visual tokens [{visual_start}:{visual_end}] exceed sequence length {llm_hidden.shape[1]}"
            )
        
        # Extract visual tokens from target layer
        visual_tokens = llm_hidden[:, visual_start:visual_end, :].contiguous()
        
        return {
            "tokens": visual_tokens,
            "meta": {"Hf": Hf, "Wf": Wf, "llm_layer": llm_layer, "position": visual_start}
        }

