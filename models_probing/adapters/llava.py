"""
LLaVA adapter for feature extraction.

Architecture:
- Vision Encoder: CLIP ViT-L/14 @ 336px → 24x24 grid = 576 spatial tokens (+ 1 CLS)
- Projector: MLP (multi-layer perceptron) → maps 1024 dims to LLM hidden size (4096)
- Feature Selection: "default" strategy removes CLS token, keeps only 576 spatial tokens

Key findings from research:
1. Vision encoder outputs: (B, 577, 1024) - includes CLS token
2. After feature selection: (B, 576, 1024) - CLS removed
3. After projector: (B, 576, 4096) - ready for LLM
4. Grid size: 24x24 (perfect square)
"""
import math
import torch
import torch.nn as nn
from transformers import AutoProcessor, LlavaForConditionalGeneration
from .base import VLAdapter


def _hw_from_tokens(T):
    """
    Infer grid height and width from token count.
    
    Args:
        T: Total token count
        
    Returns:
        tuple: (Hf, Wf, had_cls)
    """
    # Try perfect square first
    for t in [T, T - 1]:
        r = int(math.isqrt(t))
        if r * r == t:
            return r, r, (t != T)  # (Hf, Wf, had_cls)
    
    # Fallback: assume square-ish
    r = int(math.sqrt(T))
    return r, r, False


class LlavaAdapter(VLAdapter):
    """Adapter for LLaVA model (1.5 / 1.6)."""
    
    def __init__(self, model_id="llava-hf/llava-1.5-7b-hf"):
        """
        Initialize LLaVA adapter.
        
        Args:
            model_id: HuggingFace model identifier
        """
        print(f"Loading LLaVA model: {model_id}...")
        
        try:
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            print(f"✓ LLaVA model loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load {model_id}: {e}")
            raise
        
        self.model.eval()
        
        # Get components
        self.vision_tower = self.model.vision_tower
        self.projector = self.model.multi_modal_projector
        
        # Get config
        self.config = self.model.config
        self.vision_feature_select_strategy = getattr(
            self.config, 'vision_feature_select_strategy', 'default'
        )
        
        print(f"✓ Vision tower: {type(self.vision_tower).__name__}")
        print(f"✓ Projector: {type(self.projector).__name__}")
        print(f"✓ Feature select strategy: {self.vision_feature_select_strategy}")
    
    @torch.no_grad()
    def encode_grid(self, pil_img):
        """
        Extract encoder features (before projector).
        
        Returns grid of spatial features from CLIP ViT encoder.
        Strategy: Remove CLS token to get pure spatial features.
        
        Args:
            pil_img: PIL Image
            
        Returns:
            dict: {
                "grid": (1, Hf, Wf, D),  # spatial features only
                "meta": {"Hf": int, "Wf": int, "stride": float}
            }
        """
        # Process image
        inputs = self.processor(images=pil_img, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(
            self.model.device, dtype=self.model.dtype
        )
        
        # Forward through vision tower
        image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
        
        # Get last layer hidden states
        if hasattr(image_outputs, 'hidden_states'):
            image_features = image_outputs.hidden_states[-1]  # (1, N, D)
        else:
            image_features = image_outputs[0]
        
        # Feature selection: remove CLS token
        if self.vision_feature_select_strategy == "default":
            # Default: remove CLS token (first token)
            selected_features = image_features[:, 1:]  # (1, N-1, D)
        elif self.vision_feature_select_strategy == "full":
            # Full: keep all tokens including CLS
            selected_features = image_features  # (1, N, D)
        else:
            # Unknown strategy: remove CLS by default
            selected_features = image_features[:, 1:]
        
        B, N, D = selected_features.shape
        
        # Infer grid size
        Hf, Wf, _ = _hw_from_tokens(N)
        assert Hf * Wf == N, f"Grid size {Hf}x{Wf}={Hf*Wf} != token count {N}"
        
        # Reshape to grid: (B, N, D) → (B, Hf, Wf, D)
        grid = selected_features.view(B, Hf, Wf, D).contiguous()
        
        # Calculate stride
        # Image is resized to 336x336, grid is 24x24
        # Stride = 336 / 24 = 14
        img_size = pixel_values.shape[-1]  # should be 336
        stride = float(img_size) / Hf
        
        return {
            "grid": grid,
            "meta": {"Hf": Hf, "Wf": Wf, "stride": stride}
        }
    
    @torch.no_grad()
    def project_tokens(self, pil_img):
        """
        Extract projector features (after MLP projector).
        
        This is the feature that actually goes into the LLM.
        
        Args:
            pil_img: PIL Image
            
        Returns:
            dict: {
                "tokens": (1, N, D_llm),  # projected to LLM space
                "meta": {"Hf": int, "Wf": int, "stride": float}
            }
        """
        # Get encoder features first
        encoder_out = self.encode_grid(pil_img)
        grid = encoder_out["grid"]  # (1, Hf, Wf, D_encoder)
        
        B, Hf, Wf, D = grid.shape
        N = Hf * Wf
        
        # Flatten to (B, N, D)
        tokens = grid.view(B, N, D).contiguous()
        
        # Forward through projector
        projected_tokens = self.projector(tokens)  # (B, N, D_llm)
        
        return {
            "tokens": projected_tokens,
            "meta": encoder_out["meta"]
        }
    
    @torch.no_grad()
    def extract_llm_features(self, pil_img):
        """
        Extract LLM hidden states for image tokens.
        
        Note: LLaVA directly replaces image placeholder tokens with projected vision tokens.
        There's no additional processing in LLM layers for "vision-only" features.
        So this method returns the same as project_tokens.
        
        Args:
            pil_img: PIL Image
            
        Returns:
            dict: {
                "tokens": (1, N, D_llm),  # same as projector output
                "meta": {"Hf": int, "Wf": int, "stride": float}
            }
        """
        # For LLaVA, LLM features = projector features
        # because vision tokens are directly inserted into LLM input
        return self.project_tokens(pil_img)
    
    def to_feature_map(self, grid):
        """
        Convert grid (B, Hf, Wf, D) to feature map (B, C, Hf, Wf).
        
        Args:
            grid: Tensor of shape (B, Hf, Wf, D)
            
        Returns:
            dict: {
                "feat": (B, C, Hf, Wf),
                "stride": float
            }
        """
        B, Hf, Wf, D = grid.shape
        
        # Permute: (B, Hf, Wf, D) → (B, D, Hf, Wf)
        feat = grid.permute(0, 3, 1, 2).contiguous()
        
        # Calculate stride (assuming 336px input)
        stride = 336.0 / Hf
        
        return {
            "feat": feat,
            "stride": stride
        }

