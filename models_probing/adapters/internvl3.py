"""
InternVL 3.0 adapter for feature extraction.
Supports encoder, projector, and LLM feature extraction.
Uses InternVL3-2B model.

InternVL 3.0 improvements over 2.5:
- Native Multimodal Pre-Training (unified language + vision training)
- Variable Visual Position Encoding (V2PE) for longer contexts
- Mixed Preference Optimization (MPO) for better reasoning
- Same architecture: ViT-MLP-LLM paradigm
"""
import math
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from .base import VLAdapter


def _find_vision(model):
    """Find vision model in InternVL architecture."""
    # InternVL uses vision_model
    for name in ["vision_model", "visual", "vision_tower"]:
        if hasattr(model, name):
            return getattr(model, name)
    
    raise RuntimeError("No vision module found in InternVL")


def _find_projector(model):
    """Find projector module in InternVL architecture."""
    # InternVL uses mlp1 for projection
    for name in ["mlp1", "mm_projector", "multi_modal_projector", "visual_projector"]:
        if hasattr(model, name):
            return getattr(model, name)
    
    raise RuntimeError("No projector found in InternVL")


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


class InternVL3Adapter(VLAdapter):
    """
    Adapter for InternVL 3.0 model.
    Uses OpenGVLab/InternVL3-2B with Native Multimodal Pre-Training.
    
    Architecture:
    - Vision: InternViT-300M-448px-V2_5
    - Projector: mlp1 (MLP compression)
    - LLM: Qwen2.5-1.5B (total ~2B parameters)
    """
    
    def __init__(self, model_id="OpenGVLab/InternVL3-2B"):
        """
        Initialize InternVL 3.0 adapter.
        
        Args:
            model_id: HuggingFace model identifier (default: InternVL3-2B)
        """
        print(f"Loading {model_id}...")
        self.model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation="eager"  # Disable flash attention to avoid CUDA errors
        )
        
        # Manually disable flash attention in vision encoder layers
        if hasattr(self.model, 'vision_model') and hasattr(self.model.vision_model, 'encoder'):
            for layer in self.model.vision_model.encoder.layers:
                if hasattr(layer, 'attn') and hasattr(layer.attn, 'use_flash_attn'):
                    layer.attn.use_flash_attn = False
        
        # Load tokenizer once for reuse
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        
        self.vision = _find_vision(self.model)
        self.projector = _find_projector(self.model)
        print(f"Model loaded on {self.model.device}")

    @torch.no_grad()
    def encode_grid(self, image):
        """
        Extract grid features from vision encoder.
        
        Args:
            image: PIL Image
            
        Returns:
            dict: {'grid': (1, Hf, Wf, C), 'meta': {'Hf': Hf, 'Wf': Wf, 'stride': stride}}
        """
        # Prepare pixel_values manually
        import torchvision.transforms as T
        transform = T.Compose([
            T.Resize((448, 448)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        pixel_values = transform(image).unsqueeze(0).to(dtype=torch.bfloat16, device=self.model.device)
        
        # Hook vision encoder output
        vit_output = {}
        def hook_vit(mod, inp, out):
            # out could be BaseModelOutput, tuple, or tensor
            if hasattr(out, 'last_hidden_state'):
                vit_output['hidden_states'] = out.last_hidden_state.detach()
            elif isinstance(out, tuple):
                vit_output['hidden_states'] = out[0].detach()
            else:
                vit_output['hidden_states'] = out.detach()
        
        # Register hook on vision model
        handle = self.vision.register_forward_hook(hook_vit)
        
        # Use model's extract_feature method if available
        if hasattr(self.model, 'extract_feature'):
            _ = self.model.extract_feature(pixel_values)
        else:
            # Fallback: call vision model directly
            _ = self.vision(pixel_values)
        
        handle.remove()
        
        if 'hidden_states' not in vit_output:
            raise RuntimeError("Failed to capture vision encoder output")
        
        hs = vit_output['hidden_states']  # (1, T, C)
        B, T, C = hs.shape
        
        # Infer grid dimensions
        Hf, Wf, had_cls = _hw_from_tokens(T)
        
        # Remove CLS token if present
        if had_cls:
            hs = hs[:, 1:, :]
        
        # Reshape to grid
        grid = hs.view(1, Hf, Wf, C).contiguous()
        
        # Calculate stride
        H, W = 448, 448  # Input resolution
        stride = float(H) / float(Hf)
        
        return {
            "grid": grid,
            "meta": {"Hf": Hf, "Wf": Wf, "stride": stride}
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
        # Prepare pixel_values
        import torchvision.transforms as T
        transform = T.Compose([
            T.Resize((448, 448)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        pixel_values = transform(image).unsqueeze(0).to(dtype=torch.bfloat16, device=self.model.device)
        
        # Use extract_feature if available
        if hasattr(self.model, 'extract_feature'):
            vit_embeds = self.model.extract_feature(pixel_values)  # (N, D) or (1, N, D)
            if vit_embeds.dim() == 2:
                vit_embeds = vit_embeds.unsqueeze(0)  # (1, N, D)
        else:
            # Hook projector output during full model forward
            proj_output = {}
            def hook_proj(mod, inp, out):
                proj_output['tokens'] = out.detach()
            
            handle = self.projector.register_forward_hook(hook_proj)
            
            # Need to prepare minimal inputs for forward (use cached tokenizer)
            text = "Describe this image."
            input_ids = self.tokenizer(text, return_tensors='pt')['input_ids'].to(self.model.device)
            
            _ = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                return_dict=True
            )
            handle.remove()
            
            if 'tokens' not in proj_output:
                raise RuntimeError("Failed to capture projector output")
            
            vit_embeds = proj_output['tokens']
        
        # vit_embeds shape: (1, N, D)
        N = vit_embeds.shape[1]
        Hf = Wf = int(math.isqrt(N))
        
        return {
            "tokens": vit_embeds,
            "meta": {"Hf": Hf, "Wf": Wf}
        }

    @torch.no_grad()
    def extract_llm_features(self, image, prompt="Describe the objects in this image.", llm_layer=None):
        """
        Extract features after LLM processing (hook LLM layer output).
        
        InternVL3 uses same token expansion mechanism as InternVL 2.5:
        - Expands <image> → <img><IMG_CONTEXT>*256</img>
        - Model replaces IMG_CONTEXT positions with visual embeddings
        
        Args:
            image: PIL Image
            prompt: Text prompt for LLM
            llm_layer: Which LLM layer to extract from.
                      If None, auto-detect and use layer -2 (second to last).
                      Can be negative index (e.g., -2) or positive (e.g., 20).
            
        Returns:
            dict: {'tokens': (1, N, D), 'meta': {'Hf': Hf, 'Wf': Wf}}
        """
        # Get visual token count from model config
        N_visual = self.model.num_image_token  # 256 for InternVL
        Hf = Wf = int(N_visual ** 0.5)  # sqrt(256) = 16
        
        # Prepare inputs manually (AutoProcessor doesn't work properly)
        import torchvision.transforms as T
        transform = T.Compose([
            T.Resize((448, 448)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        pixel_values = transform(image).unsqueeze(0).to(dtype=torch.bfloat16, device=self.model.device)
        
        # Use cached tokenizer (loaded in __init__)
        # Build prompt with EXPANDED image tokens (256 IMG_CONTEXT tokens)
        # This is the key: InternVL replaces positions with img_context_token_id
        num_img_tok = self.model.num_image_token  # Should be 256
        image_tokens = "<img>" + "<IMG_CONTEXT>" * num_img_tok + "</img>"
        text = f"{image_tokens}\n{prompt}"
        
        # Tokenize
        tok = self.tokenizer(text, return_tensors='pt')
        input_ids = tok['input_ids'].to(self.model.device)
        attention_mask = tok['attention_mask'].to(self.model.device)
        
        # CRITICAL: Assign img_context_token_id to model
        # Without this, comparison returns bool instead of Tensor
        self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        
        # InternVL requires image_flags tensor
        image_flags = torch.tensor([1], dtype=torch.long, device=self.model.device)
        
        # Find LLM layers - InternVL structure: model.language_model.model.layers
        if hasattr(self.model, 'language_model'):
            if hasattr(self.model.language_model, 'model') and hasattr(self.model.language_model.model, 'layers'):
                llm_layers = self.model.language_model.model.layers
            elif hasattr(self.model.language_model, 'layers'):
                llm_layers = self.model.language_model.layers
            else:
                raise RuntimeError("Cannot find LLM layers in InternVL architecture")
        else:
            raise RuntimeError("Cannot find language_model in InternVL")
        
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
        
        # Forward pass with manually prepared inputs
        try:
            _ = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_flags=image_flags,
                return_dict=True
            )
        except Exception as e:
            handle.remove()
            if not hasattr(self, '_llm_fallback_logged'):
                print(f"  Warning: InternVL3 LLM forward failed ({e}), falling back to projector")
                self._llm_fallback_logged = True
            return self.project_tokens(image)
        finally:
            handle.remove()
        
        if 'hidden_states' not in llm_output:
            raise RuntimeError(f"Failed to capture LLM layer {llm_layer} output")
        
        hidden = llm_output['hidden_states']  # (B, seq_len, D)
        
        # Extract visual tokens using mask (expert-recommended approach)
        # Find all positions where IMG_CONTEXT token appears
        mask = (input_ids[0] == self.model.img_context_token_id)  # (seq_len,)
        visual_tokens = hidden[:, mask, :]  # (B, N_visual, D)
        
        # Verify we got the right number of tokens
        if visual_tokens.shape[1] != N_visual:
            if not hasattr(self, '_llm_fallback_logged'):
                print(f"  Warning: Expected {N_visual} visual tokens, got {visual_tokens.shape[1]}, falling back to projector")
                self._llm_fallback_logged = True
            return self.project_tokens(image)
        
        return {
            "tokens": visual_tokens,
            "meta": {"Hf": Hf, "Wf": Wf, "llm_layer": llm_layer}
        }

