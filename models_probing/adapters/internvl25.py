"""
InternVL 2.5 adapter for feature extraction.
Supports both encoder and projector feature extraction.
Uses InternVL2_5-2B model (compact version).
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


class InternVL25Adapter(VLAdapter):
    """Adapter for InternVL 2.5 model (using 2B variant)."""
    
    def __init__(self, model_id="OpenGVLab/InternVL2_5-2B"):
        """
        Initialize InternVL 2.5 adapter.
        
        Args:
            model_id: HuggingFace model identifier (default: 2B model)
        """
        print(f"Loading {model_id}...")
        try:
            self.model = AutoModel.from_pretrained(
                model_id, 
                torch_dtype=torch.bfloat16, 
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                attn_implementation="eager"  # Disable flash attention (causes CUDA errors)
            )
        except Exception as e:
            print(f"Failed to load {model_id}: {e}")
            print("Trying fallback model: OpenGVLab/InternVL2_5-2B")
            self.model = AutoModel.from_pretrained(
                "OpenGVLab/InternVL2_5-2B", 
                torch_dtype=torch.bfloat16, 
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                attn_implementation="eager"  # Disable flash attention
            )
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        except Exception as e:
            print(f"Failed to load tokenizer for {model_id}: {e}")
            print("Using fallback tokenizer")
            self.tokenizer = AutoTokenizer.from_pretrained("OpenGVLab/InternVL2_5-2B", trust_remote_code=True)
        
        self.vision = _find_vision(self.model)
        self.projector = _find_projector(self.model)
        
        # Manually disable flash attention in vision model (critical!)
        if hasattr(self.vision, 'encoder'):
            for layer in self.vision.encoder.layers:
                if hasattr(layer, 'attn') and hasattr(layer.attn, 'use_flash_attn'):
                    layer.attn.use_flash_attn = False
        
        print(f"Model loaded on {self.model.device}")
        print("  Flash attention disabled in vision model")

    @torch.no_grad()
    def encode_grid(self, image):
        """
        Extract grid features from InternViT vision encoder.
        
        NOTE: InternVL vision has flash attention issues. We hook vision output
        during extract_feature() call instead of calling vision model directly.
        
        Args:
            image: PIL Image
            
        Returns:
            dict: {'grid': (1, Hf, Wf, C), 'meta': {'Hf': Hf, 'Wf': Wf, 'stride': stride}}
        """
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize((448, 448), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        pixel_values = transform(image).unsqueeze(0).to(self.model.device, dtype=torch.bfloat16)
        
        # Hook vision model output (before projector)
        vision_feat = {}
        def hook_vision(mod, inp, out):
            if hasattr(out, 'last_hidden_state'):
                vision_feat['feat'] = out.last_hidden_state.detach()
            else:
                vision_feat['feat'] = out.detach()
        
        # Find vision_model to hook
        vision_model = None
        for name, mod in self.model.named_modules():
            if 'vision_model' in name and len(name.split('.')) == 1:  # Top-level vision_model
                vision_model = mod
                break
        
        if vision_model is None:
            raise RuntimeError("Cannot find vision_model in InternVL 2.5")
        
        handle = vision_model.register_forward_hook(hook_vision)
        
        try:
            # Call extract_feature() which will trigger vision_model forward
            _ = self.model.extract_feature(pixel_values)
        finally:
            handle.remove()
        
        if 'feat' not in vision_feat:
            raise RuntimeError("Failed to capture vision features")
        
        hs = vision_feat['feat']
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
        # InternVL 2.5 has extract_feature() method (like InternVL 3.5)
        # This handles the full pipeline: vision encoder → projector
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize((448, 448), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        pixel_values = transform(image).unsqueeze(0).to(self.model.device, dtype=torch.bfloat16)
        
        # Use extract_feature if available (handles internal projector pipeline)
        if hasattr(self.model, 'extract_feature'):
            tokens = self.model.extract_feature(pixel_values)
        else:
            # InternVL 2.5 may not have extract_feature, need to hook during full forward
            print("Warning: extract_feature() not found, using full model forward")
            # Prepare inputs
            messages = [{'role': 'user', 'content': '<image>\nDescribe.'}]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(text, return_tensors='pt').to(self.model.device)
            
            # Hook projector output
            proj_output = {}
            def hook_proj(mod, inp, out):
                proj_output['tokens'] = out.detach()
            
            handle = self.projector.register_forward_hook(hook_proj)
            
            try:
                _ = self.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs.get('attention_mask'),
                    pixel_values=pixel_values,
                    return_dict=True
                )
            finally:
                handle.remove()
            
            if 'tokens' not in proj_output:
                raise RuntimeError("Failed to capture projector output")
            
            tokens = proj_output['tokens']
        
        # Infer grid dimensions
        B, N, D = tokens.shape
        import math
        sqrt_n = int(math.isqrt(N))
        if sqrt_n * sqrt_n == N:
            Hf = Wf = sqrt_n
        else:
            Hf = Wf = int(math.sqrt(N))
        
        return {
            "tokens": tokens, 
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
                print(f"  Warning: InternVL 2.5 LLM forward failed ({e}), falling back to projector")
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


