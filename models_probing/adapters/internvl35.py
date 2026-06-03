"""
InternVL 3.5 adapter for feature extraction.
Released Aug 2025 - Latest InternVL version with Cascade RL and ViR.
Uses InternVL3_5-8B model.

Reference: https://huggingface.co/papers/2508.18265
"""
import math
import torch
from transformers import AutoModel, AutoTokenizer
from .base import VLAdapter


class InternVL35Adapter(VLAdapter):
    """Adapter for InternVL 3.5 model (2B variant - smallest in 2-4B range)."""
    
    def __init__(self, model_id="OpenGVLab/InternVL3_5-2B"):
        """
        Initialize InternVL 3.5 adapter.
        
        Args:
            model_id: HuggingFace model identifier (default: 2B model - smallest variant)
        """
        print(f"Loading {model_id}...")
        print("InternVL 3.5 features: Cascade RL, Visual Resolution Router (ViR)")
        
        try:
            self.model = AutoModel.from_pretrained(
                model_id, 
                torch_dtype=torch.bfloat16, 
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                attn_implementation="eager"  # Disable flash attention (causes CUDA errors)
            ).eval()
        except Exception as e:
            print(f"Failed to load {model_id}: {e}")
            raise RuntimeError(f"Could not load InternVL-3.5 model: {e}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        except Exception as e:
            print(f"Failed to load tokenizer for {model_id}: {e}")
            raise RuntimeError(f"Could not load InternVL-3.5 tokenizer: {e}")
        
        # Manually disable flash attention in vision model (critical!)
        if hasattr(self.model, 'vision_model') and hasattr(self.model.vision_model, 'encoder'):
            for layer in self.model.vision_model.encoder.layers:
                if hasattr(layer, 'attn') and hasattr(layer.attn, 'use_flash_attn'):
                    layer.attn.use_flash_attn = False
        
        print(f"Model loaded on {self.model.device}")
        print("  Flash attention disabled in vision model")

    @torch.no_grad()
    def encode_grid(self, image):
        """
        Extract grid features from vision encoder.
        
        Args:
            image: PIL Image
            
        Returns:
            dict: {'grid': (1, Hf, Wf, C), 'meta': {'Hf': Hf, 'Wf': Wf, 'stride': stride}}
        """
        from torchvision import transforms
        
        # InternVL-3.5 uses dynamic resolution with ViR
        # Default to 448x448 for consistency
        transform = transforms.Compose([
            transforms.Resize((448, 448), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        pixel_values = transform(image).unsqueeze(0).to(self.model.device, dtype=torch.bfloat16)
        
        # Extract vision features
        vision_model = None
        for name in ["vision_model", "visual", "vision_tower", "vision_encoder"]:
            if hasattr(self.model, name):
                vision_model = getattr(self.model, name)
                break
        
        if vision_model is None:
            raise RuntimeError("Could not find vision model in InternVL-3.5")
        
        vision_outputs = vision_model(pixel_values, output_hidden_states=True)
        
        # Use last hidden state
        hs = vision_outputs.last_hidden_state  # (1, T, C)
        B, T, C = hs.shape
        
        # Infer grid dimensions
        sqrt_t = int(math.isqrt(T))
        if sqrt_t * sqrt_t == T:
            Hf = Wf = sqrt_t
            had_cls = False
        elif sqrt_t * sqrt_t == T - 1:
            Hf = Wf = sqrt_t
            had_cls = True
            hs = hs[:, 1:, :]  # Remove CLS
        else:
            # Fallback
            Hf = Wf = int(math.sqrt(T))
            had_cls = False
            print(f"Warning: Non-square token count {T}, using Hf={Hf}")
        
        # Reshape to grid
        grid = hs.view(1, Hf, Wf, C).contiguous()
        
        # Calculate stride
        H, W = pixel_values.shape[-2:]
        stride = float(H) / float(Hf)
        
        return {
            "grid": grid, 
            "meta": {"Hf": Hf, "Wf": Wf, "stride": stride}
        }

    @torch.no_grad()
    def project_tokens(self, image):
        """
        Extract tokens after projector/alignment layer.
        
        InternVL 3.5 architecture:
        - Vision encoder: 1024 channels, 1024 tokens (32x32)
        - Projector (mlp1): 4096→4096, outputs 256 tokens (reduced resolution!)
        - Use extract_feature() method which returns post-projector features
        
        Args:
            image: PIL Image
            
        Returns:
            dict: {'tokens': (1, N, D), 'meta': {'Hf': Hf, 'Wf': Wf}}
        """
        # InternVL 3.5 has built-in extract_feature method
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((448, 448), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        pixel_values = transform(image).unsqueeze(0).to(self.model.device, dtype=torch.bfloat16)
        
        # extract_feature returns features AFTER projector (mlp1)
        # Output shape: (1, N_compressed, D) where N_compressed < original tokens
        tokens = self.model.extract_feature(pixel_values)
        
        # InternVL 3.5 compresses spatial resolution via projector
        # Tokens: 256 (16x16) instead of 1024 (32x32)
        B, N, D = tokens.shape
        
        # Infer compressed grid dimensions
        import math
        sqrt_n = int(math.isqrt(N))
        if sqrt_n * sqrt_n == N:
            Hf_compressed = Wf_compressed = sqrt_n
        else:
            # Fallback
            Hf_compressed = Wf_compressed = int(math.sqrt(N))
        
        return {
            "tokens": tokens, 
            "meta": {
                "Hf": Hf_compressed, 
                "Wf": Wf_compressed,
                "compressed": True,  # InternVL 3.5 uses compressed features
                "original_tokens": 1024,  # 32x32
                "compressed_tokens": N  # Typically 256 (16x16)
            }
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
        
        # Find LLM layers - InternVL 3.5 structure
        if hasattr(self.model, 'language_model'):
            if hasattr(self.model.language_model, 'model') and hasattr(self.model.language_model.model, 'layers'):
                llm_layers = self.model.language_model.model.layers
            elif hasattr(self.model.language_model, 'layers'):
                llm_layers = self.model.language_model.layers
            else:
                raise RuntimeError("Cannot find LLM layers in InternVL 3.5 architecture")
        else:
            raise RuntimeError("Cannot find language_model in InternVL 3.5")
        
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
                print(f"  Warning: InternVL 3.5 LLM forward failed ({e}), falling back to projector")
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
