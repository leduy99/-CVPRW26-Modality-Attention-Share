"""
Ovis-2.5 adapter with CORRECT architecture-aligned feature extraction.

Ovis Architecture:
    Image
      ↓
    ViT (SigLIP) → 1152 channels                    ← ENCODER
      ↓
    codebook head → 65k logits
      ↓
    softmax + VTE embedding → 2048 channels          ← PROJECTOR  
      ↓
    LLM (2048 input)                                ← LLM
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from .base import VLAdapter


class Ovis25Adapter(VLAdapter):
    """
    Ovis-2.5 adapter following the correct architecture path.
    
    - encode_grid(): ViT encoder output (1152ch)
    - project_tokens(): VTE embedding output (2048ch)
    - extract_llm_features(): LLM layer 0 input (2048ch)
    """
    
    def __init__(self, model_id="AIDC-AI/Ovis2.5-2B"):
        """Initialize Ovis-2.5 adapter."""
        print(f"Loading {model_id}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16, 
            device_map="auto", 
            trust_remote_code=True
        )
        print(f"Model loaded on {self.model.device}")
        
        # Find visual embedding table (vte)
        if hasattr(self.model, 'vte') and hasattr(self.model.vte, 'weight'):
            V, D = self.model.vte.weight.shape
            print(f"  ✓ Found VTE: ({V} vocab × {D} dim)")

    @torch.no_grad()
    def encode_grid(self, image):
        """
        Extract ENCODER features: ViT (SigLIP) output = 1152 channels.
        
        Returns:
            dict: {'grid': (1, Hf, Wf, 1152), 'meta': {...}}
        """
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "."}
        ]}]
        
        input_ids, pixel_values, grid_thws = self.model.preprocess_inputs(
            messages=msgs, add_generation_prompt=True
        )
        
        Hf = int(grid_thws[0, 1].item())
        Wf = int(grid_thws[0, 2].item())
        
        # Hook ViT post_layernorm (last layer of vision encoder)
        encoder_feat = {}
        
        def hook_vit(mod, inp, out):
            encoder_feat['feat'] = out.detach()
        
        handle = None
        for name, mod in self.model.visual_tokenizer.vit.named_modules():
            if name == 'vision_model.post_layernorm':
                handle = mod.register_forward_hook(hook_vit)
                break
        
        # Forward ONLY through ViT (not full model with LLM!)
        # This is 5-10x faster than forwarding through full model
        _ = self.model.visual_tokenizer.vit(
            pixel_values=pixel_values.to(self.model.device),
            grid_thws=grid_thws.to(self.model.device)
        )
        
        if handle:
            handle.remove()
        
        if 'feat' not in encoder_feat:
            raise RuntimeError("Failed to capture ViT encoder features")
        
        vis_feat = encoder_feat['feat']  # (N, 1152) or (B, N, 1152)
        
        if len(vis_feat.shape) == 2:
            vis_feat = vis_feat.unsqueeze(0)
        
        B, N, C = vis_feat.shape
        
        # Reshape to grid
        expected_N = Hf * Wf
        if N != expected_N:
            if N > expected_N:
                vis_feat = vis_feat[:, :expected_N, :]
            else:
                pad = torch.zeros(B, expected_N - N, C,
                                device=vis_feat.device, dtype=vis_feat.dtype)
                vis_feat = torch.cat([vis_feat, pad], dim=1)
        
        grid = vis_feat.view(B, Hf, Wf, C).contiguous()
        stride = 512.0 / float(Hf)
        
        return {'grid': grid, 'meta': {'Hf': Hf, 'Wf': Wf, 'stride': stride}}

    @torch.no_grad()
    def project_tokens(self, image):
        """
        Extract PROJECTOR features: Input to LLM layer 0 = post-VTE embedding = 2048 channels.
        
        This is AFTER visual_tokenizer applies VTE embedding, BEFORE LLM processing.
        
        Returns:
            dict: {'tokens': (1, N, 2048), 'meta': {...}}
        """
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "."}
        ]}]
        
        input_ids, pixel_values, grid_thws = self.model.preprocess_inputs(
            messages=msgs, add_generation_prompt=True
        )
        
        Hf = int(grid_thws[0, 1].item())
        Wf = int(grid_thws[0, 2].item())
        expected_tokens = Hf * Wf
        
        # Hook INPUT to LLM layer 0 (= output after projector/VTE embedding)
        llm_input = {}
        
        def hook_llm_input(mod, inp, out):
            if isinstance(inp, tuple) and len(inp) > 0:
                llm_input['feat'] = inp[0].detach()
        
        handle = None
        for name, mod in self.model.named_modules():
            if name == 'llm.model.layers.0':
                handle = mod.register_forward_hook(hook_llm_input)
                break
        
        # Forward through FULL model (needed to get LLM input)
        _ = self.model(
            input_ids=input_ids.to(self.model.device),
            attention_mask=torch.ones_like(input_ids).to(self.model.device),
            pixel_values=pixel_values.to(self.model.device),
            grid_thws=grid_thws.to(self.model.device),
            return_dict=True
        )
        
        if handle:
            handle.remove()
        
        if 'feat' not in llm_input:
            raise RuntimeError("Failed to capture LLM input (projector output)")
        
        hidden = llm_input['feat']  # (B, seq_len, D)
        B, seq_len, D = hidden.shape
        
        # Extract visual tokens (heuristic: skip first few tokens, take expected_tokens)
        visual_start = min(5, seq_len // 4)
        visual_end = visual_start + expected_tokens
        visual_end = min(visual_end, seq_len - 3)
        
        if visual_end - visual_start >= expected_tokens:
            visual_tokens = hidden[:, visual_start:visual_start + expected_tokens, :]
        else:
            # Fallback: take first N tokens after start
            visual_tokens = hidden[:, visual_start:visual_end, :]
            if visual_tokens.shape[1] < expected_tokens:
                pad = torch.zeros(B, expected_tokens - visual_tokens.shape[1], D,
                                device=visual_tokens.device, dtype=visual_tokens.dtype)
                visual_tokens = torch.cat([visual_tokens, pad], dim=1)
        
        return {'tokens': visual_tokens, 'meta': {'Hf': Hf, 'Wf': Wf}}

    @torch.no_grad()
    def extract_llm_features(self, image, prompt="Describe the objects in this image.", llm_layer=None):
        """
        Extract LLM features: Output from LLM layer N (after contextual processing).
        
        Args:
            llm_layer: Which LLM layer to extract from. 
                      If None, auto-detect and use layer -2 (second to last).
                      Can be negative index (e.g., -2) or positive (e.g., 20).
            
        Returns:
            dict: {'tokens': (1, N, 2048), 'meta': {...}}
        """
        msgs = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
        ]}]
        
        input_ids, pixel_values, grid_thws = self.model.preprocess_inputs(
            messages=msgs, add_generation_prompt=True
        )
        
        Hf = int(grid_thws[0, 1].item())
        Wf = int(grid_thws[0, 2].item())
        expected_tokens = Hf * Wf
        
        # Auto-detect number of LLM layers (only print once)
        if llm_layer is None:
            num_layers = len(self.model.llm.model.layers)
            llm_layer = num_layers - 2  # Second to last layer
            if not hasattr(self, '_llm_layer_printed'):
                print(f"  Auto-detected {num_layers} LLM layers, using layer {llm_layer}")
                self._llm_layer_printed = True
        elif llm_layer < 0:
            # Convert negative index to positive
            num_layers = len(self.model.llm.model.layers)
            llm_layer = num_layers + llm_layer
            if not hasattr(self, '_llm_layer_printed'):
                print(f"  Using LLM layer {llm_layer} (from {num_layers} total layers)")
                self._llm_layer_printed = True
        
        # Hook OUTPUT of LLM layer N (after attention processing)
        llm_output = {}
        
        def hook_llm_output(mod, inp, out):
            # out is the hidden states after this layer
            if isinstance(out, tuple):
                llm_output['feat'] = out[0].detach()
            else:
                llm_output['feat'] = out.detach()
        
        handle = None
        for name, mod in self.model.named_modules():
            if name == f'llm.model.layers.{llm_layer}':
                handle = mod.register_forward_hook(hook_llm_output)
                break
        
        # Forward
        _ = self.model(
            input_ids=input_ids.to(self.model.device),
            attention_mask=torch.ones_like(input_ids).to(self.model.device),
            pixel_values=pixel_values.to(self.model.device),
            grid_thws=grid_thws.to(self.model.device), 
            return_dict=True
        )
        
        if handle:
            handle.remove()
        
        if 'feat' not in llm_output:
            raise RuntimeError(f"Failed to capture LLM layer {llm_layer} output")
        
        hidden = llm_output['feat']  # (B, seq_len, D)
        B, seq_len, D = hidden.shape
        
        # Extract visual tokens (heuristic: skip first few tokens, take expected_tokens)
        visual_start = min(5, seq_len // 4)
        visual_end = visual_start + expected_tokens
        visual_end = min(visual_end, seq_len - 3)
        
        if visual_end - visual_start >= expected_tokens:
            visual_tokens = hidden[:, visual_start:visual_start + expected_tokens, :]
        else:
            # Fallback: take first N tokens after start
            visual_tokens = hidden[:, visual_start:visual_end, :]
            if visual_tokens.shape[1] < expected_tokens:
                pad = torch.zeros(B, expected_tokens - visual_tokens.shape[1], D,
                                device=visual_tokens.device, dtype=visual_tokens.dtype)
                visual_tokens = torch.cat([visual_tokens, pad], dim=1)
        
        return {'tokens': visual_tokens, 'meta': {'Hf': Hf, 'Wf': Wf}}
