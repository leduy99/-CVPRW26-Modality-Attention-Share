"""
Clean Attention Management System for Ovis
Separates attention capture, token analysis, and loss computation
"""

import torch
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class TokenInfo:
    """Clean token information structure"""
    vision_positions: List[int]
    system_instruction_positions: List[int] 
    generated_positions: List[int]
    total_tokens: int
    
    @property
    def vision_count(self) -> int:
        return len(self.vision_positions)
    
    @property 
    def system_count(self) -> int:
        return len(self.system_instruction_positions)
    
    @property
    def generated_count(self) -> int:
        return len(self.generated_positions)


@dataclass
class AttentionData:
    """Clean attention data structure"""
    weights: Tuple[torch.Tensor, ...]  # Per-layer attention weights
    token_info: TokenInfo
    sequence_length: int
    
    def __post_init__(self):
        if self.weights and len(self.weights) > 0:
            self.num_layers = len(self.weights)
            self.batch_size = self.weights[0].size(0) if self.weights[0].dim() >= 2 else 1
        else:
            self.num_layers = 0
            self.batch_size = 0


class AttentionAnalyzer(ABC):
    """Abstract base for attention analysis"""
    
    @abstractmethod
    def analyze(self, attention_data: AttentionData) -> Dict[str, torch.Tensor]:
        """Analyze attention and return metrics"""
        pass


class TokenAnalyzer:
    """Clean token position analysis"""
    
    def __init__(self):
        self.VISUAL_ATOM_ID = -1
        self.IM_START_ID = 151644
        self.IM_END_ID = 151645
    
    def analyze_token_positions(
        self, 
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        original_input_length: Optional[int] = None
    ) -> TokenInfo:
        """
        Clean token position analysis
        
        Args:
            input_ids: Input token IDs
            labels: Labels for identifying generated tokens
            original_input_length: Length of original input before processing
            
        Returns:
            TokenInfo with all token positions
        """
        seq_len = input_ids.size(1)
        input_ids_flat = input_ids[0]  # Assume batch_size=1 for simplicity
        
        # Find vision token positions
        vision_positions = torch.where(input_ids_flat == self.VISUAL_ATOM_ID)[0].tolist()
        
        # Find indicator positions
        im_start_positions = torch.where(input_ids_flat == self.IM_START_ID)[0].tolist()
        im_end_positions = torch.where(input_ids_flat == self.IM_END_ID)[0].tolist()
        
        # Determine system instruction positions
        if im_end_positions:
            im_end_pos = im_end_positions[0]
            # Everything after <|im_end|> until generation starts
            system_start = im_end_pos + 1
        else:
            system_start = len(vision_positions) + len(im_start_positions)
        
        # Find generated positions from labels
        generated_positions = []
        if labels is not None:
            generated_mask = (labels[0] != -100)
            generated_positions = torch.where(generated_mask)[0].tolist()
        
        # System instruction positions: between system_start and first generated token
        if generated_positions:
            system_end = min(generated_positions) 
        else:
            system_end = seq_len
            
        system_instruction_positions = list(range(system_start, system_end))
        
        return TokenInfo(
            vision_positions=vision_positions,
            system_instruction_positions=system_instruction_positions,
            generated_positions=generated_positions,
            total_tokens=seq_len
        )


class AttentionCapture:
    """Clean attention weight capture"""
    
    def __init__(self):
        self.token_analyzer = TokenAnalyzer()
    
    def capture_attention_data(
        self,
        llm_output: Any,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        original_input_length: Optional[int] = None
    ) -> Optional[AttentionData]:
        """
        Capture attention data cleanly
        
        Args:
            llm_output: Model output with attention weights
            input_ids: Input token IDs
            labels: Training labels
            original_input_length: Original input length
            
        Returns:
            AttentionData or None if no attention available
        """
        if not hasattr(llm_output, 'attentions') or llm_output.attentions is None:
            return None
        
        # Get token info
        token_info = self.token_analyzer.analyze_token_positions(
            input_ids, labels, original_input_length
        )
        
        # Create attention data
        attention_data = AttentionData(
            weights=llm_output.attentions,
            token_info=token_info,
            sequence_length=input_ids.size(1)
        )
        
        return attention_data


class MASAnalyzer(AttentionAnalyzer):
    """MAS (Modality Attention Share) analyzer"""
    
    def __init__(self, tau_mas: float = 0.5, apply_layers: str = "all"):
        self.tau_mas = tau_mas
        self.apply_layers = apply_layers
    
    def analyze(self, attention_data: AttentionData) -> Dict[str, torch.Tensor]:
        """
        Compute MAS metrics
        
        Returns:
            Dict with 'mas_score' and 'mas_loss'
        """
        if not attention_data.weights or attention_data.num_layers == 0:
            return {
                'mas_score': torch.tensor(0.0),
                'mas_loss': torch.tensor(0.0)
            }
        
        # Determine which layers to analyze
        if self.apply_layers == "all":
            layer_indices = list(range(attention_data.num_layers))
        elif self.apply_layers == "last":
            layer_indices = [attention_data.num_layers - 1]
        else:
            layer_indices = list(range(attention_data.num_layers))
        
        # Compute MAS per layer
        layer_mas_scores = []
        
        for layer_idx in layer_indices:
            if layer_idx < len(attention_data.weights):
                layer_attention = attention_data.weights[layer_idx]
                mas_score = self._compute_layer_mas(
                    layer_attention,
                    attention_data.token_info
                )
                layer_mas_scores.append(mas_score)
        
        if not layer_mas_scores:
            return {
                'mas_score': torch.tensor(0.0),
                'mas_loss': torch.tensor(0.0)
            }
        
        # Average across layers
        avg_mas_score = torch.stack(layer_mas_scores).mean(dim=0)
        
        # Compute constraint loss
        mas_loss = torch.clamp(self.tau_mas - avg_mas_score, min=0.0).mean()
        
        return {
            'mas_score': avg_mas_score,
            'mas_loss': mas_loss
        }
    
    def _compute_layer_mas(
        self, 
        layer_attention: torch.Tensor, 
        token_info: TokenInfo
    ) -> torch.Tensor:
        """Compute MAS for a single layer"""
        
        # Handle different attention tensor formats
        if isinstance(layer_attention, (tuple, list)):
            layer_attention = layer_attention[0] if len(layer_attention) > 0 else None
        
        if layer_attention is None or layer_attention.dim() < 3:
            return torch.tensor(0.0, device=layer_attention.device if layer_attention is not None else 'cpu')
        
        batch_size, num_heads, seq_len, _ = layer_attention.shape
        device = layer_attention.device
        
        # Create masks
        vision_mask = torch.zeros(seq_len, device=device, dtype=torch.bool)
        if token_info.vision_positions:
            vision_indices = torch.tensor(token_info.vision_positions, device=device)
            vision_indices = vision_indices[vision_indices < seq_len]
            if len(vision_indices) > 0:
                vision_mask[vision_indices] = True
        
        generated_mask = torch.zeros(seq_len, device=device, dtype=torch.bool)
        if token_info.generated_positions:
            generated_indices = torch.tensor(token_info.generated_positions, device=device)
            generated_indices = generated_indices[generated_indices < seq_len]
            if len(generated_indices) > 0:
                generated_mask[generated_indices] = True
        
        if not vision_mask.any() or not generated_mask.any():
            return torch.tensor(0.0, device=device)
        
        # Extract attention from generated tokens to all tokens
        generated_attention = layer_attention[:, :, generated_mask, :]  # [batch, heads, gen_tokens, seq_len]
        
        # Sum attention to vision tokens
        vision_attention = generated_attention[:, :, :, vision_mask].sum(dim=-1)  # [batch, heads, gen_tokens]
        
        # Sum total attention (should be 1.0, but be safe)
        total_attention = generated_attention.sum(dim=-1)  # [batch, heads, gen_tokens]
        
        # Compute MAS
        mas_per_head_token = vision_attention / (total_attention + 1e-8)
        
        # Average across heads and generated tokens
        mas_score = mas_per_head_token.mean(dim=[1, 2])  # [batch]
        
        return mas_score


class AttentionManager:
    """Main attention management system"""
    
    def __init__(self):
        self.capture = AttentionCapture()
        self.analyzers: Dict[str, AttentionAnalyzer] = {}
    
    def register_analyzer(self, name: str, analyzer: AttentionAnalyzer):
        """Register an attention analyzer"""
        self.analyzers[name] = analyzer
    
    def analyze_attention(
        self,
        llm_output: Any,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        original_input_length: Optional[int] = None,
        analyzer_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Analyze attention with registered analyzers
        
        Returns:
            Dict mapping analyzer names to their results
        """
        # Capture attention data
        attention_data = self.capture.capture_attention_data(
            llm_output, input_ids, labels, original_input_length
        )
        
        if attention_data is None:
            return {}
        
        # Run analyzers
        results = {}
        analyzers_to_run = analyzer_names or list(self.analyzers.keys())
        
        for analyzer_name in analyzers_to_run:
            if analyzer_name in self.analyzers:
                try:
                    results[analyzer_name] = self.analyzers[analyzer_name].analyze(attention_data)
                except Exception as e:
                    print(f"Warning: {analyzer_name} analysis failed: {e}")
                    results[analyzer_name] = {}
        
        return results
