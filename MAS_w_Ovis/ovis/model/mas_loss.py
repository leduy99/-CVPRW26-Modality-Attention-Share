"""
MAS (Modality Attention Share) Loss Implementation
Measuring how much the model "looks at the image"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from typing import Tuple, List, Optional, Dict, Any, Callable


def extract_counting_number(text: str) -> Optional[int]:
    """
    Extract the counting number from model output text
    
    Args:
        text: Model generated text (e.g., "There are 5 objects in the image.")
    
    Returns:
        Extracted number or None if not found
    """
    if not isinstance(text, str):
        return None
    
    # Look for numbers in the text (prioritize last number as it's usually the answer)
    numbers = re.findall(r'\d+', text)
    if numbers:
        return int(numbers[-1])  # Return the last number found
    return None


def validate_counting_prediction(pred_text: str, gt_text: str) -> bool:
    """
    Validate counting prediction with stricter criteria
    
    Args:
        pred_text: Predicted text from model
        gt_text: Ground truth text
    
    Returns:
        True if prediction is correct, False otherwise
    """
    pred_num = extract_counting_number(pred_text)
    gt_num = extract_counting_number(gt_text)
    
    if pred_num is None or gt_num is None:
        return False  # If we can't extract numbers, consider it wrong
    
    # Basic number match
    numbers_match = (pred_num == gt_num)
    
    # Additional quality checks for prediction text
    pred_clean = pred_text.lower().strip()
    
    # Check if prediction text is well-formed
    quality_checks = [
        len(pred_clean) > 10,  # Not too short
        not pred_clean.startswith("'ll"),  # Not truncated
        "coffee coffee" not in pred_clean,  # No repeated words
        "bowls bowls" not in pred_clean,
        "beads beads" not in pred_clean,
        "they:" not in pred_clean,  # No malformed endings
        "the:" not in pred_clean,
        not any(char * 3 in pred_clean for char in "abcdefghijklmnopqrstuvwxyz"),  # No triple letters
    ]
    
    text_quality_good = sum(quality_checks) >= len(quality_checks) * 0.7  # At least 70% pass
    
    # Only consider correct if both number matches AND text quality is good
    return numbers_match and text_quality_good


def validate_token_level_prediction(pred_tokens: torch.Tensor, gt_tokens: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """
    Validate token-level predictions (for non-counting tasks)
    
    Args:
        pred_tokens: Predicted tokens [batch, seq_len]
        gt_tokens: Ground truth tokens [batch, seq_len]
        valid_mask: Valid positions mask [batch, seq_len]
    
    Returns:
        Batch-level correctness [batch] (True if all valid tokens correct)
    """
    if not valid_mask.any():
        return torch.ones(pred_tokens.size(0), dtype=torch.bool, device=pred_tokens.device)
    
    # Check if predictions match ground truth at valid positions
    correct_mask = (pred_tokens == gt_tokens) | (~valid_mask)  # Correct or ignored
    batch_correct = correct_mask.all(dim=1)  # All tokens correct per batch
    
    return batch_correct


class MASLoss(nn.Module):
    """
    MAS (Modality Attention Share) Loss
    
    Measures the fraction of attention that lands on vision tokens and applies
    a constraint to ensure the model looks at images appropriately.
    """
    
    def __init__(
        self, 
        tau_mas: float = 0.5,  # Default neutral threshold for text+vision
        alpha: float = 1.0,    # Loss weight
        apply_to_layers: str = "all",  # "all", "last", or list of layer indices
        reduction: str = "mean",
        validation_fn: Optional[Callable] = None,  # Task-specific validation function
        task_type: str = "counting"  # Task type: "counting", "token_level", etc.
    ):
        super().__init__()
        self.tau_mas = tau_mas
        self.alpha = alpha
        self.apply_to_layers = apply_to_layers
        self.reduction = reduction
        self.task_type = task_type
        
        # Set validation function based on task type
        if validation_fn is not None:
            self.validation_fn = validation_fn
        elif task_type == "counting":
            self.validation_fn = validate_counting_prediction
        else:
            self.validation_fn = None  # Use token-level validation
    
    def compute_mas_per_layer(
        self, 
        attention_weights: torch.Tensor,  # [batch, heads, seq_len, seq_len]
        vision_positions: List[int],      # Positions of vision tokens
        target_steps: List[int]           # Steps to measure (e.g., generation steps)
    ) -> torch.Tensor:
        """
        Compute MAS for a single layer
        
        Args:
            attention_weights: Attention matrix [batch, heads, seq_len, seq_len]
            vision_positions: List of vision token positions
            target_steps: List of decoding steps to measure
            
        Returns:
            mas_score: MAS score for this layer [batch]
        """
        batch_size, num_heads, seq_len, _ = attention_weights.shape
        
        # Convert positions to tensors
        vision_mask = torch.zeros(seq_len, device=attention_weights.device, dtype=torch.bool)
        if vision_positions:
            vision_indices = torch.tensor(vision_positions, device=attention_weights.device)
            # Ensure indices are within bounds
            vision_indices = vision_indices[vision_indices < seq_len]
            if len(vision_indices) > 0:
                vision_mask[vision_indices] = True
        
        target_mask = torch.zeros(seq_len, device=attention_weights.device, dtype=torch.bool)
        if target_steps:
            target_indices = torch.tensor(target_steps, device=attention_weights.device)
            # Ensure indices are within bounds
            target_indices = target_indices[target_indices < seq_len]
            if len(target_indices) > 0:
                target_mask[target_indices] = True
        
        if not vision_mask.any() or not target_mask.any():
            # No vision tokens or target steps, return zero MAS
            return torch.zeros(batch_size, device=attention_weights.device)
        
        # Extract attention from target steps to all positions
        # attention_weights[batch, heads, query, key]
        target_attention = attention_weights[:, :, target_mask, :]  # [batch, heads, target_steps, seq_len]
        
        # Sum attention to vision tokens
        vision_attention = target_attention[:, :, :, vision_mask].sum(dim=-1)  # [batch, heads, target_steps]
        
        # Sum total attention (should be 1.0 per step, but let's be safe)
        total_attention = target_attention.sum(dim=-1)  # [batch, heads, target_steps]
        
        # Compute MAS per head per step
        mas_per_head_step = vision_attention / (total_attention + 1e-8)  # [batch, heads, target_steps]
        
        # Average across heads and steps
        mas_score = mas_per_head_step.mean(dim=[1, 2])  # [batch]
        
        return mas_score
    
    def compute_mas_total(
        self,
        attention_weights: Tuple[torch.Tensor, ...],  # Tuple of attention from all layers
        token_info: Dict[str, Any],                   # Token position information
        generated_steps: Optional[List[int]] = None   # Which steps to measure
    ) -> torch.Tensor:
        """
        Compute total MAS across specified layers
        
        Args:
            attention_weights: Tuple of attention tensors from all layers
            token_info: Dictionary containing token position information
            generated_steps: Steps to measure (if None, use all generated tokens)
            
        Returns:
            total_mas: Average MAS across layers [batch]
        """
        if not attention_weights or len(attention_weights) == 0:
            return torch.tensor(0.0, device=next(iter(token_info.values()))[0].device if token_info else torch.device('cpu'))
        
        # Get vision token positions
        vision_positions = token_info.get('vision_positions', [])
        if isinstance(vision_positions, torch.Tensor):
            vision_positions = vision_positions.tolist()
        
        # Determine which steps to measure
        if generated_steps is None:
            # Use all positions after the initial sequence
            total_tokens = token_info.get('total_tokens', 0)
            if isinstance(total_tokens, torch.Tensor):
                total_tokens = total_tokens.item()
            
            # Assume generated tokens are at the end
            initial_length = total_tokens - len(attention_weights)  # Rough estimate
            generated_steps = list(range(max(0, initial_length), total_tokens))
        
        # Determine which layers to apply MAS to
        if self.apply_to_layers == "all":
            layer_indices = list(range(len(attention_weights)))
        elif self.apply_to_layers == "last":
            layer_indices = [len(attention_weights) - 1]
        elif isinstance(self.apply_to_layers, (list, tuple)):
            layer_indices = [i for i in self.apply_to_layers if 0 <= i < len(attention_weights)]
        else:
            layer_indices = list(range(len(attention_weights)))
        
        if not layer_indices:
            return torch.tensor(0.0, device=attention_weights[0].device)
        
        # Compute MAS for each selected layer
        mas_scores = []
        for layer_idx in layer_indices:
            if layer_idx < len(attention_weights):
                layer_attention = attention_weights[layer_idx]
                if isinstance(layer_attention, (tuple, list)) and len(layer_attention) > 0:
                    # Handle case where attention is tuple of head attentions
                    layer_attention = layer_attention[0] if len(layer_attention) == 1 else torch.stack(layer_attention, dim=1)
                
                if isinstance(layer_attention, torch.Tensor) and layer_attention.dim() >= 3:
                    mas_score = self.compute_mas_per_layer(layer_attention, vision_positions, generated_steps)
                    mas_scores.append(mas_score)
        
        if not mas_scores:
            return torch.tensor(0.0, device=attention_weights[0].device)
        
        # Average MAS across selected layers
        total_mas = torch.stack(mas_scores).mean(dim=0)  # [batch]
        
        return total_mas
    
    def forward(
        self,
        attention_weights: Tuple[torch.Tensor, ...],
        token_info: Dict[str, Any],
        generated_steps: Optional[List[int]] = None,
        predictions: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        pred_texts: Optional[List[str]] = None,
        gt_texts: Optional[List[str]] = None,
        tokenizer = None
    ) -> torch.Tensor:
        """
        Compute MAS loss - only penalize when predictions are wrong
        
        Args:
            attention_weights: Attention weights from all layers
            token_info: Token position information
            generated_steps: Steps to measure
            predictions: Model predictions [batch, seq_len] or [batch, seq_len, vocab_size]
            labels: Ground truth labels [batch, seq_len]
            pred_texts: Predicted text strings (for counting tasks)
            gt_texts: Ground truth text strings (for counting tasks)
            tokenizer: Tokenizer to decode predictions/labels if needed
            
        Returns:
            loss: MAS constraint loss (only applied to wrong predictions)
        """
        # Compute MAS score
        mas_score = self.compute_mas_total(attention_weights, token_info, generated_steps)
        
        # Apply constraint: L_mas = [τ_mas - MAS]_+
        # If MAS < τ_mas → loss > 0 → push model to look at vision more
        # If MAS ≥ τ_mas → loss = 0 → don't over-force attention to vision
        constraint_loss = torch.clamp(self.tau_mas - mas_score, min=0.0)
        
        # Determine which predictions are wrong based on task type
        if self.task_type == "counting" and self.validation_fn is not None:
            # For counting tasks, use text-based validation
            batch_correct = self._validate_counting_batch(
                predictions, labels, pred_texts, gt_texts, tokenizer
            )
        else:
            # For token-level tasks, use token-based validation
            batch_correct = self._validate_token_batch(predictions, labels)
        
        # Only apply MAS loss to wrong predictions
        if batch_correct is not None:
            batch_wrong = ~batch_correct  # Invert: True where predictions are wrong
            constraint_loss = constraint_loss * batch_wrong.float()
        else:
            # No batch validation performed, use raw constraint loss
            pass
        
        # Apply reduction
        if self.reduction == "mean":
            loss = constraint_loss.mean()
        elif self.reduction == "sum":
            loss = constraint_loss.sum()
        else:
            loss = constraint_loss
        
        # Apply loss weight
        return self.alpha * loss
    
    def _validate_counting_batch(
        self, 
        predictions: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        pred_texts: Optional[List[str]],
        gt_texts: Optional[List[str]],
        tokenizer = None
    ) -> Optional[torch.Tensor]:
        """Validate batch for counting tasks"""
        
        # If we have text strings directly, use them
        if pred_texts is not None and gt_texts is not None:
            batch_size = len(pred_texts)
            batch_correct = torch.zeros(batch_size, dtype=torch.bool)
            
            for i in range(batch_size):
                try:
                    batch_correct[i] = self.validation_fn(pred_texts[i], gt_texts[i])
                except Exception:
                    batch_correct[i] = False  # Consider failed validation as wrong
            
            return batch_correct
        
        # If we have tokens and tokenizer, decode them
        elif predictions is not None and labels is not None and tokenizer is not None:
            batch_size = predictions.size(0)
            batch_correct = torch.zeros(batch_size, dtype=torch.bool, device=predictions.device)
            
            # Get predicted tokens
            if predictions.dim() == 3:  # [batch, seq_len, vocab_size]
                pred_tokens = predictions.argmax(dim=-1)
            else:  # [batch, seq_len]
                pred_tokens = predictions
            
            # Decode predictions and labels to text
            for i in range(batch_size):
                try:
                    # Get valid positions (where labels != -100)
                    valid_mask = (labels[i] != -100)
                    if not valid_mask.any():
                        batch_correct[i] = True  # No valid tokens to check
                        continue
                    
                    # Extract valid tokens
                    pred_valid = pred_tokens[i][valid_mask]
                    label_valid = labels[i][valid_mask]
                    
                    
                    # Decode to text
                    pred_text = tokenizer.decode(pred_valid, skip_special_tokens=True)
                    gt_text = tokenizer.decode(label_valid, skip_special_tokens=True)
                    
                    # Validate using counting function
                    is_correct = self.validation_fn(pred_text, gt_text)
                    batch_correct[i] = is_correct
                    
                    # Debug number extraction
                    pred_num = extract_counting_number(pred_text)
                    gt_num = extract_counting_number(gt_text)
                    
                    
                except Exception:
                    batch_correct[i] = False  # Consider failed validation as wrong
            
            return batch_correct
        
        # If we can't validate, return None (no filtering)
        return None
    
    def _validate_token_batch(
        self, 
        predictions: Optional[torch.Tensor], 
        labels: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """Validate batch for token-level tasks"""
        
        if predictions is None or labels is None:
            return None
        
        # Get predicted tokens
        if predictions.dim() == 3:  # [batch, seq_len, vocab_size]
            pred_tokens = predictions.argmax(dim=-1)
        else:  # [batch, seq_len]
            pred_tokens = predictions
        
        # Check valid positions
        valid_mask = (labels != -100)
        
        return validate_token_level_prediction(pred_tokens, labels, valid_mask)
    
    def get_mas_score(
        self,
        attention_weights: Tuple[torch.Tensor, ...],
        token_info: Dict[str, Any],
        generated_steps: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Get MAS score without computing loss (for monitoring)
        """
        return self.compute_mas_total(attention_weights, token_info, generated_steps)


def create_mas_loss(
    tau_mas: float = 0.5,
    alpha: float = 1.0,
    apply_to_layers: str = "all",
    reduction: str = "mean",
    validation_fn: Optional[Callable] = None,
    task_type: str = "counting"
) -> MASLoss:
    """
    Factory function to create MAS loss
    
    Args:
        tau_mas: Neutral threshold (0.5 for text+vision, 1/#modalities in general)
        alpha: Loss weight
        apply_to_layers: Which layers to apply MAS to ("all", "last", or list)
        reduction: How to reduce batch dimension ("mean", "sum", "none")
        validation_fn: Custom validation function for task-specific checking
        task_type: Task type ("counting", "token_level", etc.)
    
    Returns:
        MAS loss instance
    """
    return MASLoss(
        tau_mas=tau_mas,
        alpha=alpha,
        apply_to_layers=apply_to_layers,
        reduction=reduction,
        validation_fn=validation_fn,
        task_type=task_type
    )


# Example usage and testing
if __name__ == "__main__":
    # Test MAS loss
    batch_size = 2
    num_heads = 16
    seq_len = 100
    num_layers = 24
    
    # Create dummy attention weights
    attention_weights = []
    for _ in range(num_layers):
        # Each layer: [batch, heads, seq_len, seq_len]
        attn = torch.softmax(torch.randn(batch_size, num_heads, seq_len, seq_len), dim=-1)
        attention_weights.append(attn)
    attention_weights = tuple(attention_weights)
    
    # Create dummy token info
    token_info = {
        'vision_positions': list(range(10, 50)),  # Vision tokens at positions 10-49
        'total_tokens': seq_len,
        'system_instruction_positions': list(range(50, 80)),
    }
    
    # Generated steps (last 20 tokens)
    generated_steps = list(range(80, 100))
    
    # Create MAS loss
    mas_loss_fn = create_mas_loss(tau_mas=0.3, alpha=1.0)
    
    # Compute loss
    loss = mas_loss_fn(attention_weights, token_info, generated_steps)
    mas_score = mas_loss_fn.get_mas_score(attention_weights, token_info, generated_steps)
    
    print(f"MAS Score: {mas_score}")
    print(f"MAS Loss: {loss}")
