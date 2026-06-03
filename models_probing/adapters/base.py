"""
Base adapter interface for VLM models.
Provides standardized way to extract features from different VLM architectures.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any
import torch


class VLAdapter(ABC):
    """Abstract base class for VLM adapters."""
    
    @abstractmethod
    def encode_grid(self, image) -> Dict[str, Any]:
        """
        Extract grid features from vision encoder.
        
        Returns:
            Dict with keys:
                - 'grid': (1, Hf, Wf, C) tensor from vision encoder
                - 'meta': dict with 'Hf', 'Wf', 'stride' information
        """
        pass

    @abstractmethod
    def project_tokens(self, image) -> Dict[str, Any]:
        """
        Extract tokens after projector/alignment layer.
        
        Returns:
            Dict with keys:
                - 'tokens': (1, N, D) tensor after projector
                - 'meta': dict with 'Hf', 'Wf' information
        """
        pass

    def extract_llm_features(self, image, prompt="Describe the objects in this image.") -> Dict[str, Any]:
        """
        Extract features after LLM processing.
        
        Returns:
            Dict with keys:
                - 'tokens': (1, N, D) tensor after LLM
                - 'meta': dict with information
        """
        # Default implementation - override in subclasses
        raise NotImplementedError("LLM feature extraction not implemented for this adapter")

    def to_feature_map(self, grid: torch.Tensor, input_size_hw=(512, 512)) -> Dict[str, Any]:
        """
        Convert grid tensor to standard feature map format.
        
        Args:
            grid: (1, Hf, Wf, C) tensor
            input_size_hw: (height, width) of input image
            
        Returns:
            Dict with keys:
                - 'feat': (1, C, Hf, Wf) tensor
                - 'stride': float, stride from input resolution to feature resolution
                - 'Hf': int, feature map height
                - 'Wf': int, feature map width
        """
        B, Hf, Wf, C = grid.shape
        feat = grid.permute(0, 3, 1, 2).contiguous()
        
        # Use float stride for precise coordinate mapping
        stride_h = float(input_size_hw[0]) / float(Hf)
        stride_w = float(input_size_hw[1]) / float(Wf)
        assert abs(stride_h - stride_w) < 1e-6, f"Non-square stride! H={stride_h:.3f}, W={stride_w:.3f}"
        
        return {"feat": feat, "stride": stride_h, "Hf": Hf, "Wf": Wf}

