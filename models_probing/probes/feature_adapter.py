"""
Lightweight MLP adapter to refine raw VLM features.

This is needed to make fair comparisons between different feature extraction levels:
- Projector outputs are already refined with MLP + normalization
- Raw encoder/LLM outputs need similar refinement for fair comparison
"""
import torch
import torch.nn as nn


class FeatureAdapter(nn.Module):
    """
    Lightweight MLP to refine raw features before detection head.
    
    Architecture:
    - RMSNorm (like Qwen uses internally)
    - Linear projection (optional dimension change)
    - SiLU activation
    - Linear projection
    - Residual connection (if dim unchanged)
    
    This mimics the refinement that Qwen's merger/projector does internally.
    """
    
    def __init__(self, in_dim: int, out_dim: int = None, hidden_mult: float = 2.0):
        super().__init__()
        if out_dim is None:
            out_dim = in_dim
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        hidden_dim = int(in_dim * hidden_mult)
        
        # RMSNorm (like Qwen uses)
        self.norm = RMSNorm(in_dim)
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim, bias=False)
        )
        
        # Residual projection if dimensions change
        self.use_residual = (in_dim == out_dim)
        if not self.use_residual and in_dim != out_dim:
            self.proj_residual = nn.Linear(in_dim, out_dim, bias=False)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights similar to transformer layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) feature map
        
        Returns:
            (B, C_out, H, W) refined feature map
        """
        B, C, H, W = x.shape
        
        # Reshape to (B, H, W, C) for normalization and MLP
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        x_flat = x.view(B * H * W, C)  # (B*H*W, C)
        
        # Apply normalization + MLP
        identity = x_flat
        x_normed = self.norm(x_flat)
        x_out = self.mlp(x_normed)
        
        # Residual connection
        if self.use_residual:
            x_out = x_out + identity
        elif hasattr(self, 'proj_residual'):
            x_out = x_out + self.proj_residual(identity)
        
        # Reshape back to (B, C_out, H, W)
        x_out = x_out.view(B, H, W, self.out_dim)
        x_out = x_out.permute(0, 3, 1, 2).contiguous()
        
        return x_out


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (used in Qwen/LLaMA)."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., dim) any shape ending in dim
        
        Returns:
            (..., dim) normalized tensor
        """
        # RMS normalization
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_normed = x / rms
        return x_normed * self.weight


