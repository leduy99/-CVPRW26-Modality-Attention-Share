"""
Bottleneck layers for handling large channel dimensions in VLM features.
"""
import torch
import torch.nn as nn


class GroupedPointwiseBottleneck(nn.Module):
    """
    Efficient bottleneck using grouped 1x1 convolutions to reduce channel dimensions.
    
    Args:
        in_ch: Input channels
        mid_per_group: Channels per group in intermediate layer
        groups: Number of groups
        out_ch: Output channels
    """
    
    def __init__(self, in_ch: int, mid_per_group: int = 16, groups: int = 64, out_ch: int = 256):
        super().__init__()
        self.groups = groups
        mid_ch = mid_per_group * groups
        
        # Grouped 1x1 conv to reduce channels efficiently
        self.reduce = nn.Conv2d(in_ch, mid_ch, 1, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch, eps=1e-3, momentum=0.03)
        self.act1 = nn.SiLU(inplace=True)
        
        # Final 1x1 conv to target output channels
        self.expand = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.03)
        self.act2 = nn.SiLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure dtype consistency
        if x.dtype != self.reduce.conv.weight.dtype:
            x = x.to(self.reduce.conv.weight.dtype)
            
        x = self.reduce(x)
        x = self.bn1(x)
        x = self.act1(x)
        
        x = self.expand(x)
        x = self.bn2(x)
        x = self.act2(x)
        
        return x


class SimpleBottleneck(nn.Module):
    """
    Simple bottleneck using standard 1x1 convolutions.
    """
    
    def __init__(self, in_ch: int, out_ch: int = 256):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.GroupNorm(min(32, out_ch), out_ch, eps=1e-3)
        self.act = nn.SiLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure dtype consistency
        if x.dtype != self.conv.weight.dtype:
            x = x.to(self.conv.weight.dtype)
        return self.act(self.bn(self.conv(x)))
