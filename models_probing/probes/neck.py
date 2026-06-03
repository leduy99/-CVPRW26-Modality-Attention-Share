import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.03)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class Neck(nn.Module):
    """
    Light neck to reduce extremely high channel counts from VLM features and
    add nonlinearity. Keeps spatial resolution.

    Structure:
      - 1x1 conv to reduce channels to mid_channels
      - two 3x3 conv blocks
    """

    def __init__(self, in_ch: int, mid_ch: int = 512):
        super().__init__()
        self.reduce = ConvBNAct(in_ch, mid_ch, k=1, s=1, p=0)
        self.conv1 = ConvBNAct(mid_ch, mid_ch, k=3, s=1, p=1)
        self.conv2 = ConvBNAct(mid_ch, mid_ch, k=3, s=1, p=1)
        # Don't force bfloat16 - let it match input dtype for better performance

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure dtype matches module weights to avoid dtype mismatch in conv
        x = x.to(self.reduce.conv.weight.dtype)
        x = self.reduce(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


