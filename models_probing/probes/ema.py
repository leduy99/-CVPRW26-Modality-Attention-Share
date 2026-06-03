import torch
import torch.nn as nn
from copy import deepcopy


class EMA:
    """Exponential Moving Average for model weights."""
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        """
        Args:
            model: The model to create EMA for
            decay: EMA decay rate (higher = more stable, slower adaptation)
        """
        self.decay = decay
        self.ema = deepcopy(model).eval()
        
        # Freeze EMA parameters
        for p in self.ema.parameters():
            p.requires_grad_(False)
    
    def update(self, model: nn.Module):
        """Update EMA weights with current model weights."""
        with torch.no_grad():
            for (name, param), (ema_name, ema_param) in zip(
                model.state_dict().items(), 
                self.ema.state_dict().items()
            ):
                if ema_param.dtype.is_floating_point:
                    ema_param.mul_(self.decay).add_(param, alpha=1 - self.decay)
    
    def state_dict(self):
        """Get EMA state dict."""
        return self.ema.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load EMA state dict."""
        self.ema.load_state_dict(state_dict)

