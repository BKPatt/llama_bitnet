import torch
import torch.nn as nn
from typing import Callable
from BitNetB158Config import BitNetB158Config
from QuantizedLinear import QuantizedLinear

class BitNetMLP(nn.Module):
    def __init__(self, config: BitNetB158Config):
        super().__init__()
        self.gate_proj = QuantizedLinear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = QuantizedLinear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = QuantizedLinear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn: Callable[[torch.Tensor], torch.Tensor] = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))