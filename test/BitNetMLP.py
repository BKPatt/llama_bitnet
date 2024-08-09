import torch
import torch.nn as nn
from typing import Callable
from BitNetB158Config import BitNetB158Config
from QuantizedLinear import QuantizedLinear
from AbsmeanQuantization import AbsmeanQuantization

class BitNetMLP(nn.Module):
    def __init__(self, config: BitNetB158Config):
        super().__init__()
        self.gate_proj = QuantizedLinear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = QuantizedLinear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = QuantizedLinear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn: Callable[[torch.Tensor], torch.Tensor] = nn.SiLU()

    def forward(self, x: torch.Tensor, x_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate, gate_scale = self.gate_proj(x, x_scale)
        up, up_scale = self.up_proj(x, x_scale)
        
        # Quantized SiLU activation
        gate_activated, gate_activated_scale = AbsmeanQuantization.quantized_act(gate, gate_scale, self.act_fn)
        
        # Quantized multiplication
        intermediate, intermediate_scale = AbsmeanQuantization.quantized_matmul(
            gate_activated, gate_activated_scale, up, up_scale
        )
        
        return self.down_proj(intermediate, intermediate_scale)