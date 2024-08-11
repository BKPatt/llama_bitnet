import torch
import torch.nn as nn
from BitNetB158Config import BitNetB158Config
from QuantizedLinear import QuantizedLinear
from SwiGLU import SwiGLU

class BitNetMLP(nn.Module):
    def __init__(self, config: BitNetB158Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = QuantizedLinear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = QuantizedLinear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = QuantizedLinear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = SwiGLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_output = self.gate_proj(x)
        up_output = self.up_proj(x)
        
        intermediate_output = self.act_fn(gate_output, up_output)
        down_output = self.down_proj(intermediate_output)
        
        return down_output
