import torch
import torch.nn as nn
from BitNetB158Config import BitNetB158Config
from QuantizedLinear import QuantizedLinear
from SwiGLU import SwiGLU
from AbsmeanQuantization import AbsmeanQuantization

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
        # Dequantize input
        x = x.float() / 128.0
        
        gate_output = self.gate_proj(x)
        up_output = self.up_proj(x)

        intermediate_output = self.act_fn(gate_output, up_output)
        
        # Quantize intermediate output to ternary
        intermediate_output = torch.clamp(intermediate_output * 128.0, -128, 127).to(torch.int8)
        
        down_output = self.down_proj(intermediate_output)

        # Quantize output to ternary
        return torch.clamp(down_output * 128.0, -128, 127).to(torch.int8)

    def quantize(self):
        self.gate_proj.quantize()
        self.up_proj.quantize()
        self.down_proj.quantize()