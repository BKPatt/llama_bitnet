import torch
import torch.nn as nn
from BitNetLayer import BitNetLayer
from BitNetLinear import BitNetLinear

class BitNetFFN(BitNetLayer):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = BitNetLinear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = BitNetLinear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = BitNetLinear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        gated = gate * self.act_fn(up)
        return self.down_proj(gated)

    def quantize_weights(self):
        self.gate_proj.quantize_weights()
        self.up_proj.quantize_weights()
        self.down_proj.quantize_weights()

    def update_scaling_factors(self):
        self.gate_proj.update_scaling_factors()
        self.up_proj.update_scaling_factors()
        self.down_proj.update_scaling_factors()

    def extra_repr(self) -> str:
        return f'hidden_size={self.hidden_size}, intermediate_size={self.intermediate_size}'