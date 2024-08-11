import torch
import torch.nn as nn
from AbsmeanQuantization import AbsmeanQuantization

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.quantized_weight = None
        self.weight_scale = None

    def quantize(self):
        self.quantized_weight, self.weight_scale = AbsmeanQuantization.quantize(self.weight.data)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        
        if self.quantized_weight is not None:
            weight = self.quantized_weight.float() * self.weight_scale
        else:
            weight = self.weight
        
        return (weight * hidden_states).to(input_dtype)
    