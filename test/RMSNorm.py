from typing import Tuple
import torch
import torch.nn as nn
from AbsmeanQuantization import AbsmeanQuantization

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, hidden_states_scale: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states_float = hidden_states.float() * hidden_states_scale
        hidden_states_float = hidden_states_float * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states_float = self.weight * hidden_states_float
        hidden_states, new_scale = AbsmeanQuantization.quantize(hidden_states_float)
        return hidden_states.to(input_dtype), new_scale