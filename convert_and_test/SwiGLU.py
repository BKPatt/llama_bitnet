
import torch
import torch.nn.functional as F
import torch.nn as nn

class SwiGLU(nn.Module):
    def forward(self, gate_output: torch.Tensor, up_output: torch.Tensor) -> torch.Tensor:
        return F.silu(gate_output) * up_output