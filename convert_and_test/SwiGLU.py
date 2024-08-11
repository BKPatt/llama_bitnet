import torch
import torch.nn.functional as F
import torch.nn as nn

class SwiGLU(nn.Module):
    def forward(self, gate_output: torch.Tensor, up_output: torch.Tensor) -> torch.Tensor:
        # Dequantize inputs
        gate_float = gate_output.float() / 128.0
        up_float = up_output.float() / 128.0
        
        # Compute SiLU (Swish) activation
        silu_gate = gate_float * torch.sigmoid(gate_float)
        
        # Multiply with up_output
        result = silu_gate * up_float
        
        # Quantize back to ternary
        return torch.clamp(result * 128.0, -128, 127).to(torch.int8)