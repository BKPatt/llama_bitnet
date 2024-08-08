import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from AbsmeanQuantization import AbsmeanQuantization

class QuantizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights
        weight = torch.empty(out_features, in_features)
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        
        # Quantize weights
        self.quantized_weight, self.weight_scale = AbsmeanQuantization.quantize(weight)
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            nn.init.uniform_(self.bias, -0.1, 0.1)
        else:
            self.register_parameter('bias', None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Dequantize weights
        dequantized_weight = AbsmeanQuantization.dequantize(self.quantized_weight, self.weight_scale)
        
        # Perform the linear operation
        return F.linear(input, dequantized_weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
