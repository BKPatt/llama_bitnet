import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from AbsmeanQuantization import AbsmeanQuantization

class QuantizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights
        weight = torch.empty(out_features, in_features)
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        
        self.weight = nn.Parameter(weight)
        self.quantized_weight = None
        self.weight_scale = None
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def quantize(self):
        if self.quantized_weight is None:
            self.quantized_weight, self.weight_scale = AbsmeanQuantization.quantize(self.weight)
            self.quantized_weight = AbsmeanQuantization.pack(self.quantized_weight)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.quantized_weight is None:
            return F.linear(input, self.weight, self.bias)
        
        # Unpack the quantized weight
        unpacked_weight = AbsmeanQuantization.unpack(self.quantized_weight, self.weight.shape)
        
        # Dequantize the weight
        dequantized_weight = AbsmeanQuantization.dequantize(unpacked_weight, self.weight_scale)
        
        # Perform the linear operation
        output = F.linear(input, dequantized_weight, self.bias)
        
        return output

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'