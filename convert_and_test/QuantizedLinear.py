import torch
import torch.nn as nn
import torch.nn.functional as F
from AbsmeanQuantization import AbsmeanQuantization

class QuantizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.register_buffer('quantized_weight', torch.zeros(out_features, in_features, dtype=torch.int8))
        self.weight_scale = nn.Parameter(torch.ones(out_features))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Debugging: Print the input shape
        print(f"Input shape to QuantizedLinear: {input.shape}")
        
        output = F.linear(input, self.quantized_weight.float())
        output = output * self.weight_scale.unsqueeze(1)
        
        # Debugging: Print the output shape after projection
        print(f"Output shape from QuantizedLinear: {output.shape}")
        
        if self.bias is not None:
            output += self.bias
        return output

    def quantize(self, weight: torch.Tensor):
        quantized_weight, weight_scale = AbsmeanQuantization.quantize(weight)
        self.quantized_weight.copy_(quantized_weight)
        self.weight_scale.data.copy_(weight_scale.squeeze())
