import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from AbsmeanQuantization import AbsmeanQuantization

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_scale = nn.Parameter(torch.ones(out_features, 1))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, 'quantized_weight'):
            self.quantize()
        unpacked_weight = AbsmeanQuantization.unpack(self.quantized_weight, self.weight.shape)
        dequantized_weight = unpacked_weight.float() * self.weight_scale
        output = F.linear(input, dequantized_weight, self.bias)
        return output

    def quantize(self):
        weight = self.weight.data
        quantized_weight, weight_scale = AbsmeanQuantization.quantize(weight)
        packed_weight = AbsmeanQuantization.pack(quantized_weight)
        self.quantized_weight = nn.Parameter(packed_weight, requires_grad=False)
        self.weight_scale.data.copy_(weight_scale)
