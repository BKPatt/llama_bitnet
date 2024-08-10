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
        # Move the input tensor to the same device as the quantized weight
        input = input.to(self.quantized_weight.device)

        # Perform the linear operation using quantized weights
        output = F.linear(input, self.quantized_weight, self.bias)

        # Reshape the weight scale to match the output shape
        weight_scale = self.weight_scale.view(1, -1)

        # Multiply the output by the reshaped weight scale
        output = output * weight_scale

        return output

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )