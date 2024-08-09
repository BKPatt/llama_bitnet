import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from AbsmeanQuantization import AbsmeanQuantization

class QuantizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, device=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize weights
        weight = torch.empty(out_features, in_features, device=self.device)
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        
        # Quantize weights
        quantized_weight, weight_scale = AbsmeanQuantization.quantize(weight)
        self.quantized_weight = nn.Parameter(quantized_weight.to(self.device), requires_grad=False)
        self.weight_scale = nn.Parameter(weight_scale.to(self.device))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features).to(self.device))
            nn.init.uniform_(self.bias, -0.1, 0.1)
        else:
            self.register_parameter('bias', None)

    def forward(self, input: torch.Tensor, input_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dequantized_weight = self.quantized_weight.float() * self.weight_scale
        output = F.linear(input.float() * input_scale, dequantized_weight, self.bias)
        output, output_scale = AbsmeanQuantization.quantize(output)
        return output, output_scale

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )