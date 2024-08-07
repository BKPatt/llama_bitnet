import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from BitNetLayer import BitNetLayer  # Update this line

class BitNetLinear(BitNetLayer):  # This line is correct now
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        binary_weight = self.binarize(self.weight)
        scaled_weight = self.apply_scaling(binary_weight)
        
        # Straight-through estimator
        binary_weight_st = binary_weight.detach() + self.weight - self.weight.detach()
        
        output = F.linear(input, binary_weight_st, self.bias)
        return output

    def quantize_weights(self):
        with torch.no_grad():
            self.weight.data = self.binarize(self.weight.data)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'

    def get_binarized_weight(self) -> torch.Tensor:
        return self.binarize(self.weight)

    def get_scaled_weight(self) -> torch.Tensor:
        return self.apply_scaling(self.get_binarized_weight())