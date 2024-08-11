import torch
import torch.nn as nn
import torch.nn.functional as F
from AbsmeanQuantization import AbsmeanQuantization

class QuantizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Use register_buffer to ensure ternary_weight is part of the state dict but doesn't require gradients
        self.register_buffer('ternary_weight', torch.zeros(out_features, in_features, dtype=torch.int8))
        self.weight_scale = nn.Parameter(torch.ones(out_features, 1))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Ensure input is int8 and shape is correct
        if input.dtype != torch.int8:
            raise ValueError("Expected input tensor to be quantized to int8.")

        # Ensure the input shape matches the weight shape for matrix multiplication
        if input.shape[1] != self.in_features:
            raise ValueError(f"Input shape mismatch: expected input with {self.in_features} features, but got {input.shape[1]} features.")

        # Perform ternary matrix multiplication
        output = torch.zeros(input.size(0), self.out_features, dtype=torch.int32, device=input.device)
        for i in range(self.out_features):
            for j in range(self.in_features):
                if self.ternary_weight[i, j] == 1:
                    output[:, i] += input[:, j]
                elif self.ternary_weight[i, j] == -1:
                    output[:, i] -= input[:, j]

        # Apply scaling
        output = output.float() * self.weight_scale.view(1, -1)

        if self.bias is not None:
            output += self.bias

        return output

    def quantize(self, weight: torch.Tensor = None):
        if weight is None:
            return
        # Ensure weight shape matches the expected shape
        if weight.shape != self.ternary_weight.shape:
            print(f"Reshaping weight from {weight.shape} to {self.ternary_weight.shape}")
            weight = weight.view(self.ternary_weight.shape)
        
        # Compute the average absolute value
        avg_abs = weight.abs().mean()
        
        # Quantize to ternary values
        ternary = torch.zeros_like(weight, dtype=torch.int8)
        ternary[weight > 0.5 * avg_abs] = 1
        ternary[weight < -0.5 * avg_abs] = -1
        
        self.ternary_weight.copy_(ternary)
        self.weight_scale.data.copy_(avg_abs.view(-1, 1))

    def __repr__(self):
        return f'QuantizedLinear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})'
