import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, quantization_scheme='absmean'):
        super(QuantizedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quantization_scheme = quantization_scheme

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

    def forward(self, input):
        quantized_weight = self.quantize_weight(self.weight)
        return F.linear(input, quantized_weight, self.bias)

    def quantize_weight(self, weight):
        if self.quantization_scheme == 'absmean':
            scale = weight.abs().mean(dim=1, keepdim=True)
            quantized_weight = torch.round(weight / scale) * scale
        else:
            raise ValueError(f"Unsupported quantization scheme: {self.quantization_scheme}")
        return quantized_weight

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, quantization_scheme={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.quantization_scheme
        )