import torch
import torch.nn as nn

class AbsmeanQuantization(nn.Module):
    def __init__(self, num_features):
        super(AbsmeanQuantization, self).__init__()
        self.num_features = num_features
        self.weight_scale = nn.Parameter(torch.Tensor(num_features))
        self.register_buffer('quantized_weight', torch.Tensor(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight_scale)
        self.quantized_weight.zero_()

    def forward(self, input):
        abs_mean = input.abs().mean(dim=-1, keepdim=True)
        scaled_input = input / (abs_mean + 1e-5)
        quantized = scaled_input.sign() * abs_mean

        self.quantized_weight = quantized.detach()
        return quantized * self.weight_scale

    def extra_repr(self):
        return 'num_features={}'.format(self.num_features)
