import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Compute the mean square
        norm_x = x.norm(2, dim=-1, keepdim=True)
        rms_x = norm_x * (x.shape[-1] ** -0.5)
        
        # Normalize
        x_normed = x / (rms_x + self.eps)
        
        # Scale
        return self.weight * x_normed

    def extra_repr(self):
        return 'eps={}'.format(self.eps)

# Example usage
# layer = RMSNorm(4096)
# output = layer(input_tensor)
