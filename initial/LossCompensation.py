import torch
import torch.nn as nn

class LossCompensation(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, bitnet_output: torch.Tensor, fp16_output: torch.Tensor) -> torch.Tensor:
        """
        Apply loss compensation to the BitNet output.

        Args:
            bitnet_output (torch.Tensor): Output from the BitNet model
            fp16_output (torch.Tensor): Output from the original FP16 model

        Returns:
            torch.Tensor: Compensated output
        """
        return self.alpha * bitnet_output + (1 - self.alpha) * fp16_output.detach()

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}"