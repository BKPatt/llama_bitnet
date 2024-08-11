import torch
import math

class AbsmeanQuantization:
    @staticmethod
    def quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = torch.mean(torch.abs(x), dim=-1, keepdim=True)
        q = torch.round(x / (scale + 1e-8)).clamp(-1, 1)
        return q.to(torch.int8), scale

    @staticmethod
    def quantize_scale(scale: torch.Tensor) -> torch.Tensor:
        # Quantize the scale factor using a similar approach
        scale_min = scale.min()
        scale_max = scale.max()
        
        # Normalize scale to fit into int8
        scale_normalized = (scale - scale_min) / (scale_max - scale_min)
        scale_quantized = torch.round(scale_normalized * 255).to(torch.int8)
        
        return scale_quantized, scale_min, scale_max

    @staticmethod
    def pack(q: torch.Tensor, batch_size: int = 1000000) -> torch.Tensor:
        # Existing packing logic...
        pass

    @staticmethod
    def unpack(packed: torch.Tensor, original_shape: torch.Size, batch_size: int = 1000000) -> torch.Tensor:
        # Existing unpacking logic...
        pass
