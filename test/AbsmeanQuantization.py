import torch
import math

class AbsmeanQuantization:
    @staticmethod
    def quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = torch.mean(torch.abs(x), dim=-1, keepdim=True)
        q = torch.round(x / (scale + 1e-8)).clamp(-1, 1)
        return q.to(torch.int8), scale

    @staticmethod
    def dequantize(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return q.float() * scale

    @staticmethod
    def pack(q: torch.Tensor) -> torch.Tensor:
        q_unsigned = (q + 1).to(torch.uint8)  # Convert from [-1, 1] to [0, 2]
        packed = torch.zeros((q_unsigned.numel() + 1) // 2, dtype=torch.uint8, device=q.device)
        
        # Get all even-indexed elements and shift left by 4 bits for odd-indexed elements to combine them
        # Handle the case where q has an odd number of elements by ignoring the last if it exists
        if q_unsigned.numel() % 2 == 0:
            packed = (q_unsigned[0::2] | (q_unsigned[1::2] << 4))
        else:
            # If there's an odd number of q elements, handle the last element separately
            packed[:-1] = (q_unsigned[0::2] | (q_unsigned[1::2] << 4))
            packed[-1] = q_unsigned[-2]  # Assign the last single element which has no pair

        return packed

    @staticmethod
    def unpack(packed: torch.Tensor, original_shape: torch.Size) -> torch.Tensor:
        q_unsigned = torch.empty(original_shape.numel(), dtype=torch.uint8, device=packed.device)
        q_unsigned[0::2] = packed[0::2] & 0x0F
        q_unsigned[1::2] = packed[0::2] >> 4
        q = q_unsigned.to(torch.int8) - 1
        return q.reshape(original_shape)