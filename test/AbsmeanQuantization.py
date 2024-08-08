import torch
import math

class AbsmeanQuantization:
    @staticmethod
    def quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = torch.mean(torch.abs(x), dim=-1, keepdim=True)
        q = torch.round(x / (scale + 1e-8)).clamp(-1, 1)
        return q, scale

    @staticmethod
    def dequantize(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return q * scale

    @staticmethod
    def pack(q: torch.Tensor) -> torch.Tensor:
        # Convert to unsigned integers (0, 1, 2)
        q_unsigned = (q + 1).to(torch.uint8)
        
        # Calculate number of elements and packed tensor shape
        num_elements = q.numel()
        packed_size = math.ceil(num_elements * 1.58 / 8)  # 1.58 bits per value
        
        # Reshape q_unsigned to 1D
        q_unsigned = q_unsigned.reshape(-1)
        
        # Initialize packed tensor
        packed = torch.zeros(packed_size, dtype=torch.uint8, device=q.device)
        
        # Pack values in batches
        batch_size = 1000000  # Adjust this value based on your GPU memory
        for start in range(0, num_elements, batch_size):
            end = min(start + batch_size, num_elements)
            batch = q_unsigned[start:end]
            
            bit_indices = torch.arange(start, end, device=q.device) * 1.58
            byte_indices = (bit_indices // 8).long()
            bit_offsets = (bit_indices % 8).long()
            
            # Pack lower bits
            packed[byte_indices] |= batch << bit_offsets
            
            # Pack overflow bits
            overflow_mask = bit_offsets > 6
            if overflow_mask.any():
                packed[byte_indices[overflow_mask] + 1] |= batch[overflow_mask] >> (8 - bit_offsets[overflow_mask])
        
        return packed

    @staticmethod
    def unpack(packed: torch.Tensor, original_shape: torch.Size) -> torch.Tensor:
        num_elements = original_shape.numel()
        q_unsigned = torch.zeros(num_elements, dtype=torch.uint8, device=packed.device)
        
        batch_size = 1000000  # Adjust this value based on your GPU memory
        for start in range(0, num_elements, batch_size):
            end = min(start + batch_size, num_elements)
            
            bit_indices = torch.arange(start, end, device=packed.device) * 1.58
            byte_indices = (bit_indices // 8).long()
            bit_offsets = (bit_indices % 8).long()
            
            # Unpack lower bits
            q_unsigned[start:end] = (packed[byte_indices] >> bit_offsets) & 0b11
            
            # Unpack overflow bits
            overflow_mask = bit_offsets > 6
            if overflow_mask.any():
                q_unsigned[start:end][overflow_mask] |= (packed[byte_indices[overflow_mask] + 1] << (8 - bit_offsets[overflow_mask])) & 0b11
        
        # Convert back to {-1, 0, 1}
        q = q_unsigned.to(torch.int8) - 1
        return q.reshape(original_shape)
