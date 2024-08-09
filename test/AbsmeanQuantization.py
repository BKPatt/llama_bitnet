import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AbsmeanQuantization:
    @staticmethod
    def quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logger.info(f"Quantizing tensor of shape: {x.shape}")
        scale = torch.mean(torch.abs(x), dim=-1, keepdim=True)
        q = torch.round(x / (scale + 1e-8)).clamp(-1, 1)
        return q.to(torch.int8), scale

    @staticmethod
    def quantized_matmul(q1: torch.Tensor, s1: torch.Tensor, q2: torch.Tensor, s2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logger.info(f"Quantized matmul with shapes: q1={q1.shape}, q2={q2.shape}")
        
        # Handle 4D tensors for attention operations
        if q1.dim() == 4 and q2.dim() == 4:
            batch_size, num_heads, seq_len, head_dim = q1.shape
            _, _, kv_seq_len, kv_dim = q2.shape
            
            # Reshape for batch matrix multiplication
            q1_reshaped = q1.transpose(1, 2).reshape(batch_size * seq_len, num_heads * head_dim)
            q2_reshaped = q2.permute(0, 2, 1, 3).reshape(batch_size * kv_seq_len, num_heads * kv_dim)
            
            logger.info(f"Reshaped q1: {q1_reshaped.shape}, q2: {q2_reshaped.shape}")
            
            # Perform matrix multiplication
            out_int = torch.bmm(q1_reshaped.int().view(batch_size, seq_len, num_heads * head_dim),
                                q2_reshaped.int().view(batch_size, kv_seq_len, num_heads * kv_dim).transpose(1, 2))
            
            # Reshape the output
            out_int = out_int.view(batch_size, seq_len, num_heads, kv_seq_len).transpose(1, 2)
        else:
            out_int = torch.matmul(q1.int(), q2.int())
        
        logger.info(f"Quantized matmul output shape: {out_int.shape}")
        out_scale = s1 * s2
        return out_int, out_scale

    @staticmethod
    def quantized_add(a: torch.Tensor, a_scale: torch.Tensor, b: torch.Tensor, b_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logger.info(f"Quantized add with shapes: a={a.shape}, b={b.shape}")
        if a_scale != b_scale:
            if a_scale > b_scale:
                b = (b.float() * b_scale / a_scale).round().to(torch.int32)
                out_scale = a_scale
            else:
                a = (a.float() * a_scale / b_scale).round().to(torch.int32)
                out_scale = b_scale
        else:
            out_scale = a_scale
        out = a + b
        logger.info(f"Quantized add output shape: {out.shape}")
        return out, out_scale

    @staticmethod
    def quantized_act(x: torch.Tensor, scale: torch.Tensor, act_fn) -> tuple[torch.Tensor, torch.Tensor]:
        logger.info(f"Quantized activation input shape: {x.shape}")
        x_float = x.float() * scale
        out_float = act_fn(x_float)
        out, out_scale = AbsmeanQuantization.quantize(out_float)
        logger.info(f"Quantized activation output shape: {out.shape}")
        return out, out_scale