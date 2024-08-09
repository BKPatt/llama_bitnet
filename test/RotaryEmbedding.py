from typing import Optional
import torch
import torch.nn as nn

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(self.device) / dim))
        self.max_seq_len_cached = max_position_embeddings
        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len: int):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=self.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
        # Assuming q and k have been reshaped to (batch_size, num_heads, seq_length, head_dim)
        q_len, q_dim = q.shape[-2:]
        k_len, k_dim = k.shape[-2:]

        if position_ids is not None:
            cos = cos.index_select(2, position_ids)
            sin = sin.index_select(2, position_ids)

        # Adjust cos and sin to match the specific head dimensions of q and k
        cos_q = cos[:, :, :q_len, :q_dim]
        sin_q = sin[:, :, :q_len, :q_dim]
        cos_k = cos[:, :, :k_len, :k_dim]
        sin_k = sin[:, :, :k_len, :k_dim]

        q_embed = (q * cos_q) + (RotaryEmbedding.rotate_half(q) * sin_q)
        k_embed = (k * cos_k) + (RotaryEmbedding.rotate_half(k) * sin_k)
        return q_embed, k_embed