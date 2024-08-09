import math
import torch
import torch.nn as nn

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048):
        super(RotaryEmbedding, self).__init__()
        self.dim = dim
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.max_position_embeddings = max_position_embeddings

        # Cache the positional embeddings to avoid recalculating them
        self.register_buffer("cos_cache", None, persistent=False)
        self.register_buffer("sin_cache", None, persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]

        if self.cos_cache is None or self.sin_cache is None or self.cos_cache.shape[0] < seq_len:
            self._create_rotary_embeddings(seq_len)

        cos_emb = self.cos_cache[:seq_len, :]
        sin_emb = self.sin_cache[:seq_len, :]

        return (x * cos_emb) + (self._rotate_half(x) * sin_emb)

    def _create_rotary_embeddings(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.cos_cache = emb.cos().unsqueeze(0)
        self.sin_cache = emb.sin().unsqueeze(0)

    def _rotate_half(self, x):
        x1, x2 = x[..., :self.dim//2], x[..., self.dim//2:]
        return torch.cat((-x2, x1), dim=-1)

    def extra_repr(self):
        return 'dim={}, max_position_embeddings={}'.format(self.dim, self.max_position_embeddings)
