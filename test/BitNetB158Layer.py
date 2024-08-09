import torch
import torch.nn as nn
import torch.nn.functional as F

from BitNetAttention import BitNetAttention
from BitNetMLP import BitNetMLP
from RMSNorm import RMSNorm

class BitNetB158Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = BitNetAttention(config)
        self.mlp = BitNetMLP(config)
        self.norm1 = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None):
        # Self-attention with normalization
        attn_output = self.self_attn(self.norm1(hidden_states), attention_mask=attention_mask)
        hidden_states = hidden_states + attn_output

        # MLP with normalization
        mlp_output = self.mlp(self.norm2(hidden_states))
        hidden_states = hidden_states + mlp_output

        return hidden_states
