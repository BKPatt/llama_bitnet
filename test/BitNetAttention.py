import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from QuantizedLinear import QuantizedLinear
from RotaryEmbedding import RotaryEmbedding

class BitNetAttention(nn.Module):
    def __init__(self, config):
        super(BitNetAttention, self).__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        assert self.head_dim * self.num_heads == self.hidden_size, "hidden_size must be divisible by num_attention_heads"

        # Instead of separate projections, combine them into a single linear layer
        self.proj = QuantizedLinear(self.hidden_size, self.hidden_size * 3)

        self.o_proj = QuantizedLinear(self.hidden_size, self.hidden_size)

        self.rotary_emb = RotaryEmbedding(self.head_dim)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        mixed_proj_layer = self.proj(hidden_states)
        
        # Split into query, key, value
        (mixed_query_layer, mixed_key_layer, mixed_value_layer) = mixed_proj_layer.split(self.hidden_size, dim=-1)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Apply rotary embedding to query and key layers
        query_layer, key_layer = self.rotary_emb(query_layer, key_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        output = self.o_proj(context_layer)

        return output

