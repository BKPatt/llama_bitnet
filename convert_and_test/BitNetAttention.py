import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from BitNetB158Config import BitNetB158Config
from QuantizedLinear import QuantizedLinear
from RotaryEmbedding import RotaryEmbedding
from util import repeat_kv, apply_rotary_pos_emb

class BitNetAttention(nn.Module):
    def __init__(self, config: BitNetB158Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads

        self.q_proj = QuantizedLinear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = QuantizedLinear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = QuantizedLinear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = QuantizedLinear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        batch_size, seq_length, _ = hidden_states.shape

        # Project hidden states to query, key, and value states
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape to (batch_size, num_heads, seq_length, head_dim)
        query_states = query_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # Concatenate past key/value states
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # Repeat key/value states to match the number of attention heads
        key_states = repeat_kv(key_states, self.num_heads // self.num_key_value_heads)
        value_states = repeat_kv(value_states, self.num_heads // self.num_key_value_heads)

        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / torch.sqrt(torch.tensor(self.head_dim, dtype=query_states.dtype))

        if attention_mask is not None:
            # Adjust attention_mask to match the shape of attn_weights
            attention_mask = attention_mask[:, :, -query_states.size(2):, :]
            attn_weights = attn_weights + attention_mask

        # Apply softmax to get attention probabilities
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Compute attention output
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights if output_attentions else None, past_key_value

    def quantize(self):
        # Quantize the projection layers
        self.q_proj.quantize()
        self.k_proj.quantize()
        self.v_proj.quantize()
        self.o_proj.quantize()
