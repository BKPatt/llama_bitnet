import math
import torch
import torch.nn as nn
from typing import Optional, Tuple
from BitNetB158Config import BitNetB158Config
from QuantizedLinear import QuantizedLinear
from RotaryEmbedding import RotaryEmbedding
from RotaryEmbedding import apply_rotary_pos_emb

class BitNetAttention(nn.Module):
    def __init__(self, config: BitNetB158Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_heads = config.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings

        self.q_proj = QuantizedLinear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = QuantizedLinear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = QuantizedLinear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = QuantizedLinear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
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
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Handling past key values for incremental decoding
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=-2)
            value_states = torch.cat([past_key_value[1], value_states], dim=-2)

        # Rotary position embedding
        if position_ids is not None:
            cos, sin = self.rotary_emb(value_states, seq_len=key_states.shape[-2])
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask.unsqueeze(1).unsqueeze(2)

        attn_weights = torch.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        outputs = (attn_output,)
        if output_attentions:
            outputs += (attn_weights,)
        if use_cache:
            outputs += ((key_states, value_states),)

        return outputs

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)