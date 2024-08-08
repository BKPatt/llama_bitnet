import torch
import torch.nn as nn
from BitNetLayer import BitNetLayer
from BitNetAttention import BitNetAttention
from BitNetFFN import BitNetFFN
from BitNetLayerNorm import BitNetLayerNorm

class BitNetLlamaBlock(BitNetLayer):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = BitNetAttention(config)
        self.mlp = BitNetFFN(config)
        self.input_layernorm = BitNetLayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = BitNetLayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

    def quantize_weights(self):
        self.self_attn.quantize_weights()
        self.mlp.quantize_weights()
        # Note: LayerNorm typically doesn't use binary weights

    def update_scaling_factors(self):
        self.self_attn.update_scaling_factors()
        self.mlp.update_scaling_factors()

    def extra_repr(self):
        return f'hidden_size={self.hidden_size}, num_attention_heads={self.num_attention_heads}'