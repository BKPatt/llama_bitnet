import torch
import torch.nn as nn
from typing import Optional, Tuple
from BitNetB158Config import BitNetB158Config
from BitNetAttention import BitNetAttention
from BitNetMLP import BitNetMLP
from RMSNorm import RMSNorm
from AbsmeanQuantization import AbsmeanQuantization

class BitNetB158Layer(nn.Module):
    def __init__(self, config: BitNetB158Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = BitNetAttention(config)
        self.mlp = BitNetMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention_mask_scale: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states
        residual_scale = hidden_states_scale

        hidden_states, hidden_states_scale = self.input_layernorm(hidden_states, hidden_states_scale)

        # Self Attention
        hidden_states, hidden_states_scale, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            hidden_states_scale=hidden_states_scale,
            attention_mask=attention_mask,
            attention_mask_scale=attention_mask_scale,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states, hidden_states_scale = AbsmeanQuantization.quantized_add(residual, residual_scale, hidden_states, hidden_states_scale)

        # Fully Connected
        residual = hidden_states
        residual_scale = hidden_states_scale
        hidden_states, hidden_states_scale = self.post_attention_layernorm(hidden_states, hidden_states_scale)
        hidden_states, hidden_states_scale = self.mlp(hidden_states, hidden_states_scale)
        hidden_states, hidden_states_scale = AbsmeanQuantization.quantized_add(residual, residual_scale, hidden_states, hidden_states_scale)

        outputs = (hidden_states, hidden_states_scale)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs