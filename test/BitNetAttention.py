import torch
import torch.nn as nn
import math
import logging
from typing import Optional
from BitNetB158Config import BitNetB158Config
from QuantizedLinear import QuantizedLinear
from RotaryEmbedding import RotaryEmbedding
from AbsmeanQuantization import AbsmeanQuantization

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BitNetAttention(nn.Module):
    def __init__(self, config: BitNetB158Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        logger.info(f"Initializing BitNetAttention with hidden_size={self.hidden_size}, num_heads={self.num_heads}, head_dim={self.head_dim}")

        self.q_proj = QuantizedLinear(self.hidden_size, self.num_heads * self.head_dim, bias=False, device=config.device)
        self.k_proj = QuantizedLinear(self.hidden_size, 1024, bias=False, device=config.device)
        self.v_proj = QuantizedLinear(self.hidden_size, 1024, bias=False, device=config.device)
        self.o_proj = QuantizedLinear(self.hidden_size, self.hidden_size, bias=False, device=config.device)
        
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_position_embeddings=config.max_position_embeddings, device=config.device)

    def forward(self, hidden_states, hidden_states_scale, attention_mask=None, attention_mask_scale=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False):
        bsz, q_len, _ = hidden_states.size()
        logger.info(f"Input hidden_states shape: {hidden_states.shape}")

        query_states, query_states_scale = self.q_proj(hidden_states, hidden_states_scale)
        key_states, key_states_scale = self.k_proj(hidden_states, hidden_states_scale)
        value_states, value_states_scale = self.v_proj(hidden_states, hidden_states_scale)

        logger.info(f"After projection - query_states: {query_states.shape}, key_states: {key_states.shape}, value_states: {value_states.shape}")

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_heads, 1024 // self.num_heads).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_heads, 1024 // self.num_heads).transpose(1, 2)

        logger.info(f"After reshape - query_states: {query_states.shape}, key_states: {key_states.shape}, value_states: {value_states.shape}")

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        logger.info(f"Rotary embedding shapes - cos: {cos.shape}, sin: {sin.shape}")

        query_states, key_states = self.rotary_emb.apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        logger.info(f"After rotary embedding - query_states: {query_states.shape}, key_states: {key_states.shape}")

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
            logger.info(f"After concatenating past_key_value - key_states: {key_states.shape}, value_states: {value_states.shape}")

        past_key_value = (key_states, value_states) if use_cache else None

        logger.info(f"Before quantized_matmul - query_states: {query_states.shape}, key_states: {key_states.shape}")
        attn_weights, attn_weights_scale = AbsmeanQuantization.quantized_matmul(query_states, query_states_scale, key_states, key_states_scale)
        logger.info(f"After quantized_matmul - attn_weights: {attn_weights.shape}")

        attn_weights = attn_weights.float() * (1.0 / math.sqrt(self.head_dim))

        if attention_mask is not None:
            logger.info(f"Attention mask shape: {attention_mask.shape}")
            attn_weights, attn_weights_scale = AbsmeanQuantization.quantized_add(attn_weights, attn_weights_scale, attention_mask, attention_mask_scale)
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
        
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        logger.info(f"After softmax - attn_weights: {attn_weights.shape}")

        attn_output, attn_output_scale = AbsmeanQuantization.quantized_matmul(attn_weights, torch.tensor(1.0), value_states, value_states_scale)
        logger.info(f"After attention - attn_output: {attn_output.shape}")

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        logger.info(f"Before final projection - attn_output: {attn_output.shape}")

        attn_output, attn_output_scale = self.o_proj(attn_output, attn_output_scale)
        logger.info(f"Final output - attn_output: {attn_output.shape}")

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_output_scale, attn_weights, past_key_value