import torch.nn.functional as F
import torch
import torch.nn as nn
from typing import List, Tuple
from BitNetLayer import BitNetLayer
from BitNetLlamaBlock import BitNetLlamaBlock
from BitNetLayerNorm import BitNetLayerNorm

class BitNetLlamaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([BitNetLlamaBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = BitNetLayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def generate(self, input_ids, max_length=50, temperature=0.7):
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        batch_size = input_ids.shape[0]
        
        for _ in range(max_length - input_ids.shape[1]):
            # Get the last token's embedding
            inputs = input_ids[:, -1].unsqueeze(-1)
            
            # Forward pass through the model
            outputs = self(inputs)
            next_token_logits = outputs[0][:, -1, :]
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Sample from the distribution
            probs = F.softmax(next_token_logits, dim=-1)  # This line should now work correctly
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append the new token to the sequence
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            # Check if all sequences have reached the end token
            if (next_token == self.config.eos_token_id).all():
                break
        
        return input_ids

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        past_key_values: List[Tuple[torch.FloatTensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Tuple:
        batch_size, seq_length = input_ids.shape

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=input_ids.device)

        # Embed input tokens
        hidden_states = self.embed_tokens(input_ids)

        # Initialize past_key_values if not provided
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        next_cache = []
        for idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_cache.append(layer_outputs[-1])

            if output_attentions:
                all_attentions += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return (hidden_states, next_cache, all_hidden_states, all_attentions)

    def quantize_weights(self):
        for layer in self.layers:
            layer.self_attn.q_proj.weight.data = torch.sign(layer.self_attn.q_proj.weight.data)
            layer.self_attn.k_proj.weight.data = torch.sign(layer.self_attn.k_proj.weight.data)
            layer.self_attn.v_proj.weight.data = torch.sign(layer.self_attn.v_proj.weight.data)
            layer.self_attn.o_proj.weight.data = torch.sign(layer.self_attn.o_proj.weight.data)
            layer.mlp.gate_proj.weight.data = torch.sign(layer.mlp.gate_proj.weight.data)
            layer.mlp.up_proj.weight.data = torch.sign(layer.mlp.up_proj.weight.data)
            layer.mlp.down_proj.weight.data = torch.sign(layer.mlp.down_proj.weight.data)

    def verify_binarization(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                unique_values = torch.unique(param.data)
                if len(unique_values) > 2 or not all(torch.abs(unique_values) <= 1):
                    print(f"Warning: {name} is not properly binarized. Unique values: {unique_values}")
                else:
                    print(f"{name} is correctly binarized.")

    def convert_from_fp16(self, fp16_model):
        # Helper function to binarize and scale
        def binarize_and_scale(tensor):
            binary = torch.sign(tensor)
            scale = tensor.abs().mean()
            return binary, scale

        # Embed tokens (this layer typically isn't binarized)
        self.embed_tokens.weight.data = fp16_model.model.embed_tokens.weight.data.clone()

        # Convert and copy weights for each layer
        for bit_layer, fp16_layer in zip(self.layers, fp16_model.model.layers):
            for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                weight, scale = binarize_and_scale(getattr(fp16_layer.self_attn, name).weight.data)
                getattr(bit_layer.self_attn, name).weight.data = weight
                getattr(bit_layer.self_attn, name).scaling_factor.data = torch.tensor([scale])

            for name in ['gate_proj', 'up_proj', 'down_proj']:
                weight, scale = binarize_and_scale(getattr(fp16_layer.mlp, name).weight.data)
                getattr(bit_layer.mlp, name).weight.data = weight
                getattr(bit_layer.mlp, name).scaling_factor.data = torch.tensor([scale])

            # LayerNorm weights are typically not binarized
            bit_layer.input_layernorm.weight.data = fp16_layer.input_layernorm.weight.data.clone()
            bit_layer.post_attention_layernorm.weight.data = fp16_layer.post_attention_layernorm.weight.data.clone()

        # Norm weights are typically not binarized
        self.norm.weight.data = fp16_model.model.norm.weight.data.clone()

    def extra_repr(self) -> str:
        return f"num_layers={len(self.layers)}, vocab_size={self.vocab_size}"