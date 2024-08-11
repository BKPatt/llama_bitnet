import os
import torch
import torch.nn as nn
from typing import Optional, List, Union, Tuple
from BitNetB158Config import BitNetB158Config
from BitNetB158Layer import BitNetB158Layer
from RMSNorm import RMSNorm
import torch.nn.functional as F
from AbsmeanQuantization import AbsmeanQuantization
from util import _expand_mask, _make_causal_mask

class BitNetB158Model(nn.Module):
    def __init__(self, config: BitNetB158Config, tokenizer=None):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.bos_token_id = config.bos_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.bos_token_id)
        self.layers = nn.ModuleList([BitNetB158Layer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, dict]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # Past key values length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        
        if position_ids is None:
            device = inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=inputs_embeds.device)

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.layers))

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, (decoder_layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return {
            "last_hidden_state": hidden_states,
            "past_key_values": next_cache,
            "hidden_states": all_hidden_states,
            "attentions": all_self_attns,
        }

    @classmethod
    def from_pretrained(cls, model_path: str, tokenizer=None):
        # Load configuration
        config_path = os.path.join(model_path, "config.json")
        config = BitNetB158Config.from_json(config_path)

        # Initialize model
        model = cls(config)
        model.tokenizer = tokenizer

        # Load weights
        weights_path = os.path.join(model_path, "pytorch_model.bin")
        state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)

        return model

    def generate(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None, 
        max_length: int = 100, 
        temperature: float = 0.7, 
        top_p: float = 0.9, 
        top_k: int = 50, 
        **kwargs
    ) -> str:
        device = input_ids.device
        past_key_values = None
        generated_tokens = input_ids

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        for _ in range(max_length - input_ids.size(1)):
            with torch.no_grad():
                outputs = self(
                    input_ids=generated_tokens,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    **kwargs
                )
            
            if isinstance(outputs, dict):
                logits = outputs["last_hidden_state"][:, -1, :]
                past_key_values = outputs["past_key_values"]
            else:
                logits = outputs[0][:, -1, :]
                past_key_values = outputs[1]

            # Apply temperature scaling
            logits = logits / temperature

            # Apply top-k and top-p sampling
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, top_k)
                probs = torch.nn.functional.softmax(top_k_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                next_token = top_k_indices.gather(-1, next_token).squeeze(1)
            else:
                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)

            # Append the generated token to the sequence
            generated_tokens = torch.cat([generated_tokens, next_token.unsqueeze(1)], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token.unsqueeze(1))], dim=1)

            # Stop if the EOS token is generated
            if next_token.item() in self.config.eos_token_id:
                break

        # Convert generated tokens to text
        generated_text = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        return generated_text

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # Create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            
            # Ensure the shapes match before combining
            if combined_attention_mask is not None:
                max_len = max(combined_attention_mask.size(-1), expanded_attn_mask.size(-1))
                combined_attention_mask = F.pad(combined_attention_mask, (0, max_len - combined_attention_mask.size(-1)))
                expanded_attn_mask = F.pad(expanded_attn_mask, (0, max_len - expanded_attn_mask.size(-1)))
            
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def quantize(self):
        # Quantize embed_tokens
        q_weight, q_scale = AbsmeanQuantization.quantize(self.embed_tokens.weight.data)
        self.embed_tokens.weight.data = q_weight.float() * q_scale

        # Quantize norm
        q_weight, q_scale = AbsmeanQuantization.quantize(self.norm.weight.data)
        self.norm.weight.data = q_weight.float() * q_scale

        # Quantize all layers
        for layer in self.layers:
            layer.quantize()
