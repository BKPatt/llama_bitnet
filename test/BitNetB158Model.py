import torch
import torch.nn as nn

from BitNetB158Layer import BitNetB158Layer
from RMSNorm import RMSNorm

class BitNetB158Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Create the layers
        self.layers = nn.ModuleList([BitNetB158Layer(config) for _ in range(config.num_hidden_layers)])
        
        # Final layer normalization
        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Language modeling head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None):
        # Embedding layer
        hidden_states = self.embed_tokens(input_ids)

        # Process through all layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)

        # Final layer normalization
        hidden_states = self.norm(hidden_states)

        # Language modeling head
        logits = self.lm_head(hidden_states)

        return logits

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head.weight

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.weight = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None):
        return {"input_ids": input_ids, "attention_mask": attention_mask}
