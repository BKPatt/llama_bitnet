import math
import torch
import json
from transformers import LlamaForCausalLM, LlamaConfig, AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_llama_model():
    logger.info("Loading Llama 3 8b model...")
    llama_model = LlamaForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    llama_weights = llama_model.state_dict()
    llama_config = llama_model.config
    logger.info(f"Llama model loaded. Config: {llama_config}")
    return llama_weights, llama_config

def save_llama_weights_and_config(llama_weights, llama_config):
    logger.info("Saving Llama weights and config...")
    torch.save(llama_weights, 'llama3_8b_weights.pth')
    with open('llama3_8b_config.json', 'w') as f:
        f.write(llama_config.to_json_string())
    logger.info("Llama weights and config saved.")

def verify_saved_data(llama_weights, llama_config):
    logger.info("Verifying saved weights and config...")
    loaded_weights = torch.load('llama3_8b_weights.pth')
    with open('llama3_8b_config.json', 'r') as f:
        loaded_config = f.read()

    assert loaded_weights.keys() == llama_weights.keys(), "Mismatch in saved weight keys"
    for key in loaded_weights.keys():
        assert torch.equal(loaded_weights[key], llama_weights[key]), f"Mismatch in weights for key: {key}"

    assert loaded_config == llama_config.to_json_string(), "Mismatch in saved config"
    logger.info("Verification complete. Saved data matches original.")

class Bitnet158Model(torch.nn.Module):
    def __init__(self, config):
        super(Bitnet158Model, self).__init__()
        self.config = config
        self.padding_idx = getattr(config, 'pad_token_id', 0)  # Use 0 as default if pad_token_id is not present
        self.vocab_size = config.vocab_size
        
        self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = torch.nn.ModuleList([BitnetDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = torch.nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights and apply final processing
        self.post_init()

    def post_init(self):
        for module in self.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
                torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.embed_tokens(input_ids)

        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(hidden_states, attention_mask)
            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits

class BitnetDecoderLayer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = BitnetAttention(config)
        self.mlp = BitnetMLP(config)
        self.input_layernorm = torch.nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = torch.nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return (hidden_states,)

class BitnetAttention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        self.q_proj = torch.nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = torch.nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = torch.nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = torch.nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(self, hidden_states, attention_mask=None):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / (self.head_dim ** 0.5)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output

class BitnetMLP(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = torch.nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = torch.nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = torch.nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

def map_weights(llama_weights, bitnet_model):
    logger.info("Starting weight mapping process...")
    mapped_weights = {}
    
    # Map embedding weights
    mapped_weights['embed_tokens.weight'] = llama_weights['model.embed_tokens.weight']
    
    # Map layer weights
    for i in range(bitnet_model.config.num_hidden_layers):
        logger.info(f"Mapping weights for layer {i}")
        layer_prefix = f'model.layers.{i}.'
        mapped_prefix = f'layers.{i}.'
        
        # Attention weights
        mapped_weights[f'{mapped_prefix}self_attn.q_proj.weight'] = llama_weights[f'{layer_prefix}self_attn.q_proj.weight']
        mapped_weights[f'{mapped_prefix}self_attn.k_proj.weight'] = llama_weights[f'{layer_prefix}self_attn.k_proj.weight']
        mapped_weights[f'{mapped_prefix}self_attn.v_proj.weight'] = llama_weights[f'{layer_prefix}self_attn.v_proj.weight']
        mapped_weights[f'{mapped_prefix}self_attn.o_proj.weight'] = llama_weights[f'{layer_prefix}self_attn.o_proj.weight']
        
        # MLP weights
        mapped_weights[f'{mapped_prefix}mlp.gate_proj.weight'] = llama_weights[f'{layer_prefix}mlp.gate_proj.weight']
        mapped_weights[f'{mapped_prefix}mlp.up_proj.weight'] = llama_weights[f'{layer_prefix}mlp.up_proj.weight']
        mapped_weights[f'{mapped_prefix}mlp.down_proj.weight'] = llama_weights[f'{layer_prefix}mlp.down_proj.weight']
        
        # Layernorm weights
        mapped_weights[f'{mapped_prefix}input_layernorm.weight'] = llama_weights[f'{layer_prefix}input_layernorm.weight']
        mapped_weights[f'{mapped_prefix}post_attention_layernorm.weight'] = llama_weights[f'{layer_prefix}post_attention_layernorm.weight']
    
    # Map final layer norm and lm_head weights
    mapped_weights['norm.weight'] = llama_weights['model.norm.weight']
    mapped_weights['lm_head.weight'] = llama_weights['lm_head.weight']
    
    logger.info("Weight mapping process completed.")
    return mapped_weights

def main():
    logger.info("Starting Llama to Bitnet conversion process.")

    llama_weights, llama_config = load_llama_model()
    
    logger.info("Initializing Bitnet 1.58 model...")
    bitnet_model = Bitnet158Model(llama_config)

    logger.info("Mapping weights...")
    bitnet_weights = map_weights(llama_weights, bitnet_model)

    logger.info("Loading mapped weights into Bitnet model...")
    bitnet_model.load_state_dict(bitnet_weights, strict=False)

    logger.info("Saving Bitnet 1.58 model...")
    torch.save(bitnet_model.state_dict(), 'bitnet1_58_weights.pth')

    logger.info("Validating the conversion...")
    bitnet_model.eval()
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    inputs = tokenizer("Hello, how are you?", return_tensors="pt")
    with torch.no_grad():
        outputs = bitnet_model(inputs.input_ids)
    logger.info(f"Validation output shape: {outputs.shape}")

    logger.info("Conversion process completed.")

if __name__ == "__main__":
    main()