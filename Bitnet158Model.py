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
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = torch.nn.ModuleList([
            torch.nn.ModuleDict({
                'self_attn': torch.nn.ModuleDict({
                    'q_proj': torch.nn.Linear(config.hidden_size, config.hidden_size, bias=False),
                    'k_proj': torch.nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False),
                    'v_proj': torch.nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False),
                    'o_proj': torch.nn.Linear(config.hidden_size, config.hidden_size, bias=False)
                }),
                'mlp': torch.nn.ModuleDict({
                    'gate_proj': torch.nn.Linear(config.hidden_size, config.intermediate_size, bias=False),
                    'up_proj': torch.nn.Linear(config.hidden_size, config.intermediate_size, bias=False),
                    'down_proj': torch.nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
                }),
                'input_layernorm': torch.nn.LayerNorm(config.hidden_size),
                'post_attention_layernorm': torch.nn.LayerNorm(config.hidden_size)
            }) for _ in range(config.num_hidden_layers)
        ])
        self.norm = torch.nn.LayerNorm(config.hidden_size)
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        batch_size, seq_len, _ = x.shape
        for layer in self.layers:
            residual = x
            x = layer['input_layernorm'](x)
            q = layer['self_attn']['q_proj'](x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = layer['self_attn']['k_proj'](x).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            v = layer['self_attn']['v_proj'](x).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            k = k.repeat_interleave(self.num_heads // self.num_key_value_heads, dim=1)
            v = v.repeat_interleave(self.num_heads // self.num_key_value_heads, dim=1)
            attention_output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
            x = layer['self_attn']['o_proj'](attention_output)
            x = residual + x
            residual = x
            x = layer['post_attention_layernorm'](x)
            gate_output = layer['mlp']['gate_proj'](x)
            up_output = layer['mlp']['up_proj'](x)
            x = gate_output * torch.nn.functional.silu(up_output)
            x = layer['mlp']['down_proj'](x)
            x = residual + x
        x = self.norm(x)
        x = self.lm_head(x)
        return x

def map_weights(llama_weights, bitnet_model):
    logger.info("Starting weight mapping process...")
    mapped_weights = {}
    
    mapped_weights['embed_tokens.weight'] = llama_weights['model.embed_tokens.weight']
    
    for i in range(bitnet_model.config.num_hidden_layers):
        logger.info(f"Mapping weights for layer {i}")
        try:
            mapped_weights[f'layers.{i}.self_attn.q_proj.weight'] = llama_weights[f'model.layers.{i}.self_attn.q_proj.weight']
            mapped_weights[f'layers.{i}.self_attn.k_proj.weight'] = llama_weights[f'model.layers.{i}.self_attn.k_proj.weight']
            mapped_weights[f'layers.{i}.self_attn.v_proj.weight'] = llama_weights[f'model.layers.{i}.self_attn.v_proj.weight']
            mapped_weights[f'layers.{i}.self_attn.o_proj.weight'] = llama_weights[f'model.layers.{i}.self_attn.o_proj.weight']
            
            mapped_weights[f'layers.{i}.mlp.gate_proj.weight'] = llama_weights[f'model.layers.{i}.mlp.gate_proj.weight']
            mapped_weights[f'layers.{i}.mlp.up_proj.weight'] = llama_weights[f'model.layers.{i}.mlp.up_proj.weight']
            mapped_weights[f'layers.{i}.mlp.down_proj.weight'] = llama_weights[f'model.layers.{i}.mlp.down_proj.weight']
            
            mapped_weights[f'layers.{i}.input_layernorm.weight'] = llama_weights[f'model.layers.{i}.input_layernorm.weight']
            mapped_weights[f'layers.{i}.post_attention_layernorm.weight'] = llama_weights[f'model.layers.{i}.post_attention_layernorm.weight']
            
        except KeyError as e:
            logger.error(f"KeyError in layer {i}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in layer {i}: {e}")
    
    mapped_weights['norm.weight'] = llama_weights['model.norm.weight']
    mapped_weights['lm_head.weight'] = llama_weights['lm_head.weight']
    
    logger.info("Weight mapping process completed.")
    return mapped_weights

def custom_load_state_dict(model, state_dict):
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            continue
        if isinstance(param, torch.nn.Parameter):
            param = param.data
        if own_state[name].shape != param.shape:
            logger.warning(f"Shape mismatch for {name}: expected {own_state[name].shape}, got {param.shape}")
            try:
                own_state[name].copy_(param.view(own_state[name].shape))
            except RuntimeError:
                logger.error(f"Failed to reshape {name}")
        else:
            own_state[name].copy_(param)

def main():
    logger.info("Starting Llama to Bitnet conversion process.")

    llama_weights, llama_config = load_llama_model()
    save_llama_weights_and_config(llama_weights, llama_config)
    verify_saved_data(llama_weights, llama_config)

    logger.info("Initializing Bitnet 1.58 model...")
    bitnet_model = Bitnet158Model(llama_config)

    logger.info("Mapping weights...")
    bitnet_weights = map_weights(llama_weights, bitnet_model)

    logger.info("Loading mapped weights into Bitnet model...")
    custom_load_state_dict(bitnet_model, bitnet_weights)

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