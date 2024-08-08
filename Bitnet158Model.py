import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, LlamaConfig
import logging

logging.basicConfig(filename='model_inference.log', level=logging.DEBUG)
logger = logging.getLogger(__name__)

class BitNetLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(BitNetLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.scale = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.constant_(self.scale, 1.0)

    def quantize_weights(self, weight):
        abs_weight = torch.abs(weight)
        scale = torch.mean(abs_weight, dim=1).clamp(min=1e-8)
        scaled_weight = weight / scale.unsqueeze(1)
        binary_weight = torch.sign(scaled_weight)
        return binary_weight, scale

    def forward(self, input):
        binary_weight, scale = self.quantize_weights(self.weight)
        output = F.linear(input, binary_weight)
        output = output * scale.unsqueeze(0).unsqueeze(0) * self.scale.unsqueeze(0).unsqueeze(0)
        return torch.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6)
    
class BitNetAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = BitNetLayer(self.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = BitNetLayer(self.hidden_size, self.num_key_value_heads * self.head_dim)
        self.v_proj = BitNetLayer(self.hidden_size, self.num_key_value_heads * self.head_dim)
        self.o_proj = BitNetLayer(self.num_heads * self.head_dim, self.hidden_size)

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_length, _ = hidden_states.size()
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        logger.debug(f"Query shape: {query_states.shape}")
        logger.debug(f"Key shape: {key_states.shape}")
        logger.debug(f"Value shape: {value_states.shape}")

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output

class BitNetMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = BitNetLayer(config.hidden_size, config.intermediate_size)
        self.down_proj = BitNetLayer(config.intermediate_size, config.hidden_size)
        self.up_proj = BitNetLayer(config.hidden_size, config.intermediate_size)
        self.act_fn = F.silu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class BitNetDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = BitNetAttention(config)
        self.mlp = BitNetMLP(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

class BitNetModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([BitNetDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.embed_tokens(input_ids)

        for i, layer in enumerate(self.layers):
            logger.debug(f"Processing layer {i}")
            logger.debug(f"Before layer: {hidden_states.shape}")
            hidden_states = layer(hidden_states, attention_mask)
            logger.debug(f"After layer: {hidden_states.shape}")

        hidden_states = self.norm(hidden_states)
        return hidden_states

class BitNetForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = BitNetModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.model(input_ids, attention_mask)
        logits = self.lm_head(hidden_states)
        return logits

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def load_llama_model(model_name):
    llama_model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return llama_model, tokenizer

def create_bitnet_model(llama_model):
    config = LlamaConfig.from_pretrained(llama_model.config._name_or_path)
    return BitNetForCausalLM(config)

def save_bitnet_model(model, tokenizer, save_directory):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    # Save the model state dict
    model_path = os.path.join(save_directory, "bitnet_model.pt")
    torch.save(model.state_dict(), model_path)
    logger.info(f"BitNet model saved to {model_path}")

    # Save the config
    config_path = os.path.join(save_directory, "config.json")
    model.model.config.to_json_file(config_path)
    logger.info(f"Config saved to {config_path}")

    # Save the tokenizer
    tokenizer.save_pretrained(save_directory)
    logger.info(f"Tokenizer saved to {save_directory}")

def main():
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    llama_model, tokenizer = load_llama_model(model_name)
    
    bitnet_model = create_bitnet_model(llama_model)
    
    # Save the BitNet model and tokenizer
    save_directory = "./bitnet_model_saved"
    save_bitnet_model(bitnet_model, tokenizer, save_directory)
    
    input_text = "Your input text here"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    logger.debug(f"Input IDs shape: {inputs['input_ids'].shape}")
    logger.debug(f"Attention mask shape: {inputs['attention_mask'].shape}")
    
    try:
        outputs = bitnet_model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        print(outputs.shape)
    except Exception as e:
        logger.error(f"Error during model inference: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()