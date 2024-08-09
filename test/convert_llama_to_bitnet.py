import torch
import os
from transformers import LlamaForCausalLM, AutoTokenizer
from BitNetB158Model import BitNetB158Model
from BitNetB158Config import BitNetB158Config
from AbsmeanQuantization import AbsmeanQuantization

def convert_llama_to_bitnet():
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct" 
    save_directory = "./bitnet_model_saved"

    # Load the pre-trained LLaMA model and tokenizer
    llama_model = LlamaForCausalLM.from_pretrained(model_name)
    llama_config = llama_model.config
    llama_tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create BitNet b1.58 configuration
    bitnet_config = BitNetB158Config(
        vocab_size=llama_config.vocab_size,
        hidden_size=llama_config.hidden_size,
        intermediate_size=llama_config.intermediate_size,
        num_hidden_layers=llama_config.num_hidden_layers,  
        num_attention_heads=llama_config.num_attention_heads,
        max_position_embeddings=llama_config.max_position_embeddings,
        rms_norm_eps=llama_config.rms_norm_eps,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Create BitNet b1.58 model
    bitnet_model = BitNetB158Model(bitnet_config)
    
    # Copy and quantize weights from LLaMA to BitNet b1.58
    new_state_dict = {}

    # Embed tokens (keep as float)
    new_state_dict['embed_tokens.weight'] = llama_model.model.embed_tokens.weight.data

    for i, llama_layer in enumerate(llama_model.model.layers):
        # Attention weights
        new_state_dict[f'layers.{i}.self_attn.q_proj.quantized_weight'], new_state_dict[f'layers.{i}.self_attn.q_proj.weight_scale'] = AbsmeanQuantization.quantize(llama_layer.self_attn.q_proj.weight)
        new_state_dict[f'layers.{i}.self_attn.k_proj.quantized_weight'], new_state_dict[f'layers.{i}.self_attn.k_proj.weight_scale'] = AbsmeanQuantization.quantize(llama_layer.self_attn.k_proj.weight)
        new_state_dict[f'layers.{i}.self_attn.v_proj.quantized_weight'], new_state_dict[f'layers.{i}.self_attn.v_proj.weight_scale'] = AbsmeanQuantization.quantize(llama_layer.self_attn.v_proj.weight)
        new_state_dict[f'layers.{i}.self_attn.o_proj.quantized_weight'], new_state_dict[f'layers.{i}.self_attn.o_proj.weight_scale'] = AbsmeanQuantization.quantize(llama_layer.self_attn.o_proj.weight)

        # MLP weights
        new_state_dict[f'layers.{i}.mlp.gate_proj.quantized_weight'], new_state_dict[f'layers.{i}.mlp.gate_proj.weight_scale'] = AbsmeanQuantization.quantize(llama_layer.mlp.gate_proj.weight)
        new_state_dict[f'layers.{i}.mlp.up_proj.quantized_weight'], new_state_dict[f'layers.{i}.mlp.up_proj.weight_scale'] = AbsmeanQuantization.quantize(llama_layer.mlp.up_proj.weight)
        new_state_dict[f'layers.{i}.mlp.down_proj.quantized_weight'], new_state_dict[f'layers.{i}.mlp.down_proj.weight_scale'] = AbsmeanQuantization.quantize(llama_layer.mlp.down_proj.weight)
        
        # Layer norms (keep as float)
        new_state_dict[f'layers.{i}.input_layernorm.weight'] = llama_layer.input_layernorm.weight
        new_state_dict[f'layers.{i}.post_attention_layernorm.weight'] = llama_layer.post_attention_layernorm.weight
        
    # Final layer norm (keep as float)
    new_state_dict['norm.weight'] = llama_model.model.norm.weight
    
    # Load the new state dict
    bitnet_model.load_state_dict(new_state_dict)

    # Create the save directory if it doesn't exist  
    os.makedirs(save_directory, exist_ok=True)

    # Save the converted model
    torch.save(bitnet_model.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
    
    # Save the configuration
    bitnet_config.to_json(os.path.join(save_directory, "config.json"))

    # Save the tokenizer
    llama_tokenizer.save_pretrained(save_directory)

    print(f"Converted model, configuration, and tokenizer saved to {save_directory}")

def main():
    convert_llama_to_bitnet()

if __name__ == "__main__":
    main()
