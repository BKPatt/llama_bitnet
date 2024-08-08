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
    )

    # Create BitNet b1.58 model
    bitnet_model = BitNetB158Model(bitnet_config)

    # Copy weights from LLaMA to BitNet b1.58 and quantize
    bitnet_model.embed_tokens.weight.data = llama_model.model.embed_tokens.weight.data

    for i, llama_layer in enumerate(llama_model.model.layers):
        bitnet_layer = bitnet_model.layers[i]

        # Copy and quantize attention weights
        for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            llama_weight = getattr(llama_layer.self_attn, proj).weight
            quantized_weight, weight_scale = AbsmeanQuantization.quantize(llama_weight)
            getattr(bitnet_layer.self_attn, proj).quantized_weight.data = quantized_weight.float()
            getattr(bitnet_layer.self_attn, proj).weight_scale.data = weight_scale

        # Copy and quantize MLP weights
        for proj in ['gate_proj', 'up_proj', 'down_proj']:
            llama_weight = getattr(llama_layer.mlp, proj).weight
            quantized_weight, weight_scale = AbsmeanQuantization.quantize(llama_weight)
            getattr(bitnet_layer.mlp, proj).quantized_weight.data = quantized_weight.float()
            getattr(bitnet_layer.mlp, proj).weight_scale.data = weight_scale

        # Copy layer norm weights
        bitnet_layer.input_layernorm.weight.data = llama_layer.input_layernorm.weight.data
        bitnet_layer.post_attention_layernorm.weight.data = llama_layer.post_attention_layernorm.weight.data

    bitnet_model.norm.weight.data = llama_model.model.norm.weight.data

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