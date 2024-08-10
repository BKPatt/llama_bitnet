import torch
import os
from transformers import LlamaForCausalLM, AutoTokenizer
from BitNetB158Model import BitNetB158Model
from BitNetB158Config import BitNetB158Config
from AbsmeanQuantization import AbsmeanQuantization

def convert_llama_to_bitnet(model_name: str, save_directory: str):
    print(f"Loading LLaMA model: {model_name}")
    llama_model = LlamaForCausalLM.from_pretrained(model_name)
    llama_config = llama_model.config
    llama_tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Creating BitNet b1.58 configuration")
    bitnet_config = BitNetB158Config(
        vocab_size=llama_config.vocab_size,
        hidden_size=llama_config.hidden_size,
        intermediate_size=llama_config.intermediate_size,
        num_hidden_layers=llama_config.num_hidden_layers,
        num_attention_heads=llama_config.num_attention_heads,
        num_key_value_heads=llama_config.num_key_value_heads,
        max_position_embeddings=llama_config.max_position_embeddings,
        rms_norm_eps=llama_config.rms_norm_eps,
        rope_theta=llama_config.rope_theta,
        attention_bias=llama_config.attention_bias,
    )

    print("Creating BitNet b1.58 model")
    bitnet_model = BitNetB158Model(bitnet_config)

    print("Copying weights from LLaMA to BitNet b1.58 and quantizing")
    bitnet_model.embed_tokens.weight.data = llama_model.model.embed_tokens.weight.data

    for i, (llama_layer, bitnet_layer) in enumerate(zip(llama_model.model.layers, bitnet_model.layers)):
        print(f"Processing layer {i+1}/{len(llama_model.model.layers)}")

        # Copy and quantize attention weights
        for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            llama_weight = getattr(llama_layer.self_attn, proj).weight
            quantized_weight, weight_scale = AbsmeanQuantization.quantize(llama_weight)
            setattr(getattr(bitnet_layer.self_attn, proj), 'quantized_weight', quantized_weight)
            setattr(getattr(bitnet_layer.self_attn, proj), 'weight_scale', weight_scale)

        # Copy and quantize MLP weights
        for proj in ['gate_proj', 'up_proj', 'down_proj']:
            llama_weight = getattr(llama_layer.mlp, proj).weight
            quantized_weight, weight_scale = AbsmeanQuantization.quantize(llama_weight)
            setattr(getattr(bitnet_layer.mlp, proj), 'quantized_weight', quantized_weight)
            setattr(getattr(bitnet_layer.mlp, proj), 'weight_scale', weight_scale)

        # Copy layer norm weights
        bitnet_layer.input_layernorm.weight.data = llama_layer.input_layernorm.weight.data
        bitnet_layer.post_attention_layernorm.weight.data = llama_layer.post_attention_layernorm.weight.data

    bitnet_model.norm.weight.data = llama_model.model.norm.weight.data

    # Create the save directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)

    print("Saving the converted model")
    torch.save(bitnet_model.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))

    print("Saving the configuration")
    bitnet_config.to_json(os.path.join(save_directory, "config.json"))

    print("Saving the tokenizer")
    llama_tokenizer.save_pretrained(save_directory)

    print(f"Converted model, configuration, and tokenizer saved to {save_directory}")

    return bitnet_model, bitnet_config, llama_tokenizer

def main():
    model_name = "meta-llama/Llama-2-7b-chat-hf"  # Replace with the appropriate Llama 3 model name
    save_directory = "./bitnet_model_saved"
    
    convert_llama_to_bitnet(model_name, save_directory)

if __name__ == "__main__":
    main()