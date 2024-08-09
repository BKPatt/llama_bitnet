import torch
from transformers import LlamaForCausalLM

from BitNetB158Config import BitNetB158Config
from BitNetB158Model import BitNetB158Model

def convert_llama_to_bitnet(llama_model_path, bitnet_config_path, save_path):
    llama_model = LlamaForCausalLM.from_pretrained(llama_model_path)
    bitnet_config = BitNetB158Config.from_json_file(bitnet_config_path)
    bitnet_model = BitNetB158Model(bitnet_config)

    llama_state_dict = llama_model.state_dict()
    bitnet_state_dict = bitnet_model.state_dict()

    # Use the updated mapping function
    new_state_dict = map_llama_to_bitnet(llama_state_dict, bitnet_state_dict)
    
    # Load the mapped state dict into the BitNet model
    bitnet_model.load_state_dict(new_state_dict, strict=False)

    torch.save(bitnet_model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def map_llama_to_bitnet(llama_state_dict, bitnet_state_dict):
    new_state_dict = {}

    for name, param in llama_state_dict.items():
        # Embedding layer
        if "embed_tokens.weight" in name:
            new_name = "embed_tokens.weight"
            if new_name in bitnet_state_dict:
                new_state_dict[new_name] = param
            else:
                print(f"Skipping {name} as it does not exist in the BitNet model")
        
        # Handle combined q, k, v projections in one layer
        elif "self_attn.q_proj.weight" in name:
            base_name = name.replace("self_attn.q_proj.weight", "")
            try:
                # Assume BitNet combines q, k, v projections into one layer
                q_proj = param
                k_proj = llama_state_dict[f"{base_name}self_attn.k_proj.weight"]
                v_proj = llama_state_dict[f"{base_name}self_attn.v_proj.weight"]
                combined_proj = torch.cat([q_proj, k_proj, v_proj], dim=0)
                
                if f"{base_name}self_attn.proj.weight" in bitnet_state_dict:
                    new_state_dict[f"{base_name}self_attn.proj.weight"] = combined_proj
                else:
                    print(f"Skipping combined projection for {name} as it does not match any BitNet layer")
                
            except KeyError as e:
                print(f"Skipping {name} due to missing component: {e}")
        
        # MLP and LayerNorm components
        elif any(key in name for key in ["mlp", "input_layernorm", "post_attention_layernorm", "norm"]):
            if name in bitnet_state_dict:
                if param.shape == bitnet_state_dict[name].shape:
                    new_state_dict[name] = param
                else:
                    print(f"Skipping {name} due to shape mismatch: expected {bitnet_state_dict[name].shape}, got {param.shape}")
            else:
                print(f"Skipping {name} as it does not exist in the BitNet model")
        
        else:
            print(f"Skipping {name} as it does not match any BitNet layer")
    
    return new_state_dict

def main():
    llama_model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
    bitnet_config_path = "c:/Users/brant/Desktop/gemmaft/test/config.json"
    save_path = "./bitnet_model.pth"
    
    convert_llama_to_bitnet(llama_model_path, bitnet_config_path, save_path)

if __name__ == "__main__":
    main()
