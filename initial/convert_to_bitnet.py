import torch
from transformers import LlamaForCausalLM, LlamaConfig
from BitNetLlamaModel import BitNetLlamaModel

def convert_to_bitnet(model_name_or_path):
    # Load the original FP16 Llama model
    fp16_model = LlamaForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16)
    config = fp16_model.config

    # Create a new BitNetLlamaModel with the same configuration
    bitnet_model = BitNetLlamaModel(config)

    bitnet_model.convert_from_fp16(fp16_model)
    bitnet_model.quantize_weights()

    # Save the converted model
    torch.save(bitnet_model.state_dict(), "bitnet_llama_model.pth")
    config.save_pretrained("bitnet_llama_model")

    print("Conversion completed. BitNet model saved to 'bitnet_llama_model.pth' and config saved to 'bitnet_llama_model' directory.")

if __name__ == "__main__":
    model_name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"
    convert_to_bitnet(model_name_or_path)