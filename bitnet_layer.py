from transformers import AutoModelForCausalLM
import torch
import torch.nn as nn

from Bitnet158Model_copy import BitLinear

def convert_to_bitnet(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            bit_linear = BitLinear(module.in_features, module.out_features)
            bit_linear.weight.data = module.weight.data.clone()
            setattr(module.parent, name.split('.')[-1], bit_linear)
    return model

# Load and convert the model
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
original_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
bitnet_model = convert_to_bitnet(original_model)

# Verify conversion
def verify_conversion(model):
    for name, module in model.named_modules():
        if isinstance(module, BitLinear):
            print(f"Converted: {name}")
        elif isinstance(module, nn.Linear):
            print(f"Not converted: {name}")

verify_conversion(bitnet_model)