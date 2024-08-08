import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import math
import os

class BitLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def absmean_quantize(self, W):
        gamma = torch.mean(torch.abs(W))
        return torch.round(torch.clamp(W / (gamma + 1e-5), -1, 1))

    def forward(self, input):
        quantized_weight = self.absmean_quantize(self.weight)
        return F.linear(input, quantized_weight)

def convert_to_bitnet(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = model.get_submodule(parent_name)
            
            bit_linear = BitLinear(module.in_features, module.out_features)
            bit_linear.weight.data = module.weight.data.clone().float()
            
            setattr(parent, child_name, bit_linear)
    return model

# Load and convert the model
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
config = AutoConfig.from_pretrained(model_name)
original_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, config=config)
bitnet_model = convert_to_bitnet(original_model)

# Convert the entire model to float32
bitnet_model = bitnet_model.float()

# Verify conversion
def verify_conversion(model):
    for name, module in model.named_modules():
        if isinstance(module, BitLinear):
            print(f"Converted: {name}")
        elif isinstance(module, nn.Linear):
            print(f"Not converted: {name}")

verify_conversion(bitnet_model)

# Save the converted model
output_dir = "./bitnet_model"
os.makedirs(output_dir, exist_ok=True)
bitnet_model.save_pretrained(output_dir)
print(f"Converted model saved to {output_dir}")

# Test the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set pad_token
input_text = "Hello, how are you?"
inputs = tokenizer(input_text, return_tensors="pt", padding=True)

# Ensure the model is in evaluation mode
bitnet_model.eval()

with torch.no_grad():
    output = bitnet_model(**inputs)

print("Model output shape:", output.logits.shape)

# Optional: Generate some text
generated = bitnet_model.generate(
    inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_length=50,
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    do_sample=True,
    temperature=0.7,
)
print("Generated text:", tokenizer.decode(generated[0], skip_special_tokens=True))