import torch
from transformers import LlamaForCausalLM, LlamaConfig

# Step 3.1: Install Necessary Libraries
# !pip install torch llama3

# Step 3.2: Load the Llama 3 8b Model
llama_model = LlamaForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

# Step 3.3: Extract Model Weights
llama_weights = llama_model.state_dict()

# Step 3.4: Save Model Weights
torch.save(llama_weights, 'llama3_8b_weights.pth')

# Step 3.5: Extract Model Configuration
llama_config = llama_model.config
with open('llama3_8b_config.json', 'w') as f:
    f.write(llama_config.to_json_string())

# Step 3.6: Verification
# Load the saved weights
loaded_weights = torch.load('llama3_8b_weights.pth')

# Verify the weights
assert loaded_weights.keys() == llama_weights.keys()
for key in loaded_weights.keys():
    assert torch.equal(loaded_weights[key], llama_weights[key])

# Load the saved configuration
with open('llama3_8b_config.json', 'r') as f:
    loaded_config = f.read()

# Verify the configuration
assert loaded_config == llama_config.to_json_string()
