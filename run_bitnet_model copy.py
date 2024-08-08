import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from Bitnet158Model_copy import BitNet

# Load the saved tokenizer
tokenizer = AutoTokenizer.from_pretrained('bitnet_tokenizer')

# Load the pre-trained model
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)

# Initialize the BitNet model
bitnet_model = BitNet(model)

# Load the saved model state dict
bitnet_model.load_state_dict(torch.load('bitnet_model.pth'))

# Define a function to generate text
def generate_text(prompt, max_length=100):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = bitnet_model(inputs)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Test the model
prompt = "Hello, how are you?"
generated_text = generate_text(prompt)
print(generated_text)