import torch
from transformers import AutoTokenizer
from Bitnet158Model import Bitnet158Model  # Make sure this import works with your file structure
import json

def load_config(config_path):
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    return type('Config', (), config_dict)()

def load_bitnet_model(model_path, config_path):
    config = load_config(config_path)
    model = Bitnet158Model(config)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def generate_text(model, tokenizer, prompt, max_length=50):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
    
    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(input_ids)
        
        next_token_logits = outputs[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
        
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        attention_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=torch.long)], dim=-1)
        
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

def main():
    model_path = 'bitnet1_58_weights.pth'
    config_path = 'llama3_8b_config.json'
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    
    print("Loading Bitnet model...")
    model = load_bitnet_model(model_path, config_path)
    print("Model loaded successfully.")
    
    prompt = "Once upon a time, in a faraway land,"
    print(f"Generating text for prompt: '{prompt}'")
    
    generated_text = generate_text(model, tokenizer, prompt)
    print("Generated text:")
    print(generated_text)

if __name__ == "__main__":
    main()