import torch
from transformers import AutoTokenizer, LlamaConfig
import json
from Bitnet158Model import Bitnet158Model  # Make sure this import matches your file structure
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    return LlamaConfig(**config_dict)

def load_bitnet_model(model_path, config_path):
    logger.info("Loading Bitnet model configuration...")
    config = load_config(config_path)
    
    logger.info("Initializing Bitnet model...")
    model = Bitnet158Model(config)
    
    logger.info("Loading model weights...")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def generate_text(model, tokenizer, prompt, max_length=50, temperature=0.7, top_k=50, top_p=0.95):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    logger.info(f"Generating text for prompt: '{prompt}'")
    
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            next_token_logits = outputs[:, -1, :] / temperature
            
            # Apply top-k filtering
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
            
            # Apply top-p (nucleus) filtering
            cumulative_probs = torch.cumsum(torch.softmax(top_k_logits, dim=-1), dim=-1)
            top_p_mask = cumulative_probs < top_p
            top_p_mask[..., -1] = True  # Always keep the last token to ensure we have at least one to sample from
            filtered_logits = top_k_logits * top_p_mask
            
            # Sample from the filtered distribution
            probs = torch.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Map back to the original token space
            next_token = top_k_indices.gather(-1, next_token)
            
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    logger.info("Text generation completed.")
    return generated_text

def main():
    model_path = 'bitnet1_58_weights.pth'
    config_path = 'llama3_8b_config.json'
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    
    logger.info("Loading Bitnet model...")
    model = load_bitnet_model(model_path, config_path)
    logger.info("Model loaded successfully.")
    
    prompt = "Once upon a time, in a faraway land,"
    generated_text = generate_text(model, tokenizer, prompt)
    
    print("\nGenerated text:")
    print(generated_text)

if __name__ == "__main__":
    main()