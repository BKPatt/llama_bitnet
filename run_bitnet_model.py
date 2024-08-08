import torch
from transformers import AutoTokenizer, LlamaConfig
from Bitnet158Model import BitNetForCausalLM
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_bitnet_model(model_dir):
    config_path = os.path.join(model_dir, "config.json")
    model_path = os.path.join(model_dir, "bitnet_model.pt")
    
    if not os.path.exists(config_path) or not os.path.exists(model_path):
        raise FileNotFoundError(f"Model files not found in {model_dir}")
    
    config = LlamaConfig.from_json_file(config_path)
    model = BitNetForCausalLM(config)
    model.load_state_dict(torch.load(model_path))
    logger.info(f"BitNet model loaded from {model_dir}")
    return model

def generate_text(model, tokenizer, prompt, max_length=50, temperature=0.7, top_k=50, top_p=0.95):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    attention_mask = torch.ones_like(input_ids)
    
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids, attention_mask=attention_mask)
            next_token_logits = outputs[:, -1, :]
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Apply top-k filtering
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
            next_token_logits[0, :] = torch.where(next_token_logits[0, :] < top_k_logits[0, -1], torch.tensor(-float('Inf')), next_token_logits[0, :])
            
            # Apply top-p (nucleus) filtering
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_token_logits[0, indices_to_remove[0]] = -float('Inf')
            
            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append next token to input_ids and attention_mask
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

def main():
    # Load the saved BitNet model
    model_dir = "./bitnet_model_saved"
    bitnet_model = load_bitnet_model(model_dir)
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # Test prompts
    prompts = [
        "Once upon a time, in a land far away,",
        "The secret to a happy life is",
        "In the year 2050, technology has advanced to the point where"
    ]
    
    logger.info("Starting inference tests...")
    
    for i, prompt in enumerate(prompts):
        logger.info(f"Test {i+1}: Generating text for prompt: '{prompt}'")
        
        try:
            generated_text = generate_text(bitnet_model, tokenizer, prompt)
            logger.info(f"Generated text: {generated_text}")
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    logger.info("Inference tests completed.")

if __name__ == "__main__":
    main()