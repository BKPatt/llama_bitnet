import torch
import os
from transformers import AutoTokenizer
from BitNetB158Model import BitNetB158Model
from BitNetB158Config import BitNetB158Config
import torch.nn.functional as F

class BitNetB158Pipeline:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        print("Loading config...")
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            self.config = BitNetB158Config.from_json(config_path)
        else:
            raise FileNotFoundError(f"Config file not found at {config_path}")

        print("Loading model...")
        self.model = BitNetB158Model(self.config)
        model_file = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(model_file):
            self.model.load_state_dict(torch.load(model_file, map_location=self.device))
        else:
            raise FileNotFoundError(f"Model file not found at {model_file}")

        self.model.to(self.device)
        self.model.eval()

    def check_model_output(self, prompt: str):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids)
        logits = outputs[0][0, -1, :]
        top_k = torch.topk(logits, 10)
        print("Top 10 next token probabilities:")
        for value, index in zip(top_k.values, top_k.indices):
            token = self.tokenizer.decode([index])
            print(f"{token}: {value.item():.4f}")

    def check_model_parameters(self):
        print("Checking model parameters...")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(f"{name}: shape={param.shape}, mean={param.mean().item():.4f}, std={param.std().item():.4f}")
        
        print("\nChecking quantized weights...")
        for i, layer in enumerate(self.model.layers):
            for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                weight = getattr(layer.self_attn, proj).quantized_weight
                scale = getattr(layer.self_attn, proj).weight_scale
                print(f"Layer {i}, {proj}: weight shape={weight.shape}, mean={weight.float().mean().item():.4f}, std={weight.float().std().item():.4f}")
                print(f"Layer {i}, {proj} scale: shape={scale.shape}, mean={scale.mean().item():.4f}, std={scale.std().item():.4f}")
            
            for proj in ['gate_proj', 'up_proj', 'down_proj']:
                weight = getattr(layer.mlp, proj).quantized_weight
                scale = getattr(layer.mlp, proj).weight_scale
                print(f"Layer {i}, MLP {proj}: weight shape={weight.shape}, mean={weight.float().mean().item():.4f}, std={weight.float().std().item():.4f}")
                print(f"Layer {i}, MLP {proj} scale: shape={scale.shape}, mean={scale.mean().item():.4f}, std={scale.std().item():.4f}")
    
    def generate(self, prompt: str, max_length: int = 50, temperature: float = 0.7, top_k: int = 50):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        print(f"Input shape: {input_ids.shape}")
        print(f"Input tokens: {self.tokenizer.convert_ids_to_tokens(input_ids[0])}")
        
        for i in range(max_length):
            with torch.no_grad():
                outputs = self.model(input_ids)
            next_token_logits = outputs[0][0, -1, :]
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Apply top-k filtering
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
            
            # Apply softmax to top-k logits
            top_k_probs = F.softmax(top_k_logits, dim=-1)
            
            # Sample from the top-k distribution
            next_token_index = torch.multinomial(top_k_probs, num_samples=1)
            next_token = top_k_indices[next_token_index]
            
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
            
            print(f"Step {i + 1}: Generated token: {self.tokenizer.decode([next_token.item()])}")
            
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        print(f"Final input shape: {input_ids.shape}")
        print(f"Final tokens: {self.tokenizer.convert_ids_to_tokens(input_ids[0])}")
        return generated_text

def top_p_filtering(logits, top_p=0.9, filter_value=-float('Inf')):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = filter_value
    return logits

def main():
    model_path = "./bitnet_model_saved"

    print("Initializing BitNetB158Pipeline...")
    pipeline = BitNetB158Pipeline(model_path)

    print("Checking model parameters...")
    pipeline.check_model_parameters()

    print("Checking model output...")
    pipeline.check_model_output("Once upon a time")

    print("Testing the model with a sample prompt...")
    prompt = "Once upon a time, in a land far away,"
    generated_text = pipeline.generate(prompt, max_length=30)

    print(f"Prompt: {prompt}")
    print(f"Generated text: {generated_text}")

if __name__ == "__main__":
    main()