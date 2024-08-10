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
            state_dict = torch.load(model_file, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False)
        else:
            raise FileNotFoundError(f"Model file not found at {model_file}")

        self.model.to(self.device)
        self.model.eval()


    def generate(self, prompt: str, max_length: int = 100, temperature: float = 0.7, top_p: float = 0.9, top_k: int = 50):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        print(f"Input shape: {input_ids.shape}")
        print(f"Input tokens: {self.tokenizer.convert_ids_to_tokens(input_ids[0])}")
        
        attention_mask = torch.ones_like(input_ids)
        past_key_values = None

        for _ in range(max_length):
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids[:, -1:],
                    attention_mask=attention_mask[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True
                )
            logits = outputs[0][:, -1, :]  # Accessing the first element (last_hidden_state) from the tuple
            past_key_values = outputs[1]  # Accessing past_key_values from the tuple
            
            # Apply temperature
            logits = logits / temperature
            
            # Apply top-k filtering
            top_k_logits, top_k_indices = torch.topk(logits, top_k)
            
            # Apply top-p (nucleus) filtering
            top_k_probs = F.softmax(top_k_logits, dim=-1)
            cumulative_probs = torch.cumsum(top_k_probs, dim=-1)
            sorted_indices = torch.argsort(top_k_probs, descending=True)
            sorted_logits = top_k_logits.gather(-1, sorted_indices)
            cumulative_probs = cumulative_probs.gather(-1, sorted_indices)
            last_ind = (cumulative_probs < top_p).sum(dim=-1)
            last_ind[last_ind < 0] = 0
            sorted_logits[0, last_ind:] = float('-inf')
            sorted_indices = sorted_indices[0, :last_ind]

            # Sample from the filtered distribution
            probs = F.softmax(sorted_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            next_token = sorted_indices[next_token].squeeze(1)  # Flatten the next_token tensor

            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token.unsqueeze(0))], dim=-1)
            
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        print(f"Final input shape: {input_ids.shape}")
        print(f"Final tokens: {self.tokenizer.convert_ids_to_tokens(input_ids[0])}")
        return generated_text

    def check_model_output(self, prompt: str):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids)
        
        if isinstance(outputs, tuple):
            logits = outputs[0][0, -1, :]  # Accessing the last_hidden_state from the tuple
        else:
            logits = outputs["last_hidden_state"][0, -1, :]
        
        top_k = torch.topk(logits, 10)
        print("Top 10 next token probabilities:")
        for value, index in zip(top_k.values, top_k.indices):
            token = self.tokenizer.decode([index])
            print(f"{token}: {value.item():.4f}")

    def check_model_parameters(self):
        print("Checking model parameters...")
        total_params = 0
        for name, param in self.model.named_parameters():
            if 'quantized_weight' in name:
                print(f"{name}: shape={param.shape}, dtype={param.dtype}")
                total_params += param.numel() * 1.58 / 8  # 1.58 bits per parameter
            elif 'weight_scale' in name:
                print(f"{name}: shape={param.shape}, dtype={param.dtype}")
                total_params += param.numel() * 32 / 8  # 32 bits for float32
            else:
                print(f"{name}: shape={param.shape}, dtype={param.dtype}")
                total_params += param.numel() * 32 / 8  # 32 bits for float32
        
        print(f"\nTotal model size: {total_params / (1024**3):.2f} GB")

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
    generated_text = pipeline.generate(prompt, max_length=50)

    print(f"Prompt: {prompt}")
    print(f"Generated text: {generated_text}")

if __name__ == "__main__":
    main()
