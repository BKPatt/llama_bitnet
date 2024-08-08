import torch
import os
from transformers import AutoTokenizer
from BitNetB158Model import BitNetB158Model
from BitNetB158Config import BitNetB158Config

class BitNetB158Pipeline:
    def __init__(self, model_path: str, tokenizer_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        items = os.listdir(tokenizer_path)
        for item in items:
            print(item)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            self.config = BitNetB158Config.from_json(config_path)
        else:
            raise FileNotFoundError(f"Config file not found at {config_path}")

        self.model = BitNetB158Model(self.config)
        model_file = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(model_file):
            self.model.load_state_dict(torch.load(model_file, map_location=self.device))
        else:
            raise FileNotFoundError(f"Model file not found at {model_file}")

        self.model.to(self.device)
        self.model.eval()

        # Quantize the model
        self.model.quantize()

    def generate(self, prompt: str, max_length: int = 50, temperature: float = 0.7, top_p: float = 0.9):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.model(input_ids, attention_mask=attention_mask)
                next_token_logits = outputs["last_hidden_state"][:, -1, :] / temperature
                filtered_logits = top_p_filtering(next_token_logits, top_p=top_p)
                next_token = torch.multinomial(torch.softmax(filtered_logits, dim=-1), num_samples=1)

                input_ids = torch.cat([input_ids, next_token], dim=-1)
                attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

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
    pipeline = BitNetB158Pipeline(model_path, model_path)

    print("Testing the model with a sample prompt...")
    prompt = "Once upon a time, in a land far away,"
    generated_text = pipeline.generate(prompt, max_length=100)

    print(f"Prompt: {prompt}")
    print(f"Generated text: {generated_text}")

if __name__ == "__main__":
    main()
