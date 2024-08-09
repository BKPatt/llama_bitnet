import torch
import os
import logging
from transformers import AutoTokenizer
from BitNetB158Model import BitNetB158Model
from BitNetB158Config import BitNetB158Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BitNetB158Pipeline:
    def __init__(self, model_path: str, tokenizer_path: str):
        logger.info("Initializing BitNetB158Pipeline...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info("Listing items in tokenizer path...")
        items = os.listdir(tokenizer_path)
        for item in items:
            logger.info(f"Found item: {item}")

        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        logger.info("Loading model configuration...")
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            self.config = BitNetB158Config.from_json(config_path)
            logger.info("Model configuration loaded successfully.")
        else:
            logger.error(f"Config file not found at {config_path}")
            raise FileNotFoundError(f"Config file not found at {config_path}")

        self.config.device = self.device  # Set the device in config

        logger.info("Loading model...")
        self.model = BitNetB158Model(self.config).to(self.device)
        model_file = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(model_file):
            self.model.load_state_dict(torch.load(model_file, map_location=self.device))
            logger.info("Model loaded successfully.")
        else:
            logger.error(f"Model file not found at {model_file}")
            raise FileNotFoundError(f"Model file not found at {model_file}")

        logger.info("Moving model to device...")
        self.model.to(self.device)
        self.model.eval()

        logger.info("Model initialization and quantization completed.")

    def generate(self, prompt: str, max_length: int = 50, temperature: float = 0.7, top_p: float = 0.9):
        logger.info("Encoding prompt...")
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        attention_mask = torch.ones_like(input_ids).to(self.device)

        logger.info("Clearing CUDA cache...")
        torch.cuda.empty_cache()

        logger.info(f"Memory before generation: allocated={torch.cuda.memory_allocated()} bytes, reserved={torch.cuda.memory_reserved()} bytes")
        
        logger.info("Generating text...")
        with torch.no_grad():
            for _ in range(max_length):
                logger.info(f"Memory before forward pass: allocated={torch.cuda.memory_allocated()} bytes, reserved={torch.cuda.memory_reserved()} bytes")
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logger.info(f"Memory after forward pass: allocated={torch.cuda.memory_allocated()} bytes, reserved={torch.cuda.memory_reserved()} bytes")
                next_token_logits = outputs["last_hidden_state"][:, -1, :] / temperature
                filtered_logits = top_p_filtering(next_token_logits, top_p=top_p)
                next_token = torch.multinomial(torch.softmax(filtered_logits, dim=-1), num_samples=1)

                input_ids = torch.cat([input_ids, next_token], dim=-1)
                attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)

                if next_token.item() == self.tokenizer.eos_token_id:
                    logger.info("End of sequence token detected.")
                    break

        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        logger.info(f"Generated text: {generated_text}")
        return generated_text

def top_p_filtering(logits, top_p=0.9, filter_value=-float('Inf')):
    logger.info("Applying top-p filtering...")
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
    logger.info("Starting main process...")

    logger.info("Initializing BitNetB158Pipeline...")
    pipeline = BitNetB158Pipeline(model_path, model_path)

    logger.info("Testing the model with a sample prompt...")
    prompt = "Once upon a time, in a land far away,"
    generated_text = pipeline.generate(prompt, max_length=50)  # Reduce max_length to 50

    logger.info(f"Prompt: {prompt}")
    logger.info(f"Generated text: {generated_text}")

if __name__ == "__main__":
    main()
