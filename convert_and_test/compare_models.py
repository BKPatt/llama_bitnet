import torch
from transformers import LlamaForCausalLM, AutoTokenizer
from BitNetB158Model import BitNetB158Model

def generate_llama(model, tokenizer, prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def compare_models(llama_model, bitnet_llama, llama_tokenizer, prompts):
    for i, prompt in enumerate(prompts):
        print(f"\nPrompt {i+1}: {prompt}")
        
        llama_output = generate_llama(llama_model, llama_tokenizer, prompt)
        bitnet_output = generate_llama(bitnet_llama, llama_tokenizer, prompt)
        
        print("\nLLaMA 3.1 output:")
        print(llama_output)
        print("\nBitNet b1.58 output:")
        print(bitnet_output)

def main():
    # Load LLaMA 3.1 model
    llama_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    llama_model = LlamaForCausalLM.from_pretrained(llama_model_name)
    llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_name)

    # Load BitNet b1.58 model
    bitnet_model_path = "./bitnet_model_saved"
    bitnet_llama = BitNetB158Model.from_pretrained(bitnet_model_path)

    # Prepare a set of diverse prompts
    prompts = [
        "Explain the theory of relativity in simple terms.",
        "Write a short story about a time traveler.",
        "What are the main causes of climate change?",
        "Describe the process of photosynthesis in plants.",
        "How does artificial intelligence impact modern society?"
    ]

    # Compare the models
    compare_models(llama_model, bitnet_llama, llama_tokenizer, prompts)

if __name__ == "__main__":
    main()