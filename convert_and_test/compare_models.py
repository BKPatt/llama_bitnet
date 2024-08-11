import torch
from transformers import LlamaForCausalLM, AutoTokenizer
from BitNetB158Model import BitNetB158Model

def generate_llama(model, tokenizer, prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        if isinstance(model, BitNetB158Model):
            generated_text = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_length
            )
        else:
            outputs = model.generate(**inputs, max_length=max_length)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def compare_models(llama_model, bitnet_llama, llama_tokenizer, prompts):
    for i, prompt in enumerate(prompts):
        print(f"\nPrompt {i+1}: {prompt}")
        
        # llama_output = generate_llama(llama_model, llama_tokenizer, prompt)
        bitnet_output = generate_llama(bitnet_llama, llama_tokenizer, prompt)
        
        # print("\nLLaMA 3.1 output:")
        # print(llama_output)
        print("\nBitNet b1.58 output:")
        print(bitnet_output)

def main():
    llama_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    # llama_model = LlamaForCausalLM.from_pretrained(llama_model_name)
    llama_model = 0
    llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_name)

    bitnet_model_path = "./bitnet_model_saved"
    bitnet_llama = LlamaForCausalLM.from_pretrained(bitnet_model_path)

    prompts = [
        "Explain the theory of relativity in simple terms.",
        "Write a short story about a time traveler.",
        "What are the main causes of climate change?",
        "Describe the process of photosynthesis in plants.",
        "How does artificial intelligence impact modern society?"
    ]

    compare_models(llama_model, bitnet_llama, llama_tokenizer, prompts)

if __name__ == "__main__":
    main()